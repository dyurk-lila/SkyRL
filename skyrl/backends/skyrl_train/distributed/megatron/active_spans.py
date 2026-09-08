from dataclasses import dataclass
from typing import Optional

import torch

from skyrl.backends.skyrl_train.distributed.megatron.packing_utils import (
    get_packed_seq_align_size,
)

ActiveSpans = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ActiveSpanMetadata:
    active_mask: torch.Tensor
    active_spans: ActiveSpans


# H100 measurements favored coalescing exact runs across short inactive gaps.
# Keep this an internal policy until broader model/GPU coverage justifies changing it.
BLOCK_SPARSE_DHIDDEN_MAX_INACTIVE_GAP = 128


def _require_cpu(name: str, tensor: torch.Tensor) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must remain on CPU while building active spans")


def _coalesced_active_spans(active_mask: torch.Tensor) -> ActiveSpans:
    """Return ordered spans while retaining gaps of at most 128 inactive rows."""
    _require_cpu("active_mask", active_mask)
    # Keep all row-scale work in tensor operations. Only the O(number of
    # coalesced spans) boundaries cross into Python for Triton's launch list.
    active_indices = torch.nonzero(active_mask.reshape(-1), as_tuple=False).reshape(-1)
    if active_indices.numel() == 0:
        return ()

    breaks = torch.nonzero(
        active_indices[1:] - active_indices[:-1] > BLOCK_SPARSE_DHIDDEN_MAX_INACTIVE_GAP + 1,
        as_tuple=False,
    ).reshape(-1)
    starts = torch.cat((active_indices[:1], active_indices[breaks + 1])).tolist()
    ends = torch.cat((active_indices[breaks] + 1, active_indices[-1:] + 1)).tolist()
    return tuple(zip(starts, ends))


def _build_active_span_metadata(active_mask: torch.Tensor, local_active_mask: torch.Tensor) -> ActiveSpanMetadata:
    """Pair the forward mask with spans proven to cover its CP-local rows."""
    _require_cpu("active_mask", active_mask)
    _require_cpu("local_active_mask", local_active_mask)
    local_active_mask = local_active_mask.reshape(-1)
    active_spans = _coalesced_active_spans(local_active_mask)
    covered = torch.zeros_like(local_active_mask)
    # Response masks are blocky, so this loop is over a few coalesced spans;
    # expanding it into per-row Python work would dominate metadata setup.
    for start, end in active_spans:
        covered[start:end] = True
    if torch.any(local_active_mask & ~covered):
        raise AssertionError("active spans do not cover every active forward row")
    return ActiveSpanMetadata(active_mask=active_mask, active_spans=active_spans)


def build_active_mask(loss_mask: torch.Tensor, num_actions: int, sequence_length: int) -> torch.Tensor:
    """Expand an action loss mask into prediction-row coordinates."""
    if loss_mask.dim() != 2 or loss_mask.shape[1] != num_actions:
        raise ValueError(f"Expected loss_mask shape [batch, {num_actions}], got {tuple(loss_mask.shape)}")
    if num_actions > sequence_length - 1:
        raise ValueError(f"num_actions={num_actions} exceeds the {sequence_length - 1} prediction rows")

    active_mask = torch.zeros((loss_mask.shape[0], sequence_length - 1), dtype=torch.bool, device=loss_mask.device)
    if num_actions > 0:
        active_mask[:, -num_actions:] = loss_mask > 0
    return active_mask


def build_unpacked_active_metadata(
    loss_mask: torch.Tensor,
    num_actions: int,
    sequence_length: int,
) -> ActiveSpanMetadata:
    """Build one action-loss mask and its de-padded ``[batch, sequence]`` spans."""
    _require_cpu("loss_mask", loss_mask)
    prediction_mask = build_active_mask(loss_mask, num_actions, sequence_length)
    # The rolled target's final row never predicts a token.
    local_active_mask = torch.nn.functional.pad(prediction_mask, (0, 1), value=False)
    return _build_active_span_metadata(prediction_mask, local_active_mask)


def _cp_local_segment(segment: torch.Tensor, cp_size: int, cp_rank: int) -> torch.Tensor:
    if cp_size <= 0 or not 0 <= cp_rank < cp_size:
        raise ValueError(f"Expected 0 <= cp_rank < cp_size, got rank={cp_rank}, size={cp_size}")
    if cp_size == 1:
        return segment
    if segment.numel() % (2 * cp_size) != 0:
        raise ValueError(f"Packed segment length {segment.numel()} is not divisible by 2 * CP={2 * cp_size}")

    chunk_size = segment.numel() // (2 * cp_size)
    mirrored_chunk = 2 * cp_size - cp_rank - 1
    return torch.cat(
        (
            segment[cp_rank * chunk_size : (cp_rank + 1) * chunk_size],
            segment[mirrored_chunk * chunk_size : (mirrored_chunk + 1) * chunk_size],
        )
    )


def build_packed_active_metadata(
    loss_mask: torch.Tensor,
    num_actions: int,
    sequence_length: int,
    attention_mask: torch.Tensor,
    *,
    sub_seq_lengths: Optional[list[list[int]]],
    tp_size: int,
    cp_size: int,
    cp_rank: int,
    fp8_enabled: bool,
) -> ActiveSpanMetadata:
    """Build one packed action-loss mask and its exact CP-local spans."""
    _require_cpu("loss_mask", loss_mask)
    _require_cpu("attention_mask", attention_mask)
    if attention_mask.shape != (loss_mask.shape[0], sequence_length):
        raise ValueError(
            f"Expected attention_mask shape {(loss_mask.shape[0], sequence_length)}, "
            f"got {tuple(attention_mask.shape)}"
        )

    prediction_mask = build_active_mask(loss_mask, num_actions, sequence_length)
    align_size = get_packed_seq_align_size(tp_size, cp_size, fp8_enabled=fp8_enabled)
    packed_segments = []
    local_segments = []

    if sub_seq_lengths is not None:
        if len(sub_seq_lengths) != loss_mask.shape[0]:
            raise ValueError(f"sub_seq_lengths has {len(sub_seq_lengths)} rows but loss_mask has {loss_mask.shape[0]}")
        for row_index, row_lengths in enumerate(sub_seq_lengths):
            row_offset = 0
            for raw_length in row_lengths:
                seq_length = int(raw_length)
                if seq_length <= 0:
                    raise ValueError(f"Packed sub-sequence lengths must be positive, got {seq_length}")
                padded_length = ((seq_length + align_size - 1) // align_size) * align_size
                if row_offset + seq_length > sequence_length:
                    raise ValueError("sub_seq_lengths extends beyond the packed batch row")
                segment = torch.zeros(padded_length, dtype=torch.bool)
                if seq_length > 1:
                    segment[: seq_length - 1] = prediction_mask[row_index, row_offset : row_offset + seq_length - 1]
                packed_segments.append(segment)
                local_segments.append(_cp_local_segment(segment, cp_size, cp_rank))
                row_offset += padded_length
    else:
        attention_mask = attention_mask.to(dtype=torch.bool)
        for row_index in range(attention_mask.shape[0]):
            token_columns = torch.nonzero(attention_mask[row_index], as_tuple=False).reshape(-1)
            seq_length = token_columns.numel()
            if seq_length <= 0:
                raise ValueError("Packed sequences must contain at least one attended token per row")
            padded_length = ((seq_length + align_size - 1) // align_size) * align_size
            segment = torch.zeros(padded_length, dtype=torch.bool)
            if seq_length > 1:
                segment[: seq_length - 1] = prediction_mask[row_index, token_columns[:-1]]
            packed_segments.append(segment)
            local_segments.append(_cp_local_segment(segment, cp_size, cp_rank))

    if not packed_segments:
        return ActiveSpanMetadata(active_mask=torch.zeros((1, 0), dtype=torch.bool), active_spans=())
    packed_active_mask = torch.cat(packed_segments).unsqueeze(0)
    return _build_active_span_metadata(packed_active_mask, torch.cat(local_segments))
