"""Support-conditioned scoring for bounded sampler replay."""

from dataclasses import dataclass

import torch

from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import (
    TokenMetadataLayout,
    align_token_metadata,
    scatter_packed_token_values_to_batch,
)
from skyrl.backends.skyrl_train.utils.packed_ragged_tensor import PackedRaggedTensor
from skyrl.backends.skyrl_train.utils.packed_tensor import PackedTensor
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_FIELD,
    SAMPLE_SUPPORT_NO_ROW,
    SAMPLE_SUPPORT_PADDING,
    SAMPLE_SUPPORT_TORCH_DTYPE,
    PackedSampleSupport,
    align_sample_support_row_ids,
)


def missing_sample_support_message(backend: str) -> str:
    """Describe how to enable and forward a missing support payload."""
    return (
        f"sample-support replay is enabled but the {backend} forward received no "
        f"{SAMPLE_SUPPORT_FIELD!r}. Set generator.inference_engine.enable_return_sample_support_set=true "
        "so the sampler records it; if it is already set, the generator is dropping the field before "
        "the trainer sees it -- SkyRLGymGenerator forwards it, and a custom generator must too."
    )


@dataclass(frozen=True)
class SampleSupportScores:
    """Support-conditioned scores and the rows backed by recorded support."""

    logprobs: torch.Tensor
    entropy: torch.Tensor | None
    valid_mask: torch.Tensor


def reject_unsupported_sample_support_packing(sub_seq_lengths: list[list[int]] | None) -> None:
    """Require every packed row to contain one whole trajectory."""
    if sub_seq_lengths is None:
        return
    if any(len(row_lengths) > 1 for row_lengths in sub_seq_lengths):
        raise ValueError(
            "sample-support replay does not support controller-packed multi-subsequence rows, got "
            f"sub-sequence counts {[len(row_lengths) for row_lengths in sub_seq_lengths]}"
        )


def _project_pair_chunk(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    row_ids: torch.Tensor,
    token_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Score one bounded block of ``(token position, vocab row)`` candidate pairs."""
    selected_hidden = hidden.index_select(0, row_ids).to(weight.dtype)
    selected_weight = weight.index_select(0, token_ids)
    return (selected_hidden * selected_weight).sum(dim=-1) / temperature


class _ChunkedCandidateProjection(torch.autograd.Function):
    """Selected LM-head projection that never retains a ``[pairs, hidden]`` activation."""

    @staticmethod
    def forward(ctx, hidden, weight, row_ids, token_ids, temperature, chunk_size):
        ctx.save_for_backward(hidden, weight, row_ids, token_ids)
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        output = torch.empty(token_ids.shape, dtype=torch.float32, device=hidden.device)
        for start in range(0, token_ids.numel(), chunk_size):
            end = min(start + chunk_size, token_ids.numel())
            projected = _project_pair_chunk(hidden, weight, row_ids[start:end], token_ids[start:end], temperature)
            output[start:end] = projected.to(torch.float32)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, row_ids, token_ids = ctx.saved_tensors
        grad_hidden = torch.zeros_like(hidden) if ctx.needs_input_grad[0] else None
        grad_weight = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
        for start in range(0, token_ids.numel(), ctx.chunk_size):
            end = min(start + ctx.chunk_size, token_ids.numel())
            chunk_rows = row_ids[start:end]
            chunk_tokens = token_ids[start:end]
            chunk_grad = grad_output[start:end].to(weight.dtype) / ctx.temperature
            if grad_hidden is not None:
                hidden_contribution = chunk_grad.unsqueeze(1) * weight.index_select(0, chunk_tokens)
                grad_hidden.index_add_(0, chunk_rows, hidden_contribution.to(hidden.dtype))
            if grad_weight is not None:
                selected_hidden = hidden.index_select(0, chunk_rows).to(weight.dtype)
                grad_weight.index_add_(0, chunk_tokens, chunk_grad.unsqueeze(1) * selected_hidden)
        return grad_hidden, grad_weight, None, None, None, None


def _project_candidate_pairs(
    hidden: torch.Tensor,
    row_ids: torch.Tensor,
    token_ids: torch.Tensor,
    lm_head_weight: torch.Tensor,
    temperature: float,
    chunk_size: int | None,
) -> torch.Tensor:
    """Project selected ``(token position, vocab row)`` pairs in bounded chunks."""
    if token_ids.numel() == 0:
        return torch.empty(0, dtype=torch.float32, device=hidden.device)
    pair_chunk_size = token_ids.numel() if chunk_size is None else chunk_size
    if pair_chunk_size <= 0:
        raise ValueError(f"candidate projection chunk size must be positive, got {pair_chunk_size}")
    if torch.is_grad_enabled() and (hidden.requires_grad or lm_head_weight.requires_grad):
        # A plain loop retains every selected chunk for backward. The custom backward reselects
        # one chunk at a time, so peak candidate-pair storage stays at the chunk bound.
        return _ChunkedCandidateProjection.apply(
            hidden,
            lm_head_weight,
            row_ids,
            token_ids,
            temperature,
            pair_chunk_size,
        )

    output = torch.empty(token_ids.shape, dtype=torch.float32, device=hidden.device)
    for start in range(0, token_ids.numel(), pair_chunk_size):
        end = min(start + pair_chunk_size, token_ids.numel())
        projected = _project_pair_chunk(hidden, lm_head_weight, row_ids[start:end], token_ids[start:end], temperature)
        output[start:end] = projected.to(torch.float32)
    return output


def _selected_hidden_projection(
    hidden: torch.Tensor,
    token_ids: torch.Tensor,
    local_mask: torch.Tensor,
    lm_head_weight: torch.Tensor,
    temperature: float,
    chunk_size: int | None,
    invalid_value: float,
) -> torch.Tensor:
    """Score a fixed-width candidate matrix without materializing vocabulary logits."""
    num_rows, width = token_ids.shape
    row_ids = torch.arange(num_rows, device=hidden.device).unsqueeze(1).expand(-1, width).reshape(-1)
    projected = _project_candidate_pairs(
        hidden,
        row_ids,
        token_ids.reshape(-1),
        lm_head_weight,
        temperature,
        chunk_size,
    )
    return torch.where(local_mask.reshape(-1), projected, invalid_value).reshape(num_rows, width)


def sample_support_scores(
    logits_or_hidden: torch.Tensor,
    sampled_ids: torch.Tensor,
    support_ids: torch.Tensor,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group: torch.distributed.ProcessGroup | None,
    compute_entropy: bool,
    entropy_requires_grad: bool,
    lm_head_weight: torch.Tensor | None = None,
    temperature: float = 1.0,
    chunk_size: int | None = None,
) -> SampleSupportScores:
    """Renormalize scores and optionally compute entropy over each recorded support row."""
    if logits_or_hidden.shape[:-1] != sampled_ids.shape or support_ids.shape[:-1] != sampled_ids.shape:
        raise ValueError(
            "logits, sampled_ids, and support_ids must have matching prefix shapes, got "
            f"{logits_or_hidden.shape[:-1]}, {sampled_ids.shape}, and {support_ids.shape[:-1]}"
        )
    if support_ids.dtype != SAMPLE_SUPPORT_TORCH_DTYPE:
        raise ValueError(f"sample support must use {SAMPLE_SUPPORT_TORCH_DTYPE} vocab ids, got {support_ids.dtype}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if entropy_requires_grad and not compute_entropy:
        raise ValueError("entropy gradients require compute_entropy=True")

    flat_source = logits_or_hidden.reshape(-1, logits_or_hidden.shape[-1])
    flat_sampled = sampled_ids.reshape(-1).long()
    flat_support = support_ids.reshape(-1, support_ids.shape[-1]).long()
    valid_members = flat_support >= 0
    valid_rows = valid_members.any(dim=-1)
    local_members = valid_members & (flat_support >= vocab_start_index) & (flat_support < vocab_end_index)
    local_support_ids = (flat_support - vocab_start_index).clamp(0, vocab_end_index - vocab_start_index - 1)
    local_sample_mask = (flat_sampled >= vocab_start_index) & (flat_sampled < vocab_end_index)
    local_sample_ids = (flat_sampled - vocab_start_index).clamp(0, vocab_end_index - vocab_start_index - 1)

    compute_dtype = (
        torch.float32 if logits_or_hidden.dtype in (torch.float16, torch.bfloat16) else logits_or_hidden.dtype
    )
    if lm_head_weight is None:
        local_values = flat_source.gather(1, local_support_ids).to(compute_dtype)
        local_values = torch.where(local_members, local_values, float("-inf"))
        local_sampled = flat_source.gather(1, local_sample_ids.unsqueeze(1)).squeeze(1).to(compute_dtype)
        local_sampled = torch.where(local_sample_mask, local_sampled, 0.0)
    else:
        if lm_head_weight.shape[0] != vocab_end_index - vocab_start_index:
            raise ValueError(
                f"lm_head_weight holds {lm_head_weight.shape[0]} rows for vocabulary shard "
                f"[{vocab_start_index}, {vocab_end_index})"
            )
        local_values = _selected_hidden_projection(
            flat_source,
            local_support_ids,
            local_members,
            lm_head_weight,
            temperature,
            chunk_size,
            float("-inf"),
        )
        local_sampled = _selected_hidden_projection(
            flat_source,
            local_sample_ids.unsqueeze(1),
            local_sample_mask.unsqueeze(1),
            lm_head_weight,
            temperature,
            chunk_size,
            0.0,
        ).squeeze(1)

    local_max = local_values.detach().amax(dim=-1)
    global_max = local_max.clone()
    if tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
        torch.distributed.all_reduce(global_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
    safe_max = torch.where(valid_rows, global_max, 0.0)

    local_exp = torch.where(local_members, (local_values - safe_max.unsqueeze(1)).exp(), 0.0)
    local_rows = [local_exp.sum(dim=-1), local_sampled]
    if compute_entropy:
        # H = log(Z) - E_p[l - max] over the support.
        shifted_values = local_values if entropy_requires_grad else local_values.detach()
        weights = local_exp if entropy_requires_grad else local_exp.detach()
        local_rows.append((weights * torch.where(local_members, shifted_values - safe_max.unsqueeze(1), 0.0)).sum(-1))
    # Denominator, numerator, and entropy share one SUM collective.
    local_stats = torch.stack(local_rows)
    global_stats = local_stats.detach().clone()
    if tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
        torch.distributed.all_reduce(global_stats, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    global_stats = global_stats + local_stats - local_stats.detach()
    denominator, sampled_score = global_stats[0], global_stats[1]
    logprobs = sampled_score - safe_max - torch.where(valid_rows, denominator, 1.0).log()

    entropy = None
    if compute_entropy:
        weighted_sum = global_stats[2] if entropy_requires_grad else global_stats[2].detach()
        safe_denominator = torch.where(valid_rows, denominator if entropy_requires_grad else denominator.detach(), 1.0)
        entropy = torch.where(valid_rows, safe_denominator.log() - weighted_sum / safe_denominator, 0.0)
        entropy = entropy.reshape(sampled_ids.shape)
    return SampleSupportScores(
        logprobs=torch.where(valid_rows, logprobs, 0.0).reshape(sampled_ids.shape),
        entropy=entropy,
        valid_mask=valid_rows.reshape(sampled_ids.shape),
    )


def _invert_row_ids(row_ids: torch.Tensor, num_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Map support rows to model positions and reject duplicate row claims."""
    flat_rows = row_ids.reshape(-1).long()
    named = (flat_rows >= 0) & (flat_rows < num_rows)
    # One slot past the last row absorbs every position that names none, so the scatters below
    # stay in range without a separate mask.
    safe_rows = torch.where(named, flat_rows, num_rows)
    row_position = torch.full((num_rows + 1,), -1, dtype=torch.long, device=flat_rows.device)
    row_position.scatter_(0, safe_rows, torch.arange(flat_rows.numel(), device=flat_rows.device))
    claims = torch.zeros(num_rows + 1, dtype=torch.long, device=flat_rows.device).scatter_add_(
        0, safe_rows, named.long()
    )
    contested = (claims[:-1] > 1).nonzero().flatten()
    if contested.numel():
        raise ValueError(
            f"support rows {contested.tolist()} are each named by more than one model position, "
            "so no single position's scores can stand for the row"
        )
    return row_position[:-1], safe_rows


def sample_support_csr_scores(
    logits_or_hidden: torch.Tensor,
    sampled_ids: torch.Tensor,
    support: PackedRaggedTensor,
    row_ids: torch.Tensor,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group: torch.distributed.ProcessGroup | None,
    compute_entropy: bool,
    entropy_requires_grad: bool,
    lm_head_weight: torch.Tensor | None = None,
    temperature: float = 1.0,
    chunk_size: int | None = None,
) -> SampleSupportScores:
    """Renormalize over each token's recorded members without padding to a fixed width.

    Each support row must contain distinct members, including the sampled token.
    """
    if logits_or_hidden.shape[:-1] != sampled_ids.shape or row_ids.shape != sampled_ids.shape:
        raise ValueError(
            "logits, sampled_ids, and row_ids must have matching prefix shapes, got "
            f"{logits_or_hidden.shape[:-1]}, {sampled_ids.shape}, and {row_ids.shape}"
        )
    if support.dtype != SAMPLE_SUPPORT_TORCH_DTYPE:
        raise ValueError(f"sample support must use {SAMPLE_SUPPORT_TORCH_DTYPE} vocab ids, got {support.dtype}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if entropy_requires_grad and not compute_entropy:
        raise ValueError("entropy gradients require compute_entropy=True")

    device = logits_or_hidden.device
    flat_source = logits_or_hidden.reshape(-1, logits_or_hidden.shape[-1])
    flat_sampled = sampled_ids.reshape(-1).long()
    num_rows = support.num_rows
    row_position, safe_rows = _invert_row_ids(row_ids, num_rows)

    row_lengths = support.row_lengths.to(torch.long)
    scored_rows = row_position >= 0
    valid_rows = scored_rows & (row_lengths > 0)
    projected_rows = scored_rows & (row_lengths > 1)

    member_rows = torch.repeat_interleave(
        torch.arange(num_rows, device=device), row_lengths, output_size=support.values.numel()
    )
    member_vocab = support.values.long()
    local_members = projected_rows[member_rows] & (member_vocab >= vocab_start_index) & (member_vocab < vocab_end_index)
    local_rows = member_rows[local_members]
    local_vocab = member_vocab[local_members]
    local_ids = local_vocab - vocab_start_index
    local_positions = row_position[local_rows]

    compute_dtype = (
        torch.float32 if logits_or_hidden.dtype in (torch.float16, torch.bfloat16) else logits_or_hidden.dtype
    )
    if lm_head_weight is None:
        local_values = flat_source[local_positions, local_ids].to(compute_dtype)
    else:
        if lm_head_weight.shape[0] != vocab_end_index - vocab_start_index:
            raise ValueError(
                f"lm_head_weight holds {lm_head_weight.shape[0]} rows for vocabulary shard "
                f"[{vocab_start_index}, {vocab_end_index})"
            )
        local_values = _project_candidate_pairs(
            flat_source, local_positions, local_ids, lm_head_weight, temperature, chunk_size
        )

    local_max = torch.full((num_rows,), float("-inf"), dtype=local_values.dtype, device=device)
    local_max.scatter_reduce_(0, local_rows, local_values.detach(), reduce="amax", include_self=True)
    if tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
        torch.distributed.all_reduce(local_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
    safe_max = torch.where(projected_rows, local_max, 0.0)

    shifted = local_values - safe_max[local_rows]
    local_exp = shifted.exp()
    sampled_for_row = flat_sampled[row_position.clamp(min=0)]
    sampled_member = local_vocab == sampled_for_row[local_rows]
    local_stats = [
        local_values.new_zeros(num_rows).index_add(0, local_rows, local_exp),
        local_values.new_zeros(num_rows).index_add(0, local_rows, torch.where(sampled_member, local_values, 0.0)),
    ]
    if compute_entropy:
        # Same decomposition the fixed-width scorer uses: H = log(Z) - E_p[l - max], and both terms
        # are sums over the members Z already sums over, so this is another row of the reduction.
        weights = local_exp if entropy_requires_grad else local_exp.detach()
        local_stats.append(
            local_values.new_zeros(num_rows).index_add(
                0, local_rows, weights * (shifted if entropy_requires_grad else shifted.detach())
            )
        )
    # A vocabulary shard that owns no member of this microbatch would otherwise leave the backward
    # graph entirely -- `_project_candidate_pairs` short-circuits an empty pair list -- and both
    # Megatron and FSDP expect a gradient buffer for the hidden states and the LM head on every
    # rank. A zero-weighted touch of each input restores the edge and adds 0.0 to the value.
    connect = flat_source.reshape(-1)[:1].sum() * 0.0
    if lm_head_weight is not None:
        connect = connect + lm_head_weight.reshape(-1)[:1].sum() * 0.0
    local_stats[0] = local_stats[0] + connect.to(local_stats[0].dtype)

    stacked = torch.stack(local_stats)
    global_stats = stacked.detach().clone()
    if tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
        torch.distributed.all_reduce(global_stats, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    global_stats = global_stats + stacked - stacked.detach()
    denominator, sampled_score = global_stats[0], global_stats[1]
    row_logprobs = torch.where(
        projected_rows,
        sampled_score - safe_max - torch.where(projected_rows, denominator, 1.0).log(),
        0.0,
    )

    row_entropy = None
    if compute_entropy:
        weighted_sum = global_stats[2] if entropy_requires_grad else global_stats[2].detach()
        safe_denominator = torch.where(
            projected_rows, denominator if entropy_requires_grad else denominator.detach(), 1.0
        )
        row_entropy = torch.where(projected_rows, safe_denominator.log() - weighted_sum / safe_denominator, 0.0)

    def at_positions(row_values: torch.Tensor) -> torch.Tensor:
        """Read each model position's row, with the sentinel slot standing in for "no row"."""
        return torch.cat([row_values, row_values.new_zeros(1)])[safe_rows].reshape(sampled_ids.shape)

    return SampleSupportScores(
        logprobs=at_positions(row_logprobs),
        entropy=None if row_entropy is None else at_positions(row_entropy),
        valid_mask=at_positions(valid_rows),
    )


def _trajectory_ids_for_fallback(
    synthetic_eos_mask: torch.Tensor,
    metadata_layout: TokenMetadataLayout | None,
    trajectory_ids: torch.Tensor | None,
    num_trajectories: int | None,
) -> tuple[torch.Tensor, int]:
    """Return per-position trajectory ids and the required fallback capacity."""
    if trajectory_ids is not None:
        if trajectory_ids.shape != synthetic_eos_mask.shape:
            raise ValueError(
                f"trajectory_ids shape {trajectory_ids.shape} does not match the synthetic EOS mask "
                f"{synthetic_eos_mask.shape}"
            )
        if num_trajectories is None or num_trajectories <= 0:
            raise ValueError(f"num_trajectories must be positive alongside trajectory_ids, got {num_trajectories}")
        return trajectory_ids.reshape(-1).to(torch.long), num_trajectories

    if metadata_layout is None or metadata_layout.padded_sequence_lengths is None:
        if synthetic_eos_mask.shape[0] == 0 or synthetic_eos_mask.shape[1] == 0:
            raise ValueError(f"Synthetic EOS fallback requires non-empty segments, got {synthetic_eos_mask.shape}")
        width = synthetic_eos_mask.shape[1]
        positions = torch.arange(synthetic_eos_mask.numel(), device=synthetic_eos_mask.device)
        return positions // width, synthetic_eos_mask.shape[0]

    if synthetic_eos_mask.shape[0] != 1:
        raise ValueError(
            f"Packed synthetic EOS metadata must have a singleton batch dimension, got {synthetic_eos_mask.shape}"
        )
    if metadata_layout.cu_seqlens_padded is None:
        raise ValueError("Packed synthetic EOS fallback requires padded sequence boundaries")
    if any(length <= 0 for length in metadata_layout.padded_sequence_lengths):
        raise ValueError(
            f"Synthetic EOS fallback requires non-empty segments, got {metadata_layout.padded_sequence_lengths}"
        )
    expected_tokens = metadata_layout.aligned_sequence_length // metadata_layout.context_parallel_size
    if expected_tokens != synthetic_eos_mask.numel():
        raise ValueError(
            f"Synthetic EOS layout holds {expected_tokens} tokens for a {synthetic_eos_mask.numel()}-token microbatch"
        )
    # CP shards each padded segment evenly, so a CP-local segment is its padded length over cp_size.
    lengths = (
        metadata_layout.cu_seqlens_padded.to(device=synthetic_eos_mask.device, dtype=torch.long).diff()
        // metadata_layout.context_parallel_size
    )
    trajectory_ids = torch.repeat_interleave(
        torch.arange(lengths.shape[0], device=lengths.device),
        lengths,
        output_size=synthetic_eos_mask.numel(),
    )
    return trajectory_ids, lengths.shape[0]


def synthetic_eos_logprobs(
    logits_or_hidden: torch.Tensor,
    sampled_ids: torch.Tensor,
    synthetic_eos_mask: torch.Tensor,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group: torch.distributed.ProcessGroup | None,
    inference_only: bool,
    lm_head_weight: torch.Tensor | None = None,
    temperature: float = 1.0,
    chunk_size: int | None = None,
    fused_backend: str = "torch",
    metadata_layout: TokenMetadataLayout | None = None,
    trajectory_ids: torch.Tensor | None = None,
    num_trajectories: int | None = None,
) -> torch.Tensor:
    """Score an EOS appended after generation over the full vocabulary."""
    if synthetic_eos_mask.shape != sampled_ids.shape:
        raise ValueError(
            f"synthetic_eos_mask shape {synthetic_eos_mask.shape} does not match sampled ids {sampled_ids.shape}"
        )

    trajectory_ids, capacity = _trajectory_ids_for_fallback(
        synthetic_eos_mask, metadata_layout, trajectory_ids, num_trajectories
    )
    # Clamp non-loss-bearing sequence-parallel padding ids for scatter operations.
    trajectory_ids = trajectory_ids.clamp(0, capacity - 1)
    flat_mask = synthetic_eos_mask.reshape(-1)
    # Replay permits only the single appended EOS to lack recorded support.
    per_trajectory_count = torch.zeros(capacity, dtype=torch.long, device=flat_mask.device).scatter_add_(
        0, trajectory_ids, flat_mask.long()
    )
    offenders = (per_trajectory_count > 1).nonzero().flatten()
    if offenders.numel():
        raise ValueError(
            "sample-support replay permits at most one loss-bearing token without recorded support per "
            f"trajectory (the appended EOS), got counts {per_trajectory_count[offenders].tolist()} for "
            f"trajectories {offenders.tolist()}"
        )

    token_indices = torch.arange(synthetic_eos_mask.numel(), device=flat_mask.device)
    sentinel = synthetic_eos_mask.numel()
    candidate_indices = torch.where(flat_mask, token_indices, sentinel)
    selected_indices = torch.full(
        (capacity,),
        sentinel,
        dtype=torch.long,
        device=flat_mask.device,
    ).scatter_reduce(0, trajectory_ids, candidate_indices, reduce="amin", include_self=True)
    has_selection = selected_indices != sentinel
    selected_indices = torch.where(has_selection, selected_indices, 0)

    flat_source = logits_or_hidden.reshape(-1, logits_or_hidden.shape[-1])
    flat_targets = sampled_ids.reshape(-1)
    selected_source = flat_source.index_select(0, selected_indices)
    selected_targets = flat_targets.index_select(0, selected_indices)
    if lm_head_weight is None and tp_group is None:
        # Only one row per trajectory is softmaxed on the unsharded path.
        selected = selected_source.log_softmax(dim=-1).gather(1, selected_targets.unsqueeze(1)).squeeze(1)
    elif lm_head_weight is None:
        from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
            DistributedLogprob,
        )

        selected = DistributedLogprob.apply(
            selected_source.unsqueeze(0),
            selected_targets.unsqueeze(0),
            vocab_start_index,
            vocab_end_index,
            tp_group,
            inference_only,
        ).squeeze(0)
    else:
        from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
            _fused_lm_head_logprob_apply,
        )

        if temperature != 1.0:
            lm_head_weight = lm_head_weight / temperature
        selected_chunk_size = (
            selected_source.shape[0] if chunk_size is None else min(chunk_size, selected_source.shape[0])
        )
        selected = _fused_lm_head_logprob_apply(
            fused_backend,
            selected_source.unsqueeze(0),
            lm_head_weight,
            selected_targets.unsqueeze(0),
            vocab_start_index,
            vocab_end_index,
            selected_chunk_size,
            tp_group,
            inference_only,
        ).squeeze(0)
    selected = torch.where(has_selection, selected, 0.0)
    output = torch.zeros(sampled_ids.numel(), dtype=selected.dtype, device=selected.device)
    return output.scatter_add(0, selected_indices, selected).reshape(sampled_ids.shape)


def sample_support_row_ids_in_batch_positions(
    sample_support: PackedSampleSupport,
    layout: TokenMetadataLayout,
) -> torch.Tensor:
    """Map support rows into canonical left-padded batch coordinates."""
    row_ids = align_sample_support_row_ids(sample_support, layout)
    sequence_length = layout.attention_mask.shape[1]
    padding_widths = sequence_length - layout.attention_mask.sum(dim=1).to(torch.long)
    positions = torch.arange(sequence_length, device=row_ids.device).unsqueeze(0) - padding_widths.unsqueeze(1)
    return torch.where(positions >= 0, row_ids.gather(1, positions.clamp(min=0)), SAMPLE_SUPPORT_NO_ROW)


def _gather_support_rows(support: PackedTensor, row_ids: torch.Tensor) -> torch.Tensor:
    """Gather each model position's ``[top_k]`` support row, by id rather than by placement."""
    top_k = support.values.shape[1]
    if support.values.shape[0] == 0:
        return torch.full(
            (*row_ids.shape, top_k),
            SAMPLE_SUPPORT_PADDING,
            dtype=support.dtype,
            device=support.device,
        )
    flat_row_ids = row_ids.reshape(-1)
    gathered = support.values.index_select(0, flat_row_ids.clamp(min=0))
    gathered = gathered.masked_fill((flat_row_ids < 0).unsqueeze(1), SAMPLE_SUPPORT_PADDING)
    return gathered.reshape(*row_ids.shape, top_k)


def score_aligned_sample_support(
    aligned_source: torch.Tensor,
    aligned_sampled_ids: torch.Tensor,
    aligned_row_ids: torch.Tensor,
    aligned_loss_mask: torch.Tensor,
    sample_support: PackedSampleSupport,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group: torch.distributed.ProcessGroup | None,
    inference_only: bool,
    compute_entropy: bool,
    entropy_requires_grad: bool,
    lm_head_weight: torch.Tensor | None = None,
    temperature: float = 1.0,
    chunk_size: int | None = None,
    fused_backend: str = "torch",
    metadata_layout: TokenMetadataLayout | None = None,
    trajectory_ids: torch.Tensor | None = None,
    num_trajectories: int | None = None,
) -> SampleSupportScores:
    """Score token-aligned support rows and full-vocabulary EOS fallbacks.

    ``temperature`` applies only when ``aligned_source`` contains hidden states.
    Synthetic-EOS rows use full-vocabulary logprobs and remain outside the support-entropy mask.
    Both packed support forms share outer segmentation and row IDs.
    """
    scorer_kwargs = dict(
        vocab_start_index=vocab_start_index,
        vocab_end_index=vocab_end_index,
        tp_group=tp_group,
        compute_entropy=compute_entropy,
        entropy_requires_grad=entropy_requires_grad,
        lm_head_weight=lm_head_weight,
        temperature=temperature if lm_head_weight is not None else 1.0,
        chunk_size=chunk_size,
    )
    if isinstance(sample_support, PackedRaggedTensor):
        scores = sample_support_csr_scores(
            aligned_source,
            aligned_sampled_ids,
            sample_support,
            aligned_row_ids,
            **scorer_kwargs,
        )
    else:
        scores = sample_support_scores(
            aligned_source,
            aligned_sampled_ids,
            _gather_support_rows(sample_support, aligned_row_ids),
            **scorer_kwargs,
        )
    synthetic_eos_mask = aligned_loss_mask & ~scores.valid_mask
    eos_logprobs = synthetic_eos_logprobs(
        aligned_source,
        aligned_sampled_ids,
        synthetic_eos_mask,
        vocab_start_index=vocab_start_index,
        vocab_end_index=vocab_end_index,
        tp_group=tp_group,
        inference_only=inference_only,
        lm_head_weight=lm_head_weight,
        temperature=temperature if lm_head_weight is not None else 1.0,
        chunk_size=chunk_size,
        fused_backend=fused_backend,
        metadata_layout=metadata_layout,
        trajectory_ids=trajectory_ids,
        num_trajectories=num_trajectories,
    )
    return SampleSupportScores(
        logprobs=torch.where(synthetic_eos_mask, eos_logprobs, scores.logprobs),
        entropy=scores.entropy,
        valid_mask=scores.valid_mask,
    )


def compute_sample_support_scores(
    logits_or_hidden: torch.Tensor,
    sequences: torch.Tensor,
    loss_mask: torch.Tensor | None,
    sample_support: PackedSampleSupport | None,
    num_actions: int,
    *,
    packed: bool,
    metadata_layout: TokenMetadataLayout | None,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group: torch.distributed.ProcessGroup | None,
    inference_only: bool,
    lm_head_weight: torch.Tensor | None,
    temperature: float,
    chunk_size: int | None,
    fused_backend: str,
    compute_entropy: bool,
    entropy_requires_grad: bool,
) -> SampleSupportScores:
    """Score support-conditioned logprobs in canonical trainer layout."""
    if sample_support is None:
        raise ValueError(missing_sample_support_message("Megatron"))
    if loss_mask is None:
        raise ValueError("sample-support replay requires the response loss mask")
    if metadata_layout is None:
        raise ValueError("sample-support replay requires the shared token metadata layout")

    target_loss_mask = torch.zeros_like(sequences, dtype=torch.bool)
    target_loss_mask[:, -num_actions:] = loss_mask.to(torch.bool)
    if packed:
        row_ids = align_sample_support_row_ids(sample_support, metadata_layout)
        aligned_sampled_ids = align_token_metadata(sequences, metadata_layout, 0, next_token=True)
        aligned_loss_mask = align_token_metadata(target_loss_mask, metadata_layout, False, next_token=True)
        aligned_source = logits_or_hidden
    else:
        # The domain ends at real token L_i - 2, so dropping the last column loses no row.
        row_ids = sample_support_row_ids_in_batch_positions(sample_support, metadata_layout)[:, :-1]
        aligned_sampled_ids = sequences[:, 1:]
        aligned_loss_mask = target_loss_mask[:, 1:]
        aligned_source = logits_or_hidden[:, :-1]

    scores = score_aligned_sample_support(
        aligned_source,
        aligned_sampled_ids,
        row_ids,
        aligned_loss_mask,
        sample_support,
        vocab_start_index=vocab_start_index,
        vocab_end_index=vocab_end_index,
        tp_group=tp_group,
        inference_only=inference_only,
        compute_entropy=compute_entropy,
        entropy_requires_grad=entropy_requires_grad,
        lm_head_weight=lm_head_weight,
        temperature=temperature,
        chunk_size=chunk_size,
        fused_backend=fused_backend,
        metadata_layout=metadata_layout if packed else None,
    )
    if not packed:
        return scores
    return SampleSupportScores(
        logprobs=scatter_packed_token_values_to_batch(scores.logprobs, metadata_layout, 0),
        entropy=(
            None if scores.entropy is None else scatter_packed_token_values_to_batch(scores.entropy, metadata_layout, 0)
        ),
        valid_mask=scatter_packed_token_values_to_batch(scores.valid_mask, metadata_layout, False),
    )
