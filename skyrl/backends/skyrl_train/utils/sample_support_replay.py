"""Support-conditioned logprobs for bounded sampler replay."""

from dataclasses import dataclass

import torch

from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import (
    TokenMetadataLayout,
    align_token_metadata,
    scatter_packed_token_values_to_batch,
)
from skyrl.backends.skyrl_train.training_batch import TensorList
from skyrl.backends.skyrl_train.utils.torch_utils import logprobs_from_logits


@dataclass(frozen=True)
class SampleSupportScores:
    """Support-conditioned scores and the rows backed by recorded support."""

    logprobs: torch.Tensor
    entropy: torch.Tensor | None
    valid_mask: torch.Tensor


class _ChunkedCandidateProjection(torch.autograd.Function):
    """Selected LM-head projection without retaining ``[pairs, hidden]`` activations."""

    @staticmethod
    def forward(ctx, hidden, weight, row_ids, token_ids, temperature, chunk_size):
        ctx.save_for_backward(hidden, weight, row_ids, token_ids)
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        output = torch.empty(token_ids.shape, dtype=torch.float32, device=hidden.device)
        for start in range(0, token_ids.numel(), chunk_size):
            end = min(start + chunk_size, token_ids.numel())
            selected_hidden = hidden.index_select(0, row_ids[start:end]).to(weight.dtype)
            selected_weight = weight.index_select(0, token_ids[start:end])
            output[start:end] = ((selected_hidden * selected_weight).sum(dim=-1) / temperature).float()
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
        raise ValueError("candidate projection chunk size must be positive")
    if torch.is_grad_enabled() and (hidden.requires_grad or lm_head_weight.requires_grad):
        # A normal loop retains every selected chunk for backward. The custom
        # backward reselects one chunk at a time, so peak pair storage is bounded.
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
        selected_hidden = hidden.index_select(0, row_ids[start:end]).to(lm_head_weight.dtype)
        selected_weight = lm_head_weight.index_select(0, token_ids[start:end])
        projected = (selected_hidden * selected_weight).sum(dim=-1) / temperature
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
    """Project a fixed-width candidate matrix without materializing vocabulary logits."""
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
    lm_head_weight: torch.Tensor | None = None,
    temperature: float = 1.0,
    chunk_size: int | None = None,
    compute_entropy: bool,
    entropy_requires_grad: bool,
) -> SampleSupportScores:
    """Compute sampled-token logprobs and optional entropy over recorded support."""
    if logits_or_hidden.shape[:-1] != sampled_ids.shape or support_ids.shape[:-1] != sampled_ids.shape:
        raise ValueError(
            "logits, sampled_ids, and support_ids must have matching prefix shapes, got "
            f"{logits_or_hidden.shape[:-1]}, {sampled_ids.shape}, and {support_ids.shape[:-1]}"
        )
    if support_ids.dtype != torch.int32:
        raise ValueError(f"sample support must use int32 vocab ids, got {support_ids.dtype}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
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
            raise ValueError("lm_head_weight rows do not match the configured vocabulary shard")
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
    local_sum = local_exp.sum(dim=-1)
    local_stats = [local_sum, local_sampled]
    if compute_entropy:
        entropy_values = local_values if entropy_requires_grad else local_values.detach()
        entropy_exp = local_exp if entropy_requires_grad else local_exp.detach()
        shifted_values = torch.where(local_members, entropy_values - safe_max.unsqueeze(1), 0.0)
        local_stats.append((entropy_exp * shifted_values).sum(dim=-1))
    # Numerator, denominator, and optional entropy statistic share one TP SUM collective.
    local_stats = torch.stack(local_stats)
    global_stats = local_stats.detach().clone()
    if tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
        torch.distributed.all_reduce(global_stats, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    global_stats = global_stats + local_stats - local_stats.detach()
    denominator, sampled_score = global_stats[:2]
    logprobs = sampled_score - safe_max - torch.where(valid_rows, denominator, 1.0).log()
    logprobs = torch.where(valid_rows, logprobs, 0.0)
    entropy = None
    if compute_entropy:
        entropy_denominator = denominator if entropy_requires_grad else denominator.detach()
        shifted_score_sum = global_stats[2] if entropy_requires_grad else global_stats[2].detach()
        safe_denominator = torch.where(valid_rows, entropy_denominator, 1.0)
        entropy = safe_denominator.log() - shifted_score_sum / safe_denominator
        entropy = torch.where(valid_rows, entropy, 0.0).reshape(sampled_ids.shape)
    return SampleSupportScores(
        logprobs=logprobs.reshape(sampled_ids.shape),
        entropy=entropy,
        valid_mask=valid_rows.reshape(sampled_ids.shape),
    )


def build_sample_support_row_ids(
    sample_support_offsets: TensorList,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Place microbatch-local CSR row IDs at their right-aligned target tokens."""
    row_ids = torch.full((len(sample_support_offsets), sequence_length), -1, dtype=torch.long, device=device)
    for sample_index, offsets in enumerate(sample_support_offsets.tensors):
        if offsets.device != device:
            raise ValueError("sample-support offsets and sequences must be on the same device")
        num_rows = offsets.numel() - 1
        if num_rows < 0 or num_rows > sequence_length:
            raise ValueError(
                f"sample {sample_index} has {num_rows} sample-support rows for sequence length {sequence_length}"
            )
        if num_rows:
            start = sequence_length - num_rows
            row_ids[sample_index, start:] = sample_index * sequence_length + torch.arange(
                num_rows, dtype=torch.long, device=device
            )
    return row_ids


def _assemble_sample_support_csr(
    sample_support_ids: TensorList,
    sample_support_offsets: TensorList,
    grid_width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Join per-sample CSR tensors in the canonical ``batch * grid_width`` row space."""
    if len(sample_support_ids) != len(sample_support_offsets):
        raise ValueError("sample-support ids/offsets batch mismatch")

    canonical_row_sizes = torch.zeros(len(sample_support_ids) * grid_width, dtype=torch.long, device=device)
    member_rows = []
    member_vocab = []
    for sample_index, (ids, offsets) in enumerate(
        zip(sample_support_ids.tensors, sample_support_offsets.tensors, strict=True)
    ):
        if ids.device != device or offsets.device != device:
            raise ValueError("sample-support CSR tensors and model inputs must be on the same device")
        if ids.dtype != torch.int32 or offsets.dtype != torch.int32:
            raise ValueError("sample-support CSR ids and offsets must use int32")
        num_rows = offsets.numel() - 1
        if num_rows < 0 or num_rows > grid_width:
            raise ValueError(
                f"sample {sample_index} has {num_rows} sample-support rows for sequence length {grid_width}"
            )
        canonical_rows = sample_index * grid_width + torch.arange(num_rows, device=device)
        row_sizes = offsets[1:].long() - offsets[:-1].long()
        canonical_row_sizes[canonical_rows] = row_sizes
        member_rows.append(torch.repeat_interleave(canonical_rows, row_sizes, output_size=ids.numel()))
        member_vocab.append(ids.long())

    empty = torch.empty(0, dtype=torch.long, device=device)
    return (
        torch.cat(member_rows) if member_rows else empty,
        torch.cat(member_vocab) if member_vocab else empty,
        canonical_row_sizes,
    )


def sample_support_csr_scores(
    logits_or_hidden: torch.Tensor,
    sampled_ids: torch.Tensor,
    support_row_ids: torch.Tensor,
    sample_support_ids: TensorList,
    sample_support_offsets: TensorList,
    support_grid_width: int,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group: torch.distributed.ProcessGroup | None,
    lm_head_weight: torch.Tensor | None = None,
    temperature: float = 1.0,
    chunk_size: int | None = None,
    compute_entropy: bool,
    entropy_requires_grad: bool,
) -> SampleSupportScores:
    """Compute logprobs and optional entropy over ragged support."""
    if logits_or_hidden.shape[:-1] != sampled_ids.shape or sampled_ids.shape != support_row_ids.shape:
        raise ValueError(
            "logits, sampled_ids, and support row ids must have matching prefix shapes, got "
            f"{logits_or_hidden.shape[:-1]}, {sampled_ids.shape}, and {support_row_ids.shape}"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if entropy_requires_grad and not compute_entropy:
        raise ValueError("entropy gradients require compute_entropy=True")

    flat_source = logits_or_hidden.reshape(-1, logits_or_hidden.shape[-1])
    flat_sampled = sampled_ids.reshape(-1).long()
    aligned_rows = support_row_ids.reshape(-1).long()
    member_rows, member_vocab, canonical_row_sizes = _assemble_sample_support_csr(
        sample_support_ids,
        sample_support_offsets,
        support_grid_width,
        logits_or_hidden.device,
    )
    num_canonical_rows = canonical_row_sizes.numel()
    canonical_validity = canonical_row_sizes > 0
    active_rows = canonical_row_sizes > 1

    # Row IDs follow token packing/CP sharding; the CSR members stay compact and
    # join back here through the canonical microbatch row namespace.
    in_range = (aligned_rows >= 0) & (aligned_rows < num_canonical_rows)
    sentinel = num_canonical_rows
    safe_aligned_rows = torch.where(in_range, aligned_rows, sentinel)
    canonical_to_position = torch.full((num_canonical_rows + 1,), -1, dtype=torch.long, device=logits_or_hidden.device)
    canonical_to_position.scatter_(
        0,
        safe_aligned_rows,
        torch.arange(aligned_rows.numel(), device=logits_or_hidden.device),
    )
    canonical_positions = canonical_to_position[:-1]
    valid_support = in_range & canonical_validity[safe_aligned_rows.clamp_max(num_canonical_rows - 1)]

    member_positions = canonical_positions[member_rows]
    local_member_mask = (
        (member_positions >= 0)
        & active_rows[member_rows]
        & (member_vocab >= vocab_start_index)
        & (member_vocab < vocab_end_index)
    )
    local_member_rows = member_rows[local_member_mask]
    local_member_vocab = member_vocab[local_member_mask]
    local_member_positions = member_positions[local_member_mask]
    local_member_ids = local_member_vocab - vocab_start_index
    compute_dtype = (
        torch.float32 if logits_or_hidden.dtype in (torch.float16, torch.bfloat16) else logits_or_hidden.dtype
    )
    if lm_head_weight is None:
        local_member_values = flat_source[local_member_positions, local_member_ids].to(compute_dtype)
    else:
        if lm_head_weight.shape[0] != vocab_end_index - vocab_start_index:
            raise ValueError("lm_head_weight rows do not match the configured vocabulary shard")
        local_member_values = _project_candidate_pairs(
            flat_source,
            local_member_positions,
            local_member_ids,
            lm_head_weight,
            temperature,
            chunk_size,
        )

    local_max = local_member_values.new_full((num_canonical_rows,), float("-inf"))
    local_max.index_reduce_(0, local_member_rows, local_member_values.detach(), "amax", include_self=True)
    global_max = local_max.clone()
    if tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
        torch.distributed.all_reduce(global_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
    safe_max = torch.where(active_rows, global_max, 0.0)

    local_exp = (local_member_values - safe_max[local_member_rows]).exp()
    local_sum = local_member_values.new_zeros(num_canonical_rows).index_add(
        0,
        local_member_rows,
        local_exp,
    )
    sampled_positions = canonical_positions.clamp_min(0)
    sampled_for_row = flat_sampled[sampled_positions]
    sampled_member_mask = local_member_vocab == sampled_for_row[local_member_rows]
    local_sampled = local_member_values.new_zeros(num_canonical_rows).index_add(
        0,
        local_member_rows,
        torch.where(sampled_member_mask, local_member_values, 0.0),
    )

    # Keep every TP shard connected to autograd even when it owns no active
    # candidates. Megatron still expects dense gradient buffers for the hidden
    # states and LM-head weight on such a shard.
    autograd_zero = flat_source.reshape(-1)[:1].sum().to(compute_dtype) * 0.0
    if lm_head_weight is not None:
        autograd_zero = autograd_zero + lm_head_weight.reshape(-1)[:1].sum().to(compute_dtype) * 0.0
    local_sum = local_sum + autograd_zero

    local_stats = [local_sum, local_sampled]
    if compute_entropy:
        entropy_values = local_member_values if entropy_requires_grad else local_member_values.detach()
        entropy_exp = local_exp if entropy_requires_grad else local_exp.detach()
        local_shifted_score_sum = local_member_values.new_zeros(num_canonical_rows).index_add(
            0,
            local_member_rows,
            entropy_exp * (entropy_values - safe_max[local_member_rows]),
        )
        local_stats.append(local_shifted_score_sum)
    # Numerator, denominator, and optional entropy statistic share one TP SUM collective.
    local_stats = torch.stack(local_stats)
    global_stats = local_stats.detach().clone()
    if tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
        torch.distributed.all_reduce(global_stats, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    global_stats = global_stats + local_stats - local_stats.detach()
    denominator, sampled_score = global_stats[:2]
    canonical_logprobs = sampled_score - safe_max - torch.where(active_rows, denominator, 1.0).log()
    aligned_logprobs = canonical_logprobs[safe_aligned_rows.clamp_max(num_canonical_rows - 1)]
    aligned_logprobs = torch.where(valid_support, aligned_logprobs, 0.0)
    entropy = None
    if compute_entropy:
        entropy_denominator = denominator if entropy_requires_grad else denominator.detach()
        shifted_score_sum = global_stats[2] if entropy_requires_grad else global_stats[2].detach()
        safe_denominator = torch.where(active_rows, entropy_denominator, 1.0)
        canonical_entropy = safe_denominator.log() - shifted_score_sum / safe_denominator
        aligned_entropy = canonical_entropy[safe_aligned_rows.clamp_max(num_canonical_rows - 1)]
        entropy = torch.where(valid_support, aligned_entropy, 0.0).reshape(sampled_ids.shape)
    return SampleSupportScores(
        logprobs=aligned_logprobs.reshape(sampled_ids.shape),
        entropy=entropy,
        valid_mask=valid_support.reshape(sampled_ids.shape),
    )


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
    """Compute ordinary logprobs for EOS tokens appended after vLLM generation."""
    if synthetic_eos_mask.shape != sampled_ids.shape:
        raise ValueError("synthetic_eos_mask and sampled_ids must have matching shapes")

    if trajectory_ids is not None:
        if trajectory_ids.shape != synthetic_eos_mask.shape:
            raise ValueError("trajectory_ids and synthetic_eos_mask must have matching shapes")
        if num_trajectories is None or num_trajectories <= 0:
            raise ValueError("num_trajectories must be positive when trajectory_ids are provided")
        flat_trajectory_ids = trajectory_ids.reshape(-1).to(torch.long)
        capacity = num_trajectories
    elif metadata_layout is not None and metadata_layout.padded_sequence_lengths is not None:
        if synthetic_eos_mask.shape[0] != 1:
            raise ValueError("Packed synthetic EOS metadata must have a singleton batch dimension")
        if metadata_layout.cu_seqlens_padded is None:
            raise ValueError("Packed synthetic EOS fallback requires padded sequence boundaries")
        if any(length <= 0 for length in metadata_layout.padded_sequence_lengths):
            raise ValueError("Synthetic EOS fallback requires non-empty trajectory segments")
        expected_tokens = metadata_layout.aligned_sequence_length // metadata_layout.context_parallel_size
        if expected_tokens != synthetic_eos_mask.numel():
            raise ValueError("Synthetic EOS layout does not match the model token layout")
        lengths = (
            metadata_layout.cu_seqlens_padded.to(
                device=synthetic_eos_mask.device,
                dtype=torch.long,
            ).diff()
            // metadata_layout.context_parallel_size
        )
        capacity = lengths.shape[0]
        flat_trajectory_ids = torch.repeat_interleave(
            torch.arange(capacity, device=lengths.device),
            lengths,
            output_size=synthetic_eos_mask.numel(),
        )
    else:
        if synthetic_eos_mask.shape[0] == 0 or synthetic_eos_mask.shape[1] == 0:
            raise ValueError("Synthetic EOS fallback requires non-empty trajectory segments")
        lengths = torch.full(
            (synthetic_eos_mask.shape[0],),
            synthetic_eos_mask.shape[1],
            dtype=torch.long,
            device=synthetic_eos_mask.device,
        )
        capacity = lengths.shape[0]
        flat_trajectory_ids = torch.repeat_interleave(
            torch.arange(capacity, device=lengths.device),
            lengths,
            output_size=synthetic_eos_mask.numel(),
        )

    # Preprocessing permits at most one unsupported loss-bearing EOS per
    # trajectory. Select one fixed slot for every trajectory so TP collectives
    # never depend on the number of EOS fallbacks in this microbatch.
    token_indices = torch.arange(synthetic_eos_mask.numel(), device=synthetic_eos_mask.device)
    sentinel = synthetic_eos_mask.numel()
    valid_trajectory = (flat_trajectory_ids >= 0) & (flat_trajectory_ids < capacity)
    candidate_indices = torch.where(synthetic_eos_mask.reshape(-1) & valid_trajectory, token_indices, sentinel)
    selected_indices = torch.full(
        (capacity,),
        sentinel,
        dtype=torch.long,
        device=synthetic_eos_mask.device,
    ).scatter_reduce(
        0,
        flat_trajectory_ids.clamp(0, capacity - 1),
        candidate_indices,
        reduce="amin",
        include_self=True,
    )
    has_selection = selected_indices != sentinel
    selected_indices = torch.where(has_selection, selected_indices, 0)

    flat_source = logits_or_hidden.reshape(-1, logits_or_hidden.shape[-1])
    flat_targets = sampled_ids.reshape(-1)
    selected_source = flat_source.index_select(0, selected_indices)
    selected_targets = flat_targets.index_select(0, selected_indices)
    source_is_full_vocab_logits = lm_head_weight is None and tp_group is None
    source_is_tp_sharded_logits = lm_head_weight is None and tp_group is not None
    if source_is_full_vocab_logits:
        selected = logprobs_from_logits(selected_source, selected_targets, inplace_backward=False)
    elif source_is_tp_sharded_logits:
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
        assert lm_head_weight is not None and tp_group is not None
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
    selected = torch.where(has_selection, selected, 0.0).to(torch.float32)
    output = torch.zeros(sampled_ids.numel(), dtype=torch.float32, device=logits_or_hidden.device)
    output = output.scatter_add(0, selected_indices, selected)
    return output.reshape(sampled_ids.shape)


def aligned_sample_support_scores(
    logits_or_hidden: torch.Tensor,
    sampled_ids: torch.Tensor,
    support_ids: torch.Tensor,
    loss_mask: torch.Tensor,
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
    """Apply bounded replay and the synthetic-EOS exception to aligned tokens."""
    scores = sample_support_scores(
        logits_or_hidden,
        sampled_ids,
        support_ids,
        vocab_start_index=vocab_start_index,
        vocab_end_index=vocab_end_index,
        tp_group=tp_group,
        compute_entropy=compute_entropy,
        entropy_requires_grad=entropy_requires_grad,
        lm_head_weight=lm_head_weight,
        temperature=temperature if lm_head_weight is not None else 1.0,
        chunk_size=chunk_size,
    )
    # Preprocessing permits an empty loss-bearing row only for an EOS that SkyRL
    # appended after generation. vLLM never supplied a support set for that token.
    synthetic_eos_mask = loss_mask & ~scores.valid_mask
    eos_logprobs = synthetic_eos_logprobs(
        logits_or_hidden,
        sampled_ids,
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


def aligned_sample_support_csr_scores(
    logits_or_hidden: torch.Tensor,
    sampled_ids: torch.Tensor,
    support_row_ids: torch.Tensor,
    sample_support_ids: TensorList,
    sample_support_offsets: TensorList,
    support_grid_width: int,
    loss_mask: torch.Tensor,
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
    """Apply ragged replay and the synthetic-EOS exception to aligned tokens."""
    scores = sample_support_csr_scores(
        logits_or_hidden,
        sampled_ids,
        support_row_ids,
        sample_support_ids,
        sample_support_offsets,
        support_grid_width,
        vocab_start_index=vocab_start_index,
        vocab_end_index=vocab_end_index,
        tp_group=tp_group,
        compute_entropy=compute_entropy,
        entropy_requires_grad=entropy_requires_grad,
        lm_head_weight=lm_head_weight,
        temperature=temperature if lm_head_weight is not None else 1.0,
        chunk_size=chunk_size,
    )
    synthetic_eos_mask = loss_mask & ~scores.valid_mask
    eos_logprobs = synthetic_eos_logprobs(
        logits_or_hidden,
        sampled_ids,
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
    sample_support_ids: torch.Tensor | None,
    sample_support_csr_ids: TensorList | None,
    sample_support_csr_offsets: TensorList | None,
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
    """Compute support-conditioned scores in canonical trainer layout."""
    if loss_mask is None:
        raise ValueError("sample-support replay requires the response loss mask")

    dense_support = sample_support_ids is not None
    sparse_support = sample_support_csr_ids is not None or sample_support_csr_offsets is not None
    if dense_support == sparse_support:
        raise ValueError("sample-support replay requires exactly one dense or CSR representation")
    if (sample_support_csr_ids is None) != (sample_support_csr_offsets is None):
        raise ValueError("sample-support CSR ids and offsets must be supplied together")

    target_loss_mask = torch.zeros_like(sequences, dtype=torch.bool)
    target_loss_mask[:, -num_actions:] = loss_mask.to(torch.bool)
    if sparse_support:
        assert sample_support_csr_ids is not None and sample_support_csr_offsets is not None
        support_metadata = build_sample_support_row_ids(
            sample_support_csr_offsets,
            sequences.shape[1],
            sequences.device,
        )
    else:
        assert sample_support_ids is not None
        support_metadata = sample_support_ids
    if packed:
        if metadata_layout is None:
            raise ValueError("Packed sample-support replay requires the shared token metadata layout")
        aligned_sampled_ids = align_token_metadata(sequences, metadata_layout, 0, next_token=True)
        aligned_support = align_token_metadata(
            support_metadata,
            metadata_layout,
            -1,
            next_token=True,
        )
        aligned_loss_mask = align_token_metadata(target_loss_mask, metadata_layout, False, next_token=True)
    else:
        aligned_sampled_ids = sequences[:, 1:]
        aligned_support = support_metadata[:, 1:]
        aligned_loss_mask = target_loss_mask[:, 1:]

    aligned_source = logits_or_hidden if packed else logits_or_hidden[:, :-1]
    if sparse_support:
        assert sample_support_csr_ids is not None and sample_support_csr_offsets is not None
        scores = aligned_sample_support_csr_scores(
            aligned_source,
            aligned_sampled_ids,
            aligned_support,
            sample_support_csr_ids,
            sample_support_csr_offsets,
            sequences.shape[1],
            aligned_loss_mask,
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
    else:
        scores = aligned_sample_support_scores(
            aligned_source,
            aligned_sampled_ids,
            aligned_support,
            aligned_loss_mask,
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

    if packed:
        assert metadata_layout is not None
        return SampleSupportScores(
            logprobs=scatter_packed_token_values_to_batch(scores.logprobs, metadata_layout, 0),
            entropy=(
                scatter_packed_token_values_to_batch(scores.entropy, metadata_layout, 0)
                if scores.entropy is not None
                else None
            ),
            valid_mask=scatter_packed_token_values_to_batch(scores.valid_mask, metadata_layout, False),
        )
    return scores
