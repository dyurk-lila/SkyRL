"""TP benchmark for CSR replay and support-conditioned entropy.

Run on one 8-GPU node::

    torchrun --standalone --nproc-per-node=8 -m \
      skyrl.benchmarks.bench_sample_support_entropy
"""

import argparse
import json
import statistics
import time

import torch
import torch.distributed as dist

from skyrl.backends.skyrl_train.training_batch import TensorList
from skyrl.backends.skyrl_train.utils.sample_support_replay import (
    sample_support_csr_scores,
)


def _make_support(
    num_tokens: int,
    vocab_size: int,
    singleton_fraction: float,
    max_support: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TensorList, TensorList, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(1234)
    row_sizes = torch.ones(num_tokens, dtype=torch.int32)
    num_multi = round(num_tokens * (1.0 - singleton_fraction))
    if num_multi:
        multi_rows = torch.randperm(num_tokens, generator=generator)[:num_multi]
        row_sizes[multi_rows] = torch.randint(
            2,
            max_support + 1,
            (num_multi,),
            dtype=torch.int32,
            generator=generator,
        )
    offsets = torch.cat((torch.zeros(1, dtype=torch.int32), row_sizes.cumsum(0, dtype=torch.int32)))
    ids = torch.empty(int(offsets[-1]), dtype=torch.int32)
    for row, size in enumerate(row_sizes.tolist()):
        start = int(offsets[row])
        base = int(torch.randint(0, vocab_size, (), generator=generator))
        ids[start : start + size] = (base + torch.arange(size) * 7919).remainder(vocab_size).to(torch.int32)
    row_ids_for_members = torch.repeat_interleave(
        torch.arange(num_tokens),
        row_sizes.long(),
        output_size=ids.numel(),
    )
    first_member = offsets[:-1].long()
    sampled_ids = ids[first_member].long()
    dense_ids = torch.full((num_tokens, max_support), -1, dtype=torch.int32)
    for row, size in enumerate(row_sizes.tolist()):
        dense_ids[row, :size] = ids[offsets[row] : offsets[row + 1]]
    return (
        sampled_ids.unsqueeze(0).to(device),
        torch.arange(num_tokens, device=device).unsqueeze(0),
        dense_ids.unsqueeze(0).to(device),
        TensorList([ids.to(device)]),
        TensorList([offsets.to(device)]),
        row_ids_for_members.to(device),
    )


def _pr61_selected_hidden_projection(
    hidden: torch.Tensor,
    token_ids: torch.Tensor,
    local_mask: torch.Tensor,
    lm_head_weight: torch.Tensor,
    temperature: float,
    chunk_size: int | None,
    invalid_value: float,
) -> torch.Tensor:
    """Exact fixed-width candidate projection used by PR 61."""
    num_rows, width = token_ids.shape
    row_ids = torch.arange(num_rows, device=hidden.device).unsqueeze(1).expand(-1, width).reshape(-1)
    flat_token_ids = token_ids.reshape(-1)
    flat_mask = local_mask.reshape(-1)
    output = torch.empty(flat_token_ids.shape, dtype=torch.float32, device=hidden.device)
    pair_chunk_size = flat_token_ids.numel() if chunk_size is None else chunk_size
    for start in range(0, flat_token_ids.numel(), pair_chunk_size):
        end = min(start + pair_chunk_size, flat_token_ids.numel())
        selected_hidden = hidden.index_select(0, row_ids[start:end]).to(lm_head_weight.dtype)
        selected_weight = lm_head_weight.index_select(0, flat_token_ids[start:end])
        projected = (selected_hidden * selected_weight).sum(dim=-1) / temperature
        output[start:end] = torch.where(flat_mask[start:end], projected.to(torch.float32), invalid_value)
    return output.reshape(num_rows, width)


def _pr61_dense_sample_support_logprobs(
    logits_or_hidden: torch.Tensor,
    sampled_ids: torch.Tensor,
    support_ids: torch.Tensor,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group: dist.ProcessGroup,
    lm_head_weight: torch.Tensor | None = None,
    temperature: float = 1.0,
    chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact dense replay implementation at PR 61's head."""
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
        local_values = _pr61_selected_hidden_projection(
            flat_source,
            local_support_ids,
            local_members,
            lm_head_weight,
            temperature,
            chunk_size,
            float("-inf"),
        )
        local_sampled = _pr61_selected_hidden_projection(
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
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)
    safe_max = torch.where(valid_rows, global_max, 0.0)
    local_sum = torch.where(local_members, (local_values - safe_max.unsqueeze(1)).exp(), 0.0).sum(dim=-1)
    local_stats = torch.stack((local_sum, local_sampled))
    global_stats = local_stats.detach().clone()
    dist.all_reduce(global_stats, op=dist.ReduceOp.SUM, group=tp_group)
    global_stats = global_stats + local_stats - local_stats.detach()
    denominator, sampled_score = global_stats
    logprobs = sampled_score - safe_max - torch.where(valid_rows, denominator, 1.0).log()
    logprobs = torch.where(valid_rows, logprobs, 0.0)
    return logprobs.reshape(sampled_ids.shape), valid_rows.reshape(sampled_ids.shape)


def _pr80_dense_sample_support_logprobs_and_entropy(
    logits_or_hidden: torch.Tensor,
    sampled_ids: torch.Tensor,
    support_ids: torch.Tensor,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group: dist.ProcessGroup,
    entropy_requires_grad: bool,
    lm_head_weight: torch.Tensor | None = None,
    temperature: float = 1.0,
    chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact dense replay-plus-entropy implementation at PR 80's head."""
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
        local_values = _pr61_selected_hidden_projection(
            flat_source,
            local_support_ids,
            local_members,
            lm_head_weight,
            temperature,
            chunk_size,
            float("-inf"),
        )
        local_sampled = _pr61_selected_hidden_projection(
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
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)
    safe_max = torch.where(valid_rows, global_max, 0.0)
    local_exp = torch.where(local_members, (local_values - safe_max.unsqueeze(1)).exp(), 0.0)
    entropy_values = local_values if entropy_requires_grad else local_values.detach()
    entropy_exp = local_exp if entropy_requires_grad else local_exp.detach()
    shifted_values = torch.where(local_members, entropy_values - safe_max.unsqueeze(1), 0.0)
    local_stats = torch.stack(
        (
            local_exp.sum(dim=-1),
            local_sampled,
            (entropy_exp * shifted_values).sum(dim=-1),
        )
    )
    global_stats = local_stats.detach().clone()
    dist.all_reduce(global_stats, op=dist.ReduceOp.SUM, group=tp_group)
    global_stats = global_stats + local_stats - local_stats.detach()
    denominator, sampled_score, shifted_score_sum = global_stats
    logprobs = sampled_score - safe_max - torch.where(valid_rows, denominator, 1.0).log()
    logprobs = torch.where(valid_rows, logprobs, 0.0)
    entropy_denominator = denominator if entropy_requires_grad else denominator.detach()
    shifted_score_sum = shifted_score_sum if entropy_requires_grad else shifted_score_sum.detach()
    safe_denominator = torch.where(valid_rows, entropy_denominator, 1.0)
    entropy = safe_denominator.log() - shifted_score_sum / safe_denominator
    entropy = torch.where(valid_rows, entropy, 0.0)
    return (
        logprobs.reshape(sampled_ids.shape),
        entropy.reshape(sampled_ids.shape),
        valid_rows.reshape(sampled_ids.shape),
    )


def _reference_scores(
    full_logits: torch.Tensor,
    sampled_ids: torch.Tensor,
    ids: TensorList,
    offsets: TensorList,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_ids = ids.tensors[0].long()
    flat_offsets = offsets.tensors[0].long()
    logprobs = []
    entropies = []
    for row in range(sampled_ids.numel()):
        members = flat_ids[flat_offsets[row] : flat_offsets[row + 1]]
        member_logits = full_logits[row, members]
        member_logprobs = torch.log_softmax(member_logits, dim=0)
        sampled_position = torch.nonzero(members == sampled_ids.reshape(-1)[row], as_tuple=False)[0, 0]
        logprobs.append(member_logprobs[sampled_position])
        entropies.append(-(member_logprobs.exp() * member_logprobs).sum())
    return torch.stack(logprobs).unsqueeze(0), torch.stack(entropies).unsqueeze(0)


def _gather_vocab_shards(local: torch.Tensor, group: dist.ProcessGroup) -> list[torch.Tensor]:
    shards = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(shards, local.detach(), group=group)
    return shards


def _correctness(group: dist.ProcessGroup, device: torch.device) -> dict[str, float]:
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    num_tokens = 19
    hidden_size = 32
    vocab_size = 128
    local_vocab = vocab_size // world_size
    vocab_start = rank * local_vocab
    vocab_end = vocab_start + local_vocab
    sampled, row_ids, dense_ids, ids, offsets, _ = _make_support(num_tokens, vocab_size, 0.7, 7, device)

    logits = torch.randn(1, num_tokens, local_vocab, device=device, dtype=torch.float32, requires_grad=True)
    actual = sample_support_csr_scores(
        logits,
        sampled,
        row_ids,
        ids,
        offsets,
        num_tokens,
        vocab_start_index=vocab_start,
        vocab_end_index=vocab_end,
        tp_group=group,
        compute_entropy=True,
        entropy_requires_grad=True,
    )
    assert actual.entropy is not None
    actual_logprobs = actual.logprobs
    actual_entropy = actual.entropy
    full_logits = torch.cat(_gather_vocab_shards(logits, group), dim=-1).squeeze(0)
    expected_logprobs, expected_entropy = _reference_scores(full_logits, sampled, ids, offsets)
    dense_logprobs, dense_entropy, dense_valid = _pr80_dense_sample_support_logprobs_and_entropy(
        logits,
        sampled,
        dense_ids,
        vocab_start_index=vocab_start,
        vocab_end_index=vocab_end,
        tp_group=group,
        compute_entropy=True,
        entropy_requires_grad=True,
    )
    value_error = max(
        (actual_logprobs - expected_logprobs).abs().max().item(),
        (actual_entropy - expected_entropy).abs().max().item(),
    )
    dense_value_error = max(
        (dense_logprobs - expected_logprobs).abs().max().item(),
        (dense_entropy - expected_entropy).abs().max().item(),
    )
    assert actual.valid_mask.all() and dense_valid.all()
    torch.testing.assert_close(actual_logprobs, expected_logprobs, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_entropy, expected_entropy, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dense_logprobs, expected_logprobs, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dense_entropy, expected_entropy, rtol=1e-5, atol=1e-5)

    (actual_logprobs + actual_entropy).sum().backward()
    actual_logits_grad = logits.grad.detach().clone()
    reference_local_logits = logits.detach().clone().requires_grad_(True)
    reference_shards = _gather_vocab_shards(logits, group)
    reference_shards[rank] = reference_local_logits
    reference_full_logits = torch.cat(reference_shards, dim=-1).squeeze(0)
    reference_logprobs, reference_entropy = _reference_scores(reference_full_logits, sampled, ids, offsets)
    (reference_logprobs + reference_entropy).sum().backward()
    logits_grad_error = (actual_logits_grad - reference_local_logits.grad).abs().max().item()
    torch.testing.assert_close(actual_logits_grad, reference_local_logits.grad, rtol=1e-5, atol=1e-5)

    hidden = torch.randn(1, num_tokens, hidden_size, device=device, dtype=torch.float32)
    dist.broadcast(hidden, src=0, group=group)
    hidden.requires_grad_()
    weight = torch.randn(local_vocab, hidden_size, device=device, dtype=torch.float32, requires_grad=True)
    fused = sample_support_csr_scores(
        hidden,
        sampled,
        row_ids,
        ids,
        offsets,
        num_tokens,
        vocab_start_index=vocab_start,
        vocab_end_index=vocab_end,
        tp_group=group,
        compute_entropy=True,
        entropy_requires_grad=True,
        lm_head_weight=weight,
        chunk_size=64,
    )
    assert fused.entropy is not None
    fused_logprobs = fused.logprobs
    fused_entropy = fused.entropy
    full_weight = torch.cat(_gather_vocab_shards(weight, group), dim=0)
    expected_fused_logprobs, expected_fused_entropy = _reference_scores(
        hidden.detach().squeeze(0) @ full_weight.T,
        sampled,
        ids,
        offsets,
    )
    fused_value_error = max(
        (fused_logprobs - expected_fused_logprobs).abs().max().item(),
        (fused_entropy - expected_fused_entropy).abs().max().item(),
    )
    torch.testing.assert_close(fused_logprobs, expected_fused_logprobs, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fused_entropy, expected_fused_entropy, rtol=1e-5, atol=1e-5)

    (fused_logprobs + fused_entropy).sum().backward()
    actual_weight_grad = weight.grad.detach().clone()
    actual_hidden_grad = hidden.grad.detach().clone()
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    reference_weight_shards = _gather_vocab_shards(weight, group)
    reference_weight_shards[rank] = reference_weight
    reference_full_weight = torch.cat(reference_weight_shards, dim=0)
    reference_fused_logprobs, reference_fused_entropy = _reference_scores(
        reference_hidden.squeeze(0) @ reference_full_weight.T,
        sampled,
        ids,
        offsets,
    )
    (reference_fused_logprobs + reference_fused_entropy).sum().backward()
    weight_grad_error = (actual_weight_grad - reference_weight.grad).abs().max().item()
    torch.testing.assert_close(actual_weight_grad, reference_weight.grad, rtol=2e-5, atol=2e-5)
    dist.all_reduce(actual_hidden_grad, op=dist.ReduceOp.SUM, group=group)
    hidden_grad_error = (actual_hidden_grad - reference_hidden.grad).abs().max().item()
    torch.testing.assert_close(actual_hidden_grad, reference_hidden.grad, rtol=2e-5, atol=2e-5)

    return {
        "value_max_abs": value_error,
        "pr80_dense_value_max_abs": dense_value_error,
        "logits_grad_max_abs": logits_grad_error,
        "fused_value_max_abs": fused_value_error,
        "fused_weight_grad_max_abs": weight_grad_error,
        "fused_hidden_grad_sum_max_abs": hidden_grad_error,
    }


def _measure(
    function,
    warmup: int,
    iterations: int,
    group: dist.ProcessGroup,
    device: torch.device,
) -> dict[str, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    samples = []
    for _ in range(iterations):
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        start = time.perf_counter()
        function()
        torch.cuda.synchronize(device)
        elapsed = torch.tensor((time.perf_counter() - start) * 1_000, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
        samples.append(elapsed.item())

    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    function()
    torch.cuda.synchronize(device)
    peak = torch.tensor(torch.cuda.max_memory_allocated(device) - baseline, dtype=torch.float64, device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX, group=group)
    return {
        "median_ms": statistics.median(samples),
        "peak_temporary_mib": peak.item() / 2**20,
    }


def _full_vocab_entropy_from_hidden(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    group: dist.ProcessGroup,
    chunk_size: int,
) -> torch.Tensor:
    """Production-equivalent fused full-vocabulary entropy metric."""
    output = torch.empty(hidden.shape[:-1], dtype=torch.float32, device=hidden.device)
    for start in range(0, hidden.shape[1], chunk_size):
        end = min(start + chunk_size, hidden.shape[1])
        logits = (hidden[:, start:end].to(weight.dtype) @ weight.T).float()
        logits_max = logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=group)
        exp_logits = (logits - logits_max).exp()
        sum_exp = exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(sum_exp, group=group)
        weighted_logits = ((exp_logits / sum_exp) * logits).sum(dim=-1, keepdim=True)
        dist.all_reduce(weighted_logits, group=group)
        output[:, start:end] = (logits_max + sum_exp.log() - weighted_logits).squeeze(-1)
    return output


def _benchmark_case(args, singleton_fraction: float, group: dist.ProcessGroup, device: torch.device) -> dict:
    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        vocab_parallel_entropy,
    )

    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    local_vocab = args.vocab_size // world_size
    vocab_start = rank * local_vocab
    vocab_end = vocab_start + local_vocab
    sampled, row_ids, dense_ids, ids, offsets, member_rows = _make_support(
        args.tokens,
        args.vocab_size,
        singleton_fraction,
        args.max_support,
        device,
    )
    row_sizes = offsets.tensors[0][1:].long() - offsets.tensors[0][:-1].long()
    member_ids = ids.tensors[0].long()
    active_members = row_sizes[member_rows] > 1
    local_members = active_members & (member_ids >= vocab_start) & (member_ids < vocab_end)
    dense_pairs = torch.tensor(
        args.tokens * (args.max_support + 1) * world_size,
        dtype=torch.float64,
        device=device,
    )
    csr_pairs = local_members.sum(dtype=torch.float64)
    dist.all_reduce(csr_pairs, op=dist.ReduceOp.SUM, group=group)

    hidden = torch.randn(
        1,
        args.tokens,
        args.hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    dist.broadcast(hidden, src=0, group=group)
    hidden.requires_grad_()
    weight = torch.randn(
        local_vocab,
        args.hidden_size,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    logits = torch.randn(
        1,
        args.tokens,
        local_vocab,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )

    def pr61_dense_replay_backward():
        hidden.grad = None
        weight.grad = None
        logprobs, _ = _pr61_dense_sample_support_logprobs(
            hidden,
            sampled,
            dense_ids,
            vocab_start_index=vocab_start,
            vocab_end_index=vocab_end,
            tp_group=group,
            lm_head_weight=weight,
            chunk_size=args.candidate_chunk_size,
        )
        (-logprobs.mean()).backward()

    def pr62_csr_replay_backward():
        hidden.grad = None
        weight.grad = None
        logprobs = sample_support_csr_scores(
            hidden,
            sampled,
            row_ids,
            ids,
            offsets,
            args.tokens,
            vocab_start_index=vocab_start,
            vocab_end_index=vocab_end,
            tp_group=group,
            lm_head_weight=weight,
            chunk_size=args.candidate_chunk_size,
            compute_entropy=False,
            entropy_requires_grad=False,
        ).logprobs
        (-logprobs.mean()).backward()

    def pr61_full_vocab_logits_entropy_backward():
        logits.grad = None
        logprobs, _ = _pr61_dense_sample_support_logprobs(
            logits,
            sampled,
            dense_ids,
            vocab_start_index=vocab_start,
            vocab_end_index=vocab_end,
            tp_group=group,
        )
        entropy = vocab_parallel_entropy(logits, chunk_size=args.entropy_chunk_size)
        (-(logprobs.mean()) - args.entropy_coefficient * entropy.mean()).backward()

    def pr80_support_logits_entropy_backward():
        logits.grad = None
        logprobs, entropy, _ = _pr80_dense_sample_support_logprobs_and_entropy(
            logits,
            sampled,
            dense_ids,
            vocab_start_index=vocab_start,
            vocab_end_index=vocab_end,
            tp_group=group,
            entropy_requires_grad=True,
        )
        (-(logprobs.mean()) - args.entropy_coefficient * entropy.mean()).backward()

    def pr61_full_vocab_fused_entropy_metric():
        with torch.no_grad():
            _pr61_dense_sample_support_logprobs(
                hidden,
                sampled,
                dense_ids,
                vocab_start_index=vocab_start,
                vocab_end_index=vocab_end,
                tp_group=group,
                lm_head_weight=weight,
                chunk_size=args.candidate_chunk_size,
            )
            _full_vocab_entropy_from_hidden(
                hidden,
                weight,
                group,
                args.entropy_chunk_size,
            )

    def pr80_support_fused_entropy_metric():
        with torch.no_grad():
            _pr80_dense_sample_support_logprobs_and_entropy(
                hidden,
                sampled,
                dense_ids,
                vocab_start_index=vocab_start,
                vocab_end_index=vocab_end,
                tp_group=group,
                entropy_requires_grad=False,
                lm_head_weight=weight,
                chunk_size=args.candidate_chunk_size,
            )

    measurements = {
        "pr61_dense_replay_fwd_bwd": _measure(
            pr61_dense_replay_backward,
            args.warmup,
            args.iterations,
            group,
            device,
        ),
        "pr62_csr_replay_fwd_bwd": _measure(
            pr62_csr_replay_backward,
            args.warmup,
            args.iterations,
            group,
            device,
        ),
        "pr61_full_vocab_logits_entropy_fwd_bwd": _measure(
            pr61_full_vocab_logits_entropy_backward,
            args.warmup,
            args.iterations,
            group,
            device,
        ),
        "pr80_support_logits_entropy_fwd_bwd": _measure(
            pr80_support_logits_entropy_backward,
            args.warmup,
            args.iterations,
            group,
            device,
        ),
        "pr61_full_vocab_fused_entropy_metric": _measure(
            pr61_full_vocab_fused_entropy_metric,
            args.warmup,
            args.iterations,
            group,
            device,
        ),
        "pr80_support_fused_entropy_metric": _measure(
            pr80_support_fused_entropy_metric,
            args.warmup,
            args.iterations,
            group,
            device,
        ),
    }
    measurements["pr61_dense_to_pr62_csr_speedup"] = (
        measurements["pr61_dense_replay_fwd_bwd"]["median_ms"] / measurements["pr62_csr_replay_fwd_bwd"]["median_ms"]
    )
    measurements["pr80_logits_entropy_speedup"] = (
        measurements["pr61_full_vocab_logits_entropy_fwd_bwd"]["median_ms"]
        / measurements["pr80_support_logits_entropy_fwd_bwd"]["median_ms"]
    )
    measurements["pr80_fused_entropy_metric_speedup"] = (
        measurements["pr61_full_vocab_fused_entropy_metric"]["median_ms"]
        / measurements["pr80_support_fused_entropy_metric"]["median_ms"]
    )
    return {
        "singleton_fraction": singleton_fraction,
        "mean_support_size": ids.tensors[0].numel() / args.tokens,
        "pr61_dense_projected_pairs_global": int(dense_pairs.item()),
        "pr62_csr_projected_pairs_global": int(csr_pairs.item()),
        "projected_pair_reduction": dense_pairs.item() / max(1.0, csr_pairs.item()),
        "measurements": measurements,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--vocab-size", type=int, default=131072)
    parser.add_argument("--singleton-fractions", type=float, nargs="+", default=[0.8, 0.9, 0.95])
    parser.add_argument("--max-support", type=int, default=32)
    parser.add_argument("--candidate-chunk-size", type=int, default=8192)
    parser.add_argument("--entropy-chunk-size", type=int, default=128)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    local_rank = int(__import__("os").environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    world_size = dist.get_world_size(group)
    if args.vocab_size % world_size:
        raise ValueError("vocab size must divide the TP world size")

    import megatron.core.parallel_state as mpu

    mpu.initialize_model_parallel(tensor_model_parallel_size=world_size)
    correctness = _correctness(group, device)
    cases = [_benchmark_case(args, fraction, group, device) for fraction in args.singleton_fractions]
    result = {
        "world_size": world_size,
        "gpu": torch.cuda.get_device_name(device),
        "tokens": args.tokens,
        "hidden_size": args.hidden_size,
        "vocab_size": args.vocab_size,
        "correctness": correctness,
        "cases": cases,
    }
    if dist.get_rank(group) == 0:
        print("SAMPLE_SUPPORT_BENCHMARK_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
