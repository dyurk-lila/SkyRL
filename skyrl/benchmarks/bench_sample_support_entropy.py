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
    _project_candidate_pairs,
    sample_support_csr_logprobs,
    sample_support_csr_logprobs_and_entropy,
)


def _make_support(
    num_tokens: int,
    vocab_size: int,
    singleton_fraction: float,
    max_support: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, TensorList, TensorList, torch.Tensor]:
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
    return (
        sampled_ids.unsqueeze(0).to(device),
        torch.arange(num_tokens, device=device).unsqueeze(0),
        TensorList([ids.to(device)]),
        TensorList([offsets.to(device)]),
        row_ids_for_members.to(device),
    )


def _legacy_assemble(
    sample_support_ids: TensorList,
    sample_support_offsets: TensorList,
    grid_width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validity = torch.zeros(len(sample_support_ids.tensors) * grid_width, dtype=torch.bool, device=device)
    member_rows = []
    member_vocab = []
    for sample_index, (ids, offsets) in enumerate(
        zip(sample_support_ids.tensors, sample_support_offsets.tensors, strict=True)
    ):
        num_rows = offsets.numel() - 1
        canonical_rows = sample_index * grid_width + torch.arange(num_rows, device=device)
        row_sizes = offsets[1:].long() - offsets[:-1].long()
        validity[canonical_rows] = row_sizes > 0
        member_rows.append(torch.repeat_interleave(canonical_rows, row_sizes, output_size=ids.numel()))
        member_vocab.append(ids.long())
    empty = torch.empty(0, dtype=torch.long, device=device)
    return (
        torch.cat(member_rows) if member_rows else empty,
        torch.cat(member_vocab) if member_vocab else empty,
        validity,
    )


def _legacy_sample_support_csr_logprobs(
    logits_or_hidden: torch.Tensor,
    sampled_ids: torch.Tensor,
    support_row_ids: torch.Tensor,
    sample_support_ids: TensorList,
    sample_support_offsets: TensorList,
    support_grid_width: int,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group: dist.ProcessGroup,
    lm_head_weight: torch.Tensor | None = None,
    temperature: float = 1.0,
    chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CSR implementation immediately before PR 62's singleton fast path."""
    flat_source = logits_or_hidden.reshape(-1, logits_or_hidden.shape[-1])
    flat_sampled = sampled_ids.reshape(-1).long()
    aligned_rows = support_row_ids.reshape(-1).long()
    member_rows, member_vocab, canonical_validity = _legacy_assemble(
        sample_support_ids,
        sample_support_offsets,
        support_grid_width,
        logits_or_hidden.device,
    )
    num_canonical_rows = canonical_validity.numel()
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
    local_member_mask = (member_positions >= 0) & (member_vocab >= vocab_start_index) & (member_vocab < vocab_end_index)
    local_member_positions = member_positions.clamp_min(0)
    local_member_ids = (member_vocab - vocab_start_index).clamp(0, vocab_end_index - vocab_start_index - 1)
    compute_dtype = (
        torch.float32 if logits_or_hidden.dtype in (torch.float16, torch.bfloat16) else logits_or_hidden.dtype
    )
    if lm_head_weight is None:
        local_member_values = flat_source[local_member_positions, local_member_ids].to(compute_dtype)
    else:
        local_member_values = _project_candidate_pairs(
            flat_source,
            local_member_positions,
            local_member_ids,
            lm_head_weight,
            temperature,
            chunk_size,
        )
    local_member_values = torch.where(local_member_mask, local_member_values, float("-inf"))

    local_max = local_member_values.new_full((num_canonical_rows,), float("-inf"))
    local_max.index_reduce_(0, member_rows, local_member_values.detach(), "amax", include_self=True)
    global_max = local_max.clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)
    safe_max = torch.where(canonical_validity, global_max, 0.0)

    local_sum = local_member_values.new_zeros(num_canonical_rows).index_add(
        0,
        member_rows,
        torch.where(local_member_mask, (local_member_values - safe_max[member_rows]).exp(), 0.0),
    )
    sampled_positions = canonical_positions.clamp_min(0)
    sampled_for_row = flat_sampled[sampled_positions]
    local_sample_mask = (
        canonical_validity
        & (canonical_positions >= 0)
        & (sampled_for_row >= vocab_start_index)
        & (sampled_for_row < vocab_end_index)
    )
    local_sample_ids = (sampled_for_row - vocab_start_index).clamp(0, vocab_end_index - vocab_start_index - 1)
    if lm_head_weight is None:
        local_sampled = flat_source[sampled_positions, local_sample_ids].to(compute_dtype)
    else:
        local_sampled = _project_candidate_pairs(
            flat_source,
            sampled_positions,
            local_sample_ids,
            lm_head_weight,
            temperature,
            chunk_size,
        )
    local_sampled = torch.where(local_sample_mask, local_sampled, 0.0)

    local_stats = torch.stack((local_sum, local_sampled))
    global_stats = local_stats.detach().clone()
    dist.all_reduce(global_stats, op=dist.ReduceOp.SUM, group=tp_group)
    global_stats = global_stats + local_stats - local_stats.detach()
    denominator, sampled_score = global_stats
    canonical_logprobs = sampled_score - safe_max - torch.where(canonical_validity, denominator, 1.0).log()
    aligned_logprobs = canonical_logprobs[safe_aligned_rows.clamp_max(num_canonical_rows - 1)]
    aligned_logprobs = torch.where(valid_support, aligned_logprobs, 0.0)
    return aligned_logprobs.reshape(sampled_ids.shape), valid_support.reshape(sampled_ids.shape)


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
    sampled, row_ids, ids, offsets, _ = _make_support(num_tokens, vocab_size, 0.7, 7, device)

    logits = torch.randn(1, num_tokens, local_vocab, device=device, dtype=torch.float32, requires_grad=True)
    actual_logprobs, actual_entropy, valid = sample_support_csr_logprobs_and_entropy(
        logits,
        sampled,
        row_ids,
        ids,
        offsets,
        num_tokens,
        vocab_start_index=vocab_start,
        vocab_end_index=vocab_end,
        tp_group=group,
        entropy_requires_grad=True,
    )
    full_logits = torch.cat(_gather_vocab_shards(logits, group), dim=-1).squeeze(0)
    expected_logprobs, expected_entropy = _reference_scores(full_logits, sampled, ids, offsets)
    value_error = max(
        (actual_logprobs - expected_logprobs).abs().max().item(),
        (actual_entropy - expected_entropy).abs().max().item(),
    )
    assert valid.all()
    torch.testing.assert_close(actual_logprobs, expected_logprobs, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_entropy, expected_entropy, rtol=1e-5, atol=1e-5)

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

    hidden = torch.randn(1, num_tokens, hidden_size, device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(local_vocab, hidden_size, device=device, dtype=torch.float32, requires_grad=True)
    fused_logprobs, fused_entropy, _ = sample_support_csr_logprobs_and_entropy(
        hidden,
        sampled,
        row_ids,
        ids,
        offsets,
        num_tokens,
        vocab_start_index=vocab_start,
        vocab_end_index=vocab_end,
        tp_group=group,
        entropy_requires_grad=True,
        lm_head_weight=weight,
        chunk_size=64,
    )
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


def _benchmark_case(args, singleton_fraction: float, group: dist.ProcessGroup, device: torch.device) -> dict:
    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        vocab_parallel_entropy,
    )
    from skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper import (
        _fused_vocab_parallel_entropy_from_hidden,
    )

    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    local_vocab = args.vocab_size // world_size
    vocab_start = rank * local_vocab
    vocab_end = vocab_start + local_vocab
    sampled, row_ids, ids, offsets, member_rows = _make_support(
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
    legacy_pairs = torch.tensor(member_ids.numel() + args.tokens, dtype=torch.float64, device=device)
    optimized_pairs = local_members.sum(dtype=torch.float64)
    dist.all_reduce(legacy_pairs, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(optimized_pairs, op=dist.ReduceOp.SUM, group=group)

    hidden = torch.randn(
        1,
        args.tokens,
        args.hidden_size,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
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

    def legacy_replay_backward():
        hidden.grad = None
        weight.grad = None
        logprobs, _ = _legacy_sample_support_csr_logprobs(
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
        )
        (-logprobs.mean()).backward()

    def optimized_replay_backward():
        hidden.grad = None
        weight.grad = None
        logprobs, _ = sample_support_csr_logprobs(
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
        )
        (-logprobs.mean()).backward()

    def old_logits_entropy_backward():
        logits.grad = None
        logprobs, _ = sample_support_csr_logprobs(
            logits,
            sampled,
            row_ids,
            ids,
            offsets,
            args.tokens,
            vocab_start_index=vocab_start,
            vocab_end_index=vocab_end,
            tp_group=group,
        )
        entropy = vocab_parallel_entropy(logits, chunk_size=args.entropy_chunk_size)
        (-(logprobs.mean()) - args.entropy_coefficient * entropy.mean()).backward()

    def support_logits_entropy_backward():
        logits.grad = None
        logprobs, entropy, _ = sample_support_csr_logprobs_and_entropy(
            logits,
            sampled,
            row_ids,
            ids,
            offsets,
            args.tokens,
            vocab_start_index=vocab_start,
            vocab_end_index=vocab_end,
            tp_group=group,
            entropy_requires_grad=True,
        )
        (-(logprobs.mean()) - args.entropy_coefficient * entropy.mean()).backward()

    def old_fused_entropy_metric():
        with torch.no_grad():
            sample_support_csr_logprobs(
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
            )
            _fused_vocab_parallel_entropy_from_hidden(
                hidden,
                weight,
                group,
                chunk_size=args.entropy_chunk_size,
            )

    def support_fused_entropy_metric():
        with torch.no_grad():
            sample_support_csr_logprobs_and_entropy(
                hidden,
                sampled,
                row_ids,
                ids,
                offsets,
                args.tokens,
                vocab_start_index=vocab_start,
                vocab_end_index=vocab_end,
                tp_group=group,
                entropy_requires_grad=False,
                lm_head_weight=weight,
                chunk_size=args.candidate_chunk_size,
            )

    measurements = {
        "legacy_csr_fwd_bwd": _measure(legacy_replay_backward, args.warmup, args.iterations, group, device),
        "optimized_csr_fwd_bwd": _measure(
            optimized_replay_backward,
            args.warmup,
            args.iterations,
            group,
            device,
        ),
        "full_vocab_logits_entropy_fwd_bwd": _measure(
            old_logits_entropy_backward,
            args.warmup,
            args.iterations,
            group,
            device,
        ),
        "support_logits_entropy_fwd_bwd": _measure(
            support_logits_entropy_backward,
            args.warmup,
            args.iterations,
            group,
            device,
        ),
        "full_vocab_fused_entropy_metric": _measure(
            old_fused_entropy_metric,
            args.warmup,
            args.iterations,
            group,
            device,
        ),
        "support_fused_entropy_metric": _measure(
            support_fused_entropy_metric,
            args.warmup,
            args.iterations,
            group,
            device,
        ),
    }
    measurements["legacy_to_optimized_csr_speedup"] = (
        measurements["legacy_csr_fwd_bwd"]["median_ms"] / measurements["optimized_csr_fwd_bwd"]["median_ms"]
    )
    measurements["logits_entropy_speedup"] = (
        measurements["full_vocab_logits_entropy_fwd_bwd"]["median_ms"]
        / measurements["support_logits_entropy_fwd_bwd"]["median_ms"]
    )
    measurements["fused_entropy_metric_speedup"] = (
        measurements["full_vocab_fused_entropy_metric"]["median_ms"]
        / measurements["support_fused_entropy_metric"]["median_ms"]
    )
    return {
        "singleton_fraction": singleton_fraction,
        "mean_support_size": ids.tensors[0].numel() / args.tokens,
        "legacy_projected_pairs_global": int(legacy_pairs.item()),
        "optimized_projected_pairs_global": int(optimized_pairs.item()),
        "projected_pair_reduction": legacy_pairs.item() / max(1.0, optimized_pairs.item()),
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
