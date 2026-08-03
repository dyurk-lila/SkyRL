"""Benchmark dense and CSR sample-support replay through the production helper.

CUDA example (Megatron is needed for packed fused-LM-head replay)::

    uv run --isolated --extra megatron python -m \
        skyrl.benchmarks.bench_sample_support_replay --device cuda

Small CPU smoke::

    uv run --isolated --extra skyrl-train python -m \
        skyrl.benchmarks.bench_sample_support_replay --device cpu \
        --top-k-values 8 --densities 0.25 --chunk-sizes 256 --iterations 1
"""

import argparse
import time

import numpy as np
import torch

from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import (
    TokenMetadataLayout,
)
from skyrl.backends.skyrl_train.training_batch import TensorList
from skyrl.backends.skyrl_train.utils.sample_support_replay import (
    compute_sample_support_scores,
)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _median_ms(fn, iterations: int, device: torch.device) -> float:
    samples = []
    for _ in range(iterations):
        _synchronize(device)
        start = time.perf_counter()
        fn()
        _synchronize(device)
        samples.append((time.perf_counter() - start) * 1_000)
    return float(np.median(samples))


def _make_dense_support(
    batch_size: int,
    sequence_length: int,
    top_k: int,
    density: float,
    vocab_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    max_members = max(1, round(top_k * density))
    row_sizes = (
        np.full((batch_size, sequence_length), top_k)
        if density >= 1.0
        else rng.integers(1, max_members + 1, size=(batch_size, sequence_length))
    )
    support = np.full((batch_size, sequence_length, top_k), -1, dtype=np.int32)
    candidates = rng.integers(0, vocab_size, size=support.shape, dtype=np.int32)
    valid = np.arange(top_k)[None, None, :] < row_sizes[:, :, None]
    support[valid] = candidates[valid]

    sequences = support[:, :, 0].astype(np.int64)
    # Exercise the production exception for the loss-bearing EOS that SkyRL
    # appends after vLLM generation and therefore has no recorded support row.
    support[:, -1] = -1
    sequences[:, -1] = min(2, vocab_size - 1)
    return support, sequences


def _build_dense_training_tensor(trajectories: list[np.ndarray], sequence_length: int) -> torch.Tensor:
    top_k = trajectories[0].shape[1]
    output = torch.full((len(trajectories), sequence_length, top_k), -1, dtype=torch.int32)
    for sample_index, sample in enumerate(trajectories):
        output[sample_index, sequence_length - sample.shape[0] :] = torch.from_numpy(sample)
    return output


def _build_csr_training_tensors(trajectories: list[np.ndarray]) -> tuple[TensorList, TensorList]:
    ids = []
    offsets = []
    for sample in trajectories:
        valid = sample >= 0
        row_sizes = valid.sum(axis=1, dtype=np.int32)
        sample_offsets = np.empty(sample.shape[0] + 1, dtype=np.int32)
        sample_offsets[0] = 0
        np.cumsum(row_sizes, out=sample_offsets[1:])
        ids.append(torch.from_numpy(np.ascontiguousarray(sample[valid])))
        offsets.append(torch.from_numpy(sample_offsets))
    return TensorList(ids), TensorList(offsets)


def _tensor_bytes(value: torch.Tensor | TensorList) -> int:
    tensors = value.tensors if isinstance(value, TensorList) else [value]
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _peak_temporary_bytes(fn, device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    torch.cuda.empty_cache()
    _synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    _synchronize(device)
    return torch.cuda.max_memory_allocated(device) - baseline


def _packed_layout(batch_size: int, sequence_length: int, device: torch.device) -> TokenMetadataLayout:
    attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.bool, device=device)
    padded_lengths = [sequence_length] * batch_size
    cu_seqlens_padded = torch.arange(
        0,
        (batch_size + 1) * sequence_length,
        sequence_length,
        dtype=torch.int32,
        device=device,
    )
    return TokenMetadataLayout(
        attention_mask=attention_mask,
        sequence_lengths=padded_lengths,
        aligned_sequence_length=batch_size * sequence_length,
        padded_sequence_lengths=padded_lengths,
        cu_seqlens_padded=cu_seqlens_padded,
    )


def _run_case(args, top_k: int, density: float, chunk_size: int, device: torch.device) -> None:
    support_np, sequences_np = _make_dense_support(
        args.batch_size,
        args.sequence_length,
        top_k,
        density,
        args.vocab_size,
    )
    trajectories = [np.ascontiguousarray(sample) for sample in support_np]
    dense_cpu = _build_dense_training_tensor(trajectories, args.sequence_length)
    csr_ids_cpu, csr_offsets_cpu = _build_csr_training_tensors(trajectories)

    dense_build_ms = _median_ms(
        lambda: _build_dense_training_tensor(trajectories, args.sequence_length),
        args.iterations,
        torch.device("cpu"),
    )
    csr_build_ms = _median_ms(
        lambda: _build_csr_training_tensors(trajectories),
        args.iterations,
        torch.device("cpu"),
    )

    dense_pageable_h2d_ms = 0.0
    csr_pageable_h2d_ms = 0.0
    dense_pin_ms = 0.0
    csr_pin_ms = 0.0
    dense_pinned_h2d_ms = 0.0
    csr_pinned_h2d_ms = 0.0
    if device.type == "cuda":
        # This is the transfer behavior used by TensorBatch.to today. CSR makes
        # two small blocking copies per sample through its two TensorLists.
        dense_pageable_h2d_ms = _median_ms(lambda: dense_cpu.to(device), args.iterations, device)
        csr_pageable_h2d_ms = _median_ms(
            lambda: (csr_ids_cpu.to(device), csr_offsets_cpu.to(device)),
            args.iterations,
            device,
        )

        dense_pin_ms = _median_ms(lambda: dense_cpu.pin_memory(), args.iterations, torch.device("cpu"))
        csr_pin_ms = _median_ms(
            lambda: (csr_ids_cpu.pin_memory(), csr_offsets_cpu.pin_memory()),
            args.iterations,
            torch.device("cpu"),
        )
        dense_pinned = dense_cpu.pin_memory()
        csr_ids_pinned = csr_ids_cpu.pin_memory()
        csr_offsets_pinned = csr_offsets_cpu.pin_memory()
        # Report this separately: it is an optimization opportunity, not the
        # current production transfer path.
        dense_pinned_h2d_ms = _median_ms(
            lambda: dense_pinned.to(device, non_blocking=True),
            args.iterations,
            device,
        )
        csr_pinned_h2d_ms = _median_ms(
            lambda: (
                csr_ids_pinned.to(device, non_blocking=True),
                csr_offsets_pinned.to(device, non_blocking=True),
            ),
            args.iterations,
            device,
        )

    dense_support = dense_cpu.to(device)
    csr_ids = csr_ids_cpu.to(device)
    csr_offsets = csr_offsets_cpu.to(device)
    sequences = torch.from_numpy(sequences_np).to(device)
    loss_mask = torch.ones(
        (args.batch_size, args.sequence_length - 1),
        dtype=torch.bool,
        device=device,
    )
    packed = args.layout == "packed"
    metadata_layout = _packed_layout(args.batch_size, args.sequence_length, device) if packed else None
    source_prefix = (1, args.batch_size * args.sequence_length) if packed else sequences.shape
    source_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if args.source == "hidden":
        source = torch.randn(
            (*source_prefix, args.hidden_size),
            dtype=source_dtype,
            device=device,
            requires_grad=True,
        )
        lm_head_weight = torch.randn(
            (args.vocab_size, args.hidden_size),
            dtype=source_dtype,
            device=device,
            requires_grad=True,
        )
    else:
        source = torch.randn(
            (*source_prefix, args.vocab_size),
            dtype=source_dtype,
            device=device,
            requires_grad=True,
        )
        lm_head_weight = None

    def replay(*, sparse: bool) -> torch.Tensor:
        return compute_sample_support_scores(
            source,
            sequences,
            loss_mask,
            None if sparse else dense_support,
            csr_ids if sparse else None,
            csr_offsets if sparse else None,
            num_actions=args.sequence_length - 1,
            packed=packed,
            metadata_layout=metadata_layout,
            vocab_start_index=0,
            vocab_end_index=args.vocab_size,
            tp_group=None,
            inference_only=False,
            lm_head_weight=lm_head_weight,
            temperature=args.temperature,
            chunk_size=chunk_size,
            fused_backend="torch",
            compute_entropy=False,
            entropy_requires_grad=False,
        ).logprobs

    with torch.no_grad():
        torch.testing.assert_close(replay(sparse=True), replay(sparse=False), rtol=2e-2, atol=2e-2)
        for _ in range(args.warmup):
            replay(sparse=False)
            replay(sparse=True)

    def forward_only(sparse: bool) -> None:
        with torch.no_grad():
            replay(sparse=sparse)

    dense_forward_ms = _median_ms(lambda: forward_only(False), args.iterations, device)
    csr_forward_ms = _median_ms(lambda: forward_only(True), args.iterations, device)

    def forward_backward(sparse: bool) -> None:
        source.grad = None
        if lm_head_weight is not None:
            lm_head_weight.grad = None
        replay(sparse=sparse).sum().backward()

    for _ in range(args.warmup):
        forward_backward(False)
        forward_backward(True)
    dense_forward_backward_ms = _median_ms(lambda: forward_backward(False), args.iterations, device)
    csr_forward_backward_ms = _median_ms(lambda: forward_backward(True), args.iterations, device)
    source.grad = None
    if lm_head_weight is not None:
        lm_head_weight.grad = None
    dense_peak_bytes = _peak_temporary_bytes(lambda: forward_backward(False), device)
    source.grad = None
    if lm_head_weight is not None:
        lm_head_weight.grad = None
    csr_peak_bytes = _peak_temporary_bytes(lambda: forward_backward(True), device)

    dense_bytes = _tensor_bytes(dense_support)
    csr_bytes = _tensor_bytes(csr_ids) + _tensor_bytes(csr_offsets)
    mean_members = sum(tensor.numel() for tensor in csr_ids.tensors) / (args.batch_size * args.sequence_length)
    print(
        f"top_k={top_k} density={density:.2f} chunk={chunk_size} mean_members={mean_members:.1f} "
        f"storage_dense={dense_bytes / 2**20:.2f}MiB storage_csr={csr_bytes / 2**20:.2f}MiB "
        f"build_dense={dense_build_ms:.2f}ms build_csr={csr_build_ms:.2f}ms "
        f"pageable_blocking_h2d_dense={dense_pageable_h2d_ms:.2f}ms "
        f"pageable_blocking_h2d_csr={csr_pageable_h2d_ms:.2f}ms "
        f"pin_dense={dense_pin_ms:.2f}ms pin_csr={csr_pin_ms:.2f}ms "
        f"pinned_async_h2d_dense={dense_pinned_h2d_ms:.2f}ms "
        f"pinned_async_h2d_csr={csr_pinned_h2d_ms:.2f}ms "
        f"full_forward_dense={dense_forward_ms:.2f}ms full_forward_csr={csr_forward_ms:.2f}ms "
        f"full_fwd_bwd_dense={dense_forward_backward_ms:.2f}ms "
        f"full_fwd_bwd_csr={csr_forward_backward_ms:.2f}ms "
        f"full_peak_dense={dense_peak_bytes / 2**20:.2f}MiB "
        f"full_peak_csr={csr_peak_bytes / 2**20:.2f}MiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--top-k-values", type=int, nargs="+", default=[8, 64])
    parser.add_argument("--densities", type=float, nargs="+", default=[1.0, 0.25, 0.1])
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[1024, 4096, 16384])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--layout", choices=("packed", "unpacked"))
    parser.add_argument("--source", choices=("hidden", "logits"))
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    args = parser.parse_args()

    device = torch.device(args.device)
    cuda = device.type == "cuda"
    args.batch_size = args.batch_size if args.batch_size is not None else (2 if cuda else 1)
    args.sequence_length = args.sequence_length if args.sequence_length is not None else (2048 if cuda else 64)
    args.vocab_size = args.vocab_size if args.vocab_size is not None else (32768 if cuda else 1024)
    args.hidden_size = args.hidden_size if args.hidden_size is not None else (4096 if cuda else 128)
    args.layout = args.layout or ("packed" if cuda else "unpacked")
    args.source = args.source or ("hidden" if cuda else "logits")
    args.warmup = args.warmup if args.warmup is not None else (3 if cuda else 1)
    args.iterations = args.iterations if args.iterations is not None else (10 if cuda else 3)

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.sequence_length < 2:
        parser.error("--sequence-length must be at least 2")
    if args.vocab_size < 1:
        parser.error("--vocab-size must be positive")
    if args.hidden_size < 1:
        parser.error("--hidden-size must be positive")
    if not 0 < args.temperature:
        parser.error("--temperature must be positive")
    if any(not 0 < density <= 1 for density in args.densities):
        parser.error("--densities must be in (0, 1]")
    if any(top_k < 1 for top_k in args.top_k_values):
        parser.error("--top-k-values must be positive")

    print(
        f"device={device} layout={args.layout} source={args.source} batch={args.batch_size} "
        f"sequence_length={args.sequence_length} vocab={args.vocab_size} hidden={args.hidden_size} "
        f"temperature={args.temperature} path=full_compute_sample_support_scores"
    )
    for top_k in args.top_k_values:
        for density in args.densities:
            for chunk_size in args.chunk_sizes:
                _run_case(args, top_k, density, chunk_size, device)


if __name__ == "__main__":
    main()
