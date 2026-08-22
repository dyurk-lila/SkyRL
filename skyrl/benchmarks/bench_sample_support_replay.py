"""Fixed-width versus ragged sample-support replay, through the production scorer.

The baseline is the ``PackedTensor`` ``[rows, top_k]`` form the trainer carries TODAY: packed to
response tokens, so the prompt region and all inter-sequence padding are already gone. A
comparison against the old ``[batch, sequence_length, top_k]`` rectangle would re-bank a win that
already landed, and would roughly double every number here.

What is left to win is the padding INSIDE a row plus the rows that are padding all the way across.
Both depend on the per-token support-size distribution, which is a property of the run, not of the
code, so this benchmark takes it as input rather than assuming one. Every line is labelled with
where its distribution came from: ``ASSUMED`` unless ``--histogram`` supplied a measured one.

A histogram file is one ``<support size> <token count>`` pair per line, comments with ``#``; size
0 rows are the observation tokens and appended EOS that carry no support at all.

CUDA (a 24k-token microbatch at top_k=256, the shape the fp32 intermediates were sized against)::

    uv run --isolated --extra skyrl-train python -m \
        skyrl.benchmarks.bench_sample_support_replay --device cuda \
        --top-k-values 256 --histogram measured_support_sizes.txt

CPU smoke::

    uv run --isolated --extra skyrl-train python -m \
        skyrl.benchmarks.bench_sample_support_replay --device cpu --iterations 1
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import (
    TokenMetadataLayout,
)
from skyrl.backends.skyrl_train.utils.packed_ragged_tensor import PackedRaggedTensor
from skyrl.backends.skyrl_train.utils.packed_tensor import (
    PackedTensor,
    cu_seqlens_from_lengths,
)
from skyrl.backends.skyrl_train.utils.sample_support import SAMPLE_SUPPORT_PADDING
from skyrl.backends.skyrl_train.utils.sample_support_replay import (
    compute_sample_support_scores,
)

# A placeholder shape, NOT a measurement: a nucleus that usually collapses onto a few candidates,
# a long tail, and one row in five carrying nothing at all. Replace it with --histogram.
ASSUMED_SUPPORT_SIZES = (0, 1, 3, 8, 20)
ASSUMED_SUPPORT_WEIGHTS = (0.20, 0.25, 0.25, 0.20, 0.10)
# Prime, so it is coprime to any power-of-two vocabulary and walks distinct members.
_MEMBER_STRIDE = 977


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


def _read_histogram(path: Path) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Read ``<support size> <token count>`` pairs into sizes and normalized weights."""
    sizes, counts = [], []
    for line in path.read_text().splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        size, count = stripped.split()
        sizes.append(int(size))
        counts.append(float(count))
    if not sizes:
        raise ValueError(f"{path} holds no <support size> <token count> pairs")
    total = sum(counts)
    return tuple(sizes), tuple(count / total for count in counts)


def _row_sizes(rows: int, sizes, weights, top_k: int, seed: int) -> np.ndarray:
    """Draw one support size per token, clipped to ``top_k``.

    A size at or above ``top_k`` would mean capture bound the support before the nucleus did,
    which is the configuration that raises at generation time, so the clip is not a rounding
    convenience: it is the only shape the trainer can be handed.
    """
    rng = np.random.default_rng(seed)
    drawn = rng.choice(np.asarray(sizes), size=rows, p=np.asarray(weights))
    return np.minimum(drawn, top_k)


def _fixed_width_support(row_sizes: np.ndarray, segment_lengths, top_k: int, vocab_size: int) -> PackedTensor:
    """Build the packed ``[rows, top_k]`` field the trainer carries today.

    A row's members are a random start walked by a stride coprime to the vocabulary, so they are
    distinct -- a top-k set is -- and spread out rather than contiguous, which keeps the LM-head
    gather from getting a locality windfall neither arm would see in production.
    """
    if top_k > vocab_size:
        raise ValueError(f"top_k {top_k} exceeds the {vocab_size}-token vocabulary")
    rng = np.random.default_rng(1)
    rows = np.full((row_sizes.shape[0], top_k), SAMPLE_SUPPORT_PADDING, dtype=np.int32)
    starts = rng.integers(0, vocab_size, size=(row_sizes.shape[0], 1), dtype=np.int64)
    candidates = ((starts + _MEMBER_STRIDE * np.arange(top_k)[None, :]) % vocab_size).astype(np.int32)
    occupied = np.arange(top_k)[None, :] < row_sizes[:, None]
    rows[occupied] = candidates[occupied]
    return PackedTensor(torch.from_numpy(rows), cu_seqlens_from_lengths(segment_lengths))


def _sampled_sequences(support: PackedTensor, row_sizes: np.ndarray, segment_lengths, sequence_length: int):
    """Place each row's weakest member at the position whose logit predicts that token.

    vLLM drew from the recorded set and capture repairs any row that lost its sampled id, so a
    benchmark that sampled outside the support would measure a state replay never sees.
    """
    sequences = torch.zeros((len(segment_lengths), sequence_length), dtype=torch.long)
    row = 0
    for index, response in enumerate(segment_lengths):
        for offset in range(response):
            if row_sizes[row]:
                sequences[index, sequence_length - response + offset] = int(support.values[row, row_sizes[row] - 1])
            row += 1
    return sequences


def _field_bytes(field: PackedTensor | PackedRaggedTensor) -> int:
    """Every buffer the field ships to the worker."""
    if isinstance(field, PackedRaggedTensor):
        buffers = (field.values, field.row_offsets, field.cu_seqlens)
    else:
        buffers = (field.values, field.cu_seqlens)
    return sum(buffer.numel() * buffer.element_size() for buffer in buffers)


def _saved_activation_bytes(fn, inputs) -> int:
    """Bytes the autograd graph retains for backward, counting only fresh intermediates.

    Replay's activation cost, which a peak-allocator reading cannot separate from whatever else
    the process held. Two components make it up, and the ``fixed_row_matrix_fp32`` figure on the
    output line is there to tell them apart:

    * the ``[rows, top_k]`` fp32 intermediates the fixed-width scorer materializes whether or not
      the slots are occupied -- the part that scales with ``top_k``;
    * in the unfused logits arm, the flattened source itself, because ``gather`` retains its input
      and the ragged path's advanced index does not. That part is ``top_k``-independent, and it
      does not arise when the fused LM-head path is used.

    ``inputs`` and views of them are excluded: the forward that produced the hidden states holds
    them either way, so counting them would swamp what replay adds.
    """
    input_storages = {tensor.untyped_storage().data_ptr() for tensor in inputs if tensor is not None}
    total = 0
    seen: set[tuple[int, int]] = set()

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal total
        if tensor.untyped_storage().data_ptr() in input_storages:
            return tensor
        key = (tensor.data_ptr(), tensor.numel() * tensor.element_size())
        if key not in seen:
            seen.add(key)
            total += key[1]
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        fn()
    return total


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


def _packed_layout(segment_lengths, sequence_length: int, device: torch.device) -> TokenMetadataLayout:
    totals = [sequence_length] * len(segment_lengths)
    return TokenMetadataLayout(
        attention_mask=torch.ones((len(segment_lengths), sequence_length), dtype=torch.bool, device=device),
        sequence_lengths=totals,
        aligned_sequence_length=sum(totals),
        padded_sequence_lengths=totals,
        cu_seqlens_padded=cu_seqlens_from_lengths(totals, device=device),
    )


def _unpacked_layout(segment_lengths, sequence_length: int, device: torch.device) -> TokenMetadataLayout:
    return TokenMetadataLayout(
        attention_mask=torch.ones((len(segment_lengths), sequence_length), dtype=torch.bool, device=device),
        sequence_lengths=[sequence_length] * len(segment_lengths),
        aligned_sequence_length=sequence_length,
    )


def _run_case(args, top_k: int, chunk_size: int, device: torch.device, sizes, weights, label: str) -> None:
    # Support covers generated tokens only; a whole-response benchmark is the worst case for the
    # fixed-width form and the fairest one for it, since every row it holds is a real row.
    response_len = args.sequence_length - 1
    segment_lengths = [response_len] * args.batch_size
    drawn = _row_sizes(sum(segment_lengths), sizes, weights, top_k, args.seed).reshape(args.batch_size, response_len)
    # The EOS SkyRL appends is loss-bearing and has no recorded support -- exactly one per
    # trajectory, which is what the fallback's capacity permits. Every other empty row is an
    # observation or tool result, which is loss-masked. The ragged form frees both.
    drawn[:, -1] = 0
    loss_bearing = drawn > 0
    loss_bearing[:, -1] = True
    row_sizes = drawn.reshape(-1)
    fixed_cpu = _fixed_width_support(row_sizes, segment_lengths, top_k, args.vocab_size)
    sequences = _sampled_sequences(fixed_cpu, row_sizes, segment_lengths, args.sequence_length).to(device)

    compress_ms = _median_ms(
        lambda: PackedRaggedTensor.from_padded_rows(fixed_cpu, padding_value=SAMPLE_SUPPORT_PADDING),
        args.iterations,
        torch.device("cpu"),
    )
    ragged_cpu = PackedRaggedTensor.from_padded_rows(fixed_cpu, padding_value=SAMPLE_SUPPORT_PADDING)

    fixed_h2d_ms = ragged_h2d_ms = 0.0
    if device.type == "cuda":
        fixed_h2d_ms = _median_ms(lambda: fixed_cpu.to(device), args.iterations, device)
        ragged_h2d_ms = _median_ms(lambda: ragged_cpu.to(device), args.iterations, device)

    fixed_field = fixed_cpu.to(device)
    ragged_field = ragged_cpu.to(device)
    packed = args.layout == "packed"
    layout = (
        _packed_layout(segment_lengths, args.sequence_length, device)
        if packed
        else _unpacked_layout(segment_lengths, args.sequence_length, device)
    )
    loss_mask = torch.from_numpy(loss_bearing).to(device)
    source_prefix = (1, args.batch_size * args.sequence_length) if packed else (args.batch_size, args.sequence_length)
    source_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if args.source == "hidden":
        source = torch.randn((*source_prefix, args.hidden_size), dtype=source_dtype, device=device, requires_grad=True)
        lm_head_weight = torch.randn(
            (args.vocab_size, args.hidden_size), dtype=source_dtype, device=device, requires_grad=True
        )
    else:
        source = torch.randn((*source_prefix, args.vocab_size), dtype=source_dtype, device=device, requires_grad=True)
        lm_head_weight = None

    def replay(field) -> torch.Tensor:
        return compute_sample_support_scores(
            source,
            sequences,
            loss_mask,
            field,
            response_len,
            packed=packed,
            metadata_layout=layout,
            vocab_start_index=0,
            vocab_end_index=args.vocab_size,
            tp_group=None,
            inference_only=False,
            lm_head_weight=lm_head_weight,
            temperature=args.temperature,
            chunk_size=chunk_size,
            fused_backend="torch",
            compute_entropy=args.compute_entropy,
            entropy_requires_grad=False,
        ).logprobs

    with torch.no_grad():
        torch.testing.assert_close(replay(ragged_field), replay(fixed_field), rtol=args.tolerance, atol=args.tolerance)
        for _ in range(args.warmup):
            replay(fixed_field)
            replay(ragged_field)

    def forward_only(field) -> None:
        with torch.no_grad():
            replay(field)

    fixed_forward_ms = _median_ms(lambda: forward_only(fixed_field), args.iterations, device)
    ragged_forward_ms = _median_ms(lambda: forward_only(ragged_field), args.iterations, device)

    def forward_backward(field) -> None:
        source.grad = None
        if lm_head_weight is not None:
            lm_head_weight.grad = None
        replay(field).sum().backward()

    for _ in range(args.warmup):
        forward_backward(fixed_field)
        forward_backward(ragged_field)
    fixed_fwd_bwd_ms = _median_ms(lambda: forward_backward(fixed_field), args.iterations, device)
    ragged_fwd_bwd_ms = _median_ms(lambda: forward_backward(ragged_field), args.iterations, device)

    replay_inputs = (source, lm_head_weight)
    fixed_saved = _saved_activation_bytes(lambda: replay(fixed_field), replay_inputs)
    ragged_saved = _saved_activation_bytes(lambda: replay(ragged_field), replay_inputs)
    source.grad = None
    if lm_head_weight is not None:
        lm_head_weight.grad = None
    fixed_peak = _peak_temporary_bytes(lambda: forward_backward(fixed_field), device)
    source.grad = None
    if lm_head_weight is not None:
        lm_head_weight.grad = None
    ragged_peak = _peak_temporary_bytes(lambda: forward_backward(ragged_field), device)

    mib = float(2**20)
    print(
        f"top_k={top_k} chunk={chunk_size} distribution={label} "
        f"rows={row_sizes.shape[0]} empty_rows={int((row_sizes == 0).sum())} "
        f"mean_members={row_sizes.mean():.2f} occupancy={row_sizes.mean() / top_k:.3f} "
        f"transport_fixed={_field_bytes(fixed_field) / mib:.2f}MiB "
        f"transport_ragged={_field_bytes(ragged_field) / mib:.2f}MiB "
        f"compress_cpu={compress_ms:.2f}ms "
        f"h2d_fixed={fixed_h2d_ms:.2f}ms h2d_ragged={ragged_h2d_ms:.2f}ms "
        f"forward_fixed={fixed_forward_ms:.2f}ms forward_ragged={ragged_forward_ms:.2f}ms "
        f"fwd_bwd_fixed={fixed_fwd_bwd_ms:.2f}ms fwd_bwd_ragged={ragged_fwd_bwd_ms:.2f}ms "
        f"saved_activations_fixed={fixed_saved / mib:.2f}MiB "
        f"saved_activations_ragged={ragged_saved / mib:.2f}MiB "
        f"fixed_row_matrix_fp32={row_sizes.shape[0] * top_k * 4 / mib:.2f}MiB "
        f"cuda_peak_fixed={fixed_peak / mib:.2f}MiB cuda_peak_ragged={ragged_peak / mib:.2f}MiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--top-k-values", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[4096])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=2e-2)
    parser.add_argument("--layout", choices=("packed", "unpacked"))
    parser.add_argument("--source", choices=("hidden", "logits"))
    parser.add_argument("--compute-entropy", action="store_true")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--histogram",
        type=Path,
        help="measured per-token support sizes: '<support size> <token count>' per line",
    )
    parser.add_argument("--support-sizes", type=int, nargs="+", help="override the assumed size support")
    parser.add_argument("--support-weights", type=float, nargs="+", help="probability of each --support-sizes entry")
    args = parser.parse_args()

    device = torch.device(args.device)
    cuda = device.type == "cuda"
    args.batch_size = args.batch_size if args.batch_size is not None else (4 if cuda else 2)
    args.sequence_length = args.sequence_length if args.sequence_length is not None else (6144 if cuda else 64)
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
    if args.vocab_size < 1 or args.hidden_size < 1:
        parser.error("--vocab-size and --hidden-size must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if any(top_k < 2 for top_k in args.top_k_values):
        parser.error("--top-k-values must each be at least 2")

    if args.histogram is not None:
        if args.support_sizes or args.support_weights:
            parser.error("--histogram already states the distribution; drop --support-sizes/--support-weights")
        sizes, weights = _read_histogram(args.histogram)
        label = f"MEASURED({args.histogram})"
    elif args.support_sizes or args.support_weights:
        if not args.support_sizes or not args.support_weights:
            parser.error("--support-sizes and --support-weights go together")
        if len(args.support_sizes) != len(args.support_weights):
            parser.error("--support-sizes and --support-weights must be the same length")
        total = sum(args.support_weights)
        if total <= 0 or any(weight < 0 for weight in args.support_weights):
            parser.error("--support-weights must be non-negative and sum to more than zero")
        sizes = tuple(args.support_sizes)
        weights = tuple(weight / total for weight in args.support_weights)
        label = "ASSUMED(--support-sizes)"
    else:
        sizes, weights = ASSUMED_SUPPORT_SIZES, ASSUMED_SUPPORT_WEIGHTS
        label = "ASSUMED(placeholder)"
    if any(size < 0 for size in sizes):
        parser.error("support sizes must be non-negative")

    print(
        f"device={device} layout={args.layout} source={args.source} batch={args.batch_size} "
        f"sequence_length={args.sequence_length} vocab={args.vocab_size} hidden={args.hidden_size} "
        f"entropy={args.compute_entropy} baseline=PackedTensor[rows,top_k] distribution={label} "
        f"sizes={sizes} weights={tuple(round(weight, 4) for weight in weights)}"
    )
    if label == "ASSUMED(placeholder)":
        print("NOTE: the support-size distribution is a placeholder. Feed --histogram before quoting a speedup.")
    for top_k in args.top_k_values:
        for chunk_size in args.chunk_sizes:
            _run_case(args, top_k, chunk_size, device, sizes, weights, label)


if __name__ == "__main__":
    main()
