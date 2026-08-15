"""Fused MoE router-replay CUDA kernel: build, availability probe and autograd wiring.

The extension is compiled with ``torch.utils.cpp_extension.load`` (JIT). Two things make
that safe enough to ship, and both matter:

* ``build_directory`` defaults to a **node-local** path, not the shared filesystem, for
  the same reason the training image moves ``TRITON_CACHE_DIR`` to ``/tmp``: a first-use
  compile from many ranks against one NFS cache serializes and can corrupt.
* ``warm_compile()`` is called once from the worker preflight so the compile happens
  before any forward, and every failure mode degrades to "kernel unavailable" with a
  logged reason instead of an exception inside the pipeline schedule.

Prebuilding into the training image removes the JIT entirely; see
``SKYRL_REPLAY_ROUTER_BUILD_DIR``.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional, Tuple

import torch
from loguru import logger

_SOURCE = Path(__file__).parent / "csrc" / "replay_router.cu"

BUILD_DIR_ENV_VAR = "SKYRL_REPLAY_ROUTER_BUILD_DIR"
VALIDATE_INDICES_ENV_VAR = "SKYRL_REPLAY_ROUTER_VALIDATE_INDICES"

# The kernel is warp-per-token: one lane per expert slot.
MAX_SUPPORTED_TOPK = 32
SUPPORTED_ROUTER_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

_lock = threading.Lock()
_extension = None
_unavailable_reason: Optional[str] = None


def _default_build_dir() -> Path:
    # Keyed on torch version so a torch bump cannot reuse an ABI-incompatible object file.
    return Path(f"/tmp/skyrl_replay_router_build/torch{torch.__version__}")


def _build() -> None:
    """Compile the extension, recording a reason string instead of raising."""
    global _extension, _unavailable_reason

    if not torch.cuda.is_available():
        _unavailable_reason = "CUDA is not available"
        return
    if not _SOURCE.exists():
        _unavailable_reason = f"CUDA source missing from the installed package: {_SOURCE}"
        return

    build_dir = Path(os.environ.get(BUILD_DIR_ENV_VAR) or _default_build_dir())
    major, minor = torch.cuda.get_device_capability()
    try:
        from torch.utils.cpp_extension import load

        build_dir.mkdir(parents=True, exist_ok=True)
        # -std=c++20 is required, not stylistic: under C++17 the training image's gcc 12.2
        # rejects PyTorch 2.11's own ATen/core/List_inl.h ("need 'typename' before
        # decltype(...)::difference_type"), which C++20 (P0634) made legal. -fpermissive
        # and -ccbin g++-12 do not help.
        _extension = load(
            name="skyrl_replay_router",
            sources=[str(_SOURCE)],
            extra_cflags=["-O3", "-std=c++20"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-std=c++20",
                f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}",
            ],
            build_directory=str(build_dir),
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - any build failure must degrade, not crash
        _unavailable_reason = f"extension build failed ({type(exc).__name__}): {exc}".replace("\n", " ")[:500]


def warm_compile() -> Optional[str]:
    """Build the extension now. Returns ``None`` on success or the failure reason.

    Call this from the worker preflight so the compile is not paid inside a pipeline
    schedule, and so an unusable toolchain is reported once at startup.
    """
    with _lock:
        if _extension is None and _unavailable_reason is None:
            _build()
        return _unavailable_reason


def is_available() -> bool:
    return warm_compile() is None


def unavailable_reason() -> Optional[str]:
    return warm_compile()


class _FusedReplayRoutingDense(torch.autograd.Function):
    """Megatron's ``(routing_probs, routing_map)`` contract, single launch each way."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, indices: torch.Tensor, scaling: float):
        routing_probs, routing_map, scores, denom = _extension.replay_fwd_dense(logits, indices, scaling)
        ctx.save_for_backward(scores, denom, indices)
        ctx.num_experts = logits.shape[1]
        ctx.scaling = scaling
        ctx.mark_non_differentiable(routing_map)
        return routing_probs, routing_map

    @staticmethod
    def backward(
        ctx,
        grad_routing_probs: Optional[torch.Tensor],
        grad_routing_map: Optional[torch.Tensor],
    ):
        if grad_routing_probs is None:
            return None, None, None
        scores, denom, indices = ctx.saved_tensors
        grad_logits = _extension.replay_bwd_dense(
            grad_routing_probs.contiguous(), scores, denom, indices, ctx.num_experts, ctx.scaling
        )
        return grad_logits, None, None


def fused_replay_routing_dense(
    logits: torch.Tensor,
    indices: torch.Tensor,
    scaling: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid router replay in Megatron's dense contract.

    Args:
        logits: ``[num_tokens, num_experts]`` fp32, fp16, or bf16 router logits.
        indices: ``[num_tokens, topk]`` replayed expert indices (any integer dtype).
        scaling: ``moe_router_topk_scaling_factor``. Megatron gates this on ``if
            scaling_factor:``, so ``None`` *and* ``0.0`` both mean "do not scale".

    Returns:
        ``(routing_probs[num_tokens, num_experts] logits.dtype, routing_map[...] bool)``.
    """
    if _extension is None:
        raise RuntimeError(f"fused router-replay kernel unavailable: {unavailable_reason()}")

    indices = indices.to(device=logits.device, dtype=torch.int32).contiguous()
    if os.environ.get(VALIDATE_INDICES_ENV_VAR):
        # Device-to-host sync; debugging only. The kernel has a device assertion too, but
        # this produces a clear ValueError without poisoning the CUDA context.
        num_experts = logits.shape[1]
        if indices.numel():
            min_index, max_index = int(indices.min()), int(indices.max())
            if min_index < 0 or max_index >= num_experts:
                raise ValueError(
                    f"replay indices out of range for num_experts={num_experts}: [{min_index}, {max_index}]"
                )
    # Falsy, not just None: topk_routing_with_score_function applies the factor under
    # `if scaling_factor:`, so a configured 0.0 leaves the probabilities alone there and
    # must not zero them here.
    resolved_scaling = 1.0 if not scaling else scaling
    return _FusedReplayRoutingDense.apply(logits.contiguous(), indices, float(resolved_scaling))


def log_availability() -> None:
    """Log once whether the kernel is usable. Safe to call on every rank."""
    reason = unavailable_reason()
    if reason is None:
        logger.info("fused MoE router-replay kernel ready")
    else:
        logger.warning(f"fused MoE router-replay kernel unavailable, using unfused replay: {reason}")
