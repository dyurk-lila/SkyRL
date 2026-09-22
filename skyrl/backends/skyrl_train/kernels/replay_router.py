"""Build and autograd wiring for the fused MoE router-replay CUDA kernel.

The extension compiles before the first forward in a node-local directory. Build failures
make the kernel unavailable so callers can fall back to unfused replay.
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
        # PyTorch 2.11 headers require C++20 with the training image's compiler.
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
    """Build the extension, returning ``None`` or the failure reason."""
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
        scaling: ``moe_router_topk_scaling_factor``; falsy values disable scaling.

    Returns:
        ``(routing_probs[num_tokens, num_experts] logits.dtype, routing_map[...] bool)``.
    """
    if _extension is None:
        raise RuntimeError(f"fused router-replay kernel unavailable: {unavailable_reason()}")

    indices = indices.to(device=logits.device, dtype=torch.int32).contiguous()
    if os.environ.get(VALIDATE_INDICES_ENV_VAR):
        # Debug-only host validation provides a clear error before the device assertion.
        num_experts = logits.shape[1]
        if indices.numel():
            min_index, max_index = int(indices.min()), int(indices.max())
            if min_index < 0 or max_index >= num_experts:
                raise ValueError(
                    f"replay indices out of range for num_experts={num_experts}: [{min_index}, {max_index}]"
                )
    # Match Megatron's falsy scaling-factor gate, including 0.0.
    resolved_scaling = 1.0 if not scaling else scaling
    return _FusedReplayRoutingDense.apply(logits.contiguous(), indices, float(resolved_scaling))


def log_availability() -> None:
    """Log once whether the kernel is usable. Safe to call on every rank."""
    reason = unavailable_reason()
    if reason is None:
        logger.info("fused MoE router-replay kernel ready")
    else:
        logger.warning(f"fused MoE router-replay kernel unavailable, using unfused replay: {reason}")
