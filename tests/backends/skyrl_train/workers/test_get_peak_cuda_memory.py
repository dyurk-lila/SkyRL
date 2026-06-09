"""
CPU-collectable tests for ``Worker.get_peak_cuda_memory``.

The signature/shape assertions run on any host. The behavioral assertions
(values, types, and reset semantics) require a real CUDA device and are
skipped otherwise, since SkyRL's CPU CI lane has no GPU.

uv run --isolated --extra dev pytest tests/backends/skyrl_train/workers/test_get_peak_cuda_memory.py
"""

import inspect

import pytest
import torch

from skyrl.backends.skyrl_train.workers.worker import Worker

EXPECTED_KEYS = {"max_allocated", "max_reserved", "total", "rank"}


def test_get_peak_cuda_memory_exists_and_signature():
    """The method exists on the base Worker with a ``reset`` arg defaulting to True."""
    assert hasattr(Worker, "get_peak_cuda_memory")
    sig = inspect.signature(Worker.get_peak_cuda_memory)
    # (self, reset)
    assert list(sig.parameters) == ["self", "reset"]
    assert sig.parameters["reset"].default is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")
def test_get_peak_cuda_memory_returns_expected_keys_and_types():
    """On a CUDA host the method reports the four peak-memory keys with int values."""
    # The method only touches torch.cuda / torch.distributed (not worker state),
    # so we can exercise it without running the (Ray/CUDA) Worker constructor.
    worker = object.__new__(Worker)

    result = Worker.get_peak_cuda_memory(worker, reset=False)
    assert set(result) == EXPECTED_KEYS
    assert all(isinstance(result[k], int) for k in EXPECTED_KEYS)
    assert result["max_allocated"] >= 0
    assert result["max_reserved"] >= 0
    assert result["total"] > 0
    # With no torch.distributed process group, rank defaults to 0.
    if not torch.distributed.is_initialized():
        assert result["rank"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")
def test_get_peak_cuda_memory_reset_clears_high_water_mark():
    """``reset=True`` should clear the peak so a subsequent window starts lower."""
    worker = object.__new__(Worker)

    # Drive up the high-water mark, then read with reset.
    big = torch.empty(1024 * 1024, device="cuda")  # ~4 MiB
    peak_before = Worker.get_peak_cuda_memory(worker, reset=True)["max_allocated"]
    del big
    torch.cuda.empty_cache()

    # After reset, the new window's peak should not exceed the prior peak.
    small = torch.empty(1024, device="cuda")
    peak_after = Worker.get_peak_cuda_memory(worker, reset=False)["max_allocated"]
    del small
    assert peak_after <= peak_before
