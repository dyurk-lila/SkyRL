"""Single-slot async double-buffer for deterministic per-step collation."""

import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable


class AsyncBatchCollator:
    """Run ``compute(step)`` in a one-worker, one-future buffer.

    The caller may only submit steps whose inputs will not change before
    consumption. ``get`` checks the expected step so stale batches fail loudly.
    """

    def __init__(self, compute: Callable[[int], Any], thread_name_prefix: str = "batch-collate"):
        self._compute = compute
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name_prefix)
        self._future: Future | None = None
        self._pending_step: int | None = None
        self._last_compute_seconds: float | None = None

    @property
    def last_compute_seconds(self) -> float | None:
        """Wall seconds the worker spent on the last completed ``compute``.

        This is the whole cost of producing a batch, most of which overlaps the
        caller's own work and so never shows up in the caller's wait. ``None``
        until the first compute finishes.
        """
        return self._last_compute_seconds

    def submit(self, step: int) -> None:
        """Schedule ``compute(step)``; call ``get`` before submitting again."""
        assert self._future is None, (
            f"collate-ahead slot already occupied (pending step {self._pending_step}); "
            f"call get() before submitting step {step}"
        )
        self._pending_step = step
        self._future = self._executor.submit(self._timed_compute, step)

    def _timed_compute(self, step: int) -> Any:
        # Recorded in a ``finally`` inside the worker, so it is committed before
        # the future resolves and is current by the time ``get`` returns.
        start = time.perf_counter()
        try:
            return self._compute(step)
        finally:
            self._last_compute_seconds = time.perf_counter() - start

    def has_pending(self) -> bool:
        return self._future is not None

    def pending_step(self) -> int | None:
        return self._pending_step

    def get(self, expected_step: int) -> Any:
        """Return the in-flight batch for ``expected_step``."""
        assert self._future is not None, "get() called with no in-flight batch"
        assert self._pending_step == expected_step, (
            f"collated-ahead step {self._pending_step} != expected step {expected_step}; "
            f"refusing to serve a mismatched batch"
        )
        future = self._future
        self._future = None
        self._pending_step = None
        # Propagates any exception raised inside the worker thread.
        return future.result()

    def clear(self) -> None:
        """Drain and discard the in-flight batch, propagating worker errors."""
        if self._future is not None:
            self._future.result()
        self._future = None
        self._pending_step = None

    def shutdown(self) -> None:
        """Drain any in-flight batch and join the worker thread."""
        self.clear()
        self._executor.shutdown(wait=True)
