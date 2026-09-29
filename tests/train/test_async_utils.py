"""CPU unit tests for skyrl.train.utils.async_utils."""

import asyncio

import pytest

from skyrl.train.utils.async_utils import BackgroundFailure, cleanup_preserving_primary

# --------------------------------------------------------------------------------------
# BackgroundFailure
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guard_returns_value():
    failure = BackgroundFailure()
    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait("x")
    assert await failure.guard(queue.get()) == "x"
    assert not failure.failed


@pytest.mark.asyncio
async def test_guard_raises_failure_recorded_while_blocked():
    failure = BackgroundFailure()
    queue: asyncio.Queue = asyncio.Queue()
    err = ValueError("worker died")

    async def fail_later():
        await asyncio.sleep(0.01)
        failure.record(err, "worker")

    task = asyncio.create_task(fail_later())
    with pytest.raises(ValueError) as exc_info:
        await asyncio.wait_for(failure.guard(queue.get()), timeout=5)
    await task
    assert exc_info.value is err
    # The losing get() was cancelled rather than left waiting on the queue.
    await asyncio.sleep(0)
    assert not queue._getters or all(g.cancelled() for g in queue._getters)


@pytest.mark.asyncio
async def test_record_keeps_first_exception_and_adds_note():
    failure = BackgroundFailure()
    first, second = RuntimeError("first"), RuntimeError("second")
    failure.record(first, "generation worker")
    failure.record(second, "generation worker")
    assert failure.failed
    with pytest.raises(RuntimeError) as exc_info:
        failure.raise_if_failed()
    assert exc_info.value is first
    assert first.__notes__ == ["raised in background generation worker"]
    assert not hasattr(second, "__notes__")


def test_raise_if_failed_noop_without_failure():
    BackgroundFailure().raise_if_failed()


@pytest.mark.asyncio
async def test_guard_prefers_result_over_simultaneous_failure():
    """An item popped in the same iteration the failure lands must be returned, not dropped."""
    failure = BackgroundFailure()
    queue: asyncio.Queue = asyncio.Queue()

    def put_and_fail():
        queue.put_nowait("item")
        failure.record(RuntimeError("boom"), "worker")

    asyncio.get_running_loop().call_soon(put_and_fail)
    assert await failure.guard(queue.get()) == "item"
    assert queue.empty()


@pytest.mark.asyncio
async def test_guard_raises_immediately_once_failed_without_consuming():
    """After a failure, guard raises instead of draining items that are still buffered."""
    failure = BackgroundFailure()
    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait("item")
    failure.record(RuntimeError("boom"), "worker")
    with pytest.raises(RuntimeError, match="boom"):
        await failure.guard(queue.get())
    assert queue.qsize() == 1


@pytest.mark.asyncio
async def test_guard_propagates_awaitable_exception():
    failure = BackgroundFailure()

    async def bad():
        raise KeyError("k")

    with pytest.raises(KeyError):
        await failure.guard(bad())


# --------------------------------------------------------------------------------------
# cleanup_preserving_primary
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_mask_primary():
    async def failing_cleanup():
        raise ConnectionError("engine unreachable")

    with pytest.raises(ValueError, match="broadcast failed") as exc_info:
        async with cleanup_preserving_primary(failing_cleanup, "resume_generation"):
            raise ValueError("broadcast failed")
    (note,) = exc_info.value.__notes__
    assert note.startswith("resume_generation also failed during cleanup: ConnectionError(")


@pytest.mark.asyncio
async def test_cleanup_error_propagates_on_success():
    async def failing_cleanup():
        raise ConnectionError("engine unreachable")

    with pytest.raises(ConnectionError):
        async with cleanup_preserving_primary(failing_cleanup, "resume_generation"):
            pass


@pytest.mark.asyncio
async def test_cleanup_runs_on_success_and_on_cancel():
    calls = []

    async def cleanup():
        calls.append("cleanup")

    async with cleanup_preserving_primary(cleanup, "resume_generation"):
        pass
    assert calls == ["cleanup"]

    started = asyncio.Event()

    async def body():
        async with cleanup_preserving_primary(cleanup, "resume_generation"):
            started.set()
            await asyncio.sleep(3600)

    task = asyncio.create_task(body())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == ["cleanup", "cleanup"]
