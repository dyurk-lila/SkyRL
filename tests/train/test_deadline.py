"""CPU unit tests for skyrl.train.utils.deadline."""

import asyncio
import pickle
import time

import pytest
import ray

from skyrl.train.utils import Timer, deadline
from skyrl.train.utils.async_utils import cleanup_preserving_primary
from skyrl.train.utils.deadline import StepTimeoutError, WeightSyncTimeoutError


@pytest.mark.asyncio
async def test_remaining_is_none_outside_a_deadline():
    assert deadline.remaining() is None
    async with deadline.step_deadline(1, None):
        assert deadline.remaining() is None
    async with deadline.step_deadline(1, 60.0):
        assert 0 < deadline.remaining() <= 60.0
    assert deadline.remaining() is None


@pytest.mark.asyncio
async def test_nested_deadline_cannot_extend_parent(monkeypatch):
    def fake_get(refs, timeout=None):
        raise ray.exceptions.GetTimeoutError()

    monkeypatch.setattr(deadline.ray, "get", fake_get)

    async with deadline.step_deadline(3, 10.0):
        # A longer nested budget leaves the parent binding, so the parent's error class is raised.
        async with deadline.step_deadline(3, 1000.0, WeightSyncTimeoutError):
            assert deadline.remaining() <= 10.0
            with pytest.raises(StepTimeoutError) as exc_info:
                deadline.ray_get([], "broadcast_to_inference_engines")
            assert type(exc_info.value) is StepTimeoutError
            assert exc_info.value.budget_s == 10.0

        # A shorter nested budget binds, with its own error class.
        async with deadline.step_deadline(3, 1.0, WeightSyncTimeoutError):
            assert deadline.remaining() <= 1.0
            with pytest.raises(WeightSyncTimeoutError) as exc_info:
                deadline.ray_get([], "broadcast_to_inference_engines")
            assert exc_info.value.budget_s == 1.0
        assert deadline.remaining() > 1.0


@pytest.mark.asyncio
async def test_ray_get_passes_timeout_and_names_the_stage(monkeypatch):
    timeouts = []

    def fake_get(refs, timeout=None):
        timeouts.append(timeout)
        if timeout is not None:
            raise ray.exceptions.GetTimeoutError()
        return "ok"

    monkeypatch.setattr(deadline.ray, "get", fake_get)

    assert deadline.ray_get(["ref"], "forward") == "ok"
    assert timeouts == [None]

    async with deadline.step_deadline(7, 30.0):
        with Timer("step"):
            with Timer("train_critic_and_policy"):
                with pytest.raises(StepTimeoutError) as exc_info:
                    deadline.ray_get(["ref"], "forward_backward")
    assert 0 < timeouts[-1] <= 30.0
    err = exc_info.value
    assert (err.global_step, err.stage, err.operation, err.budget_s) == (
        7,
        "train_critic_and_policy",
        "forward_backward",
        30.0,
    )
    assert "step 7 exceeded its 30s budget" in str(err)


@pytest.mark.asyncio
async def test_ray_get_times_out_a_real_remote_task():
    @ray.remote
    def slow():
        time.sleep(30)

    ref = slow.remote()
    try:
        async with deadline.step_deadline(1, 0.2):
            with Timer("fwd_logprobs_values_reward"):
                with pytest.raises(StepTimeoutError) as exc_info:
                    deadline.ray_get(ref, "forward")
    finally:
        ray.cancel(ref, force=True)
    assert exc_info.value.stage == "fwd_logprobs_values_reward"
    assert 0.15 <= exc_info.value.elapsed_s < 10


@pytest.mark.parametrize("error_cls", [StepTimeoutError, WeightSyncTimeoutError])
def test_timeout_errors_pickle_and_are_not_timeout_errors(error_cls):
    err = error_cls(4, "sync_weights", "broadcast_to_inference_engines", 60.0, 61.5)
    restored = pickle.loads(pickle.dumps(err))
    assert type(restored) is error_cls
    assert (restored.global_step, restored.stage, restored.operation, restored.budget_s, restored.elapsed_s) == (
        4,
        "sync_weights",
        "broadcast_to_inference_engines",
        60.0,
        61.5,
    )
    assert str(restored) == str(err)
    assert not isinstance(err, (TimeoutError, OSError))


# --------------------------------------------------------------------------------------
# Awaited sections
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expiry_cancels_the_await_and_names_stage_and_operation():
    with pytest.raises(StepTimeoutError) as exc_info:
        async with deadline.step_deadline(2, 0.05), Timer("step"):
            with Timer("generate"), deadline.operation("generate"):
                await asyncio.sleep(30)
    err = exc_info.value
    assert (err.global_step, err.stage, err.operation, err.budget_s) == (2, "generate", "generate", 0.05)
    assert err.elapsed_s < 10


@pytest.mark.asyncio
async def test_expiry_outside_any_operation_reports_await():
    with pytest.raises(StepTimeoutError) as exc_info:
        async with deadline.step_deadline(2, 0.05), Timer("wait_for_generation_buffer"):
            await asyncio.Event().wait()
    assert (exc_info.value.stage, exc_info.value.operation) == ("wait_for_generation_buffer", "await")


@pytest.mark.asyncio
async def test_nested_weight_sync_deadline_raises_its_own_error():
    with pytest.raises(WeightSyncTimeoutError) as exc_info:
        async with deadline.step_deadline(5, 60.0), Timer("step"):
            async with deadline.step_deadline(5, 0.05, WeightSyncTimeoutError), Timer("sync_weights"):
                with deadline.operation("/pause"):
                    await asyncio.sleep(30)
    err = exc_info.value
    assert (err.stage, err.operation, err.budget_s) == ("sync_weights", "/pause", 0.05)


@pytest.mark.asyncio
async def test_unrelated_timeout_error_passes_through():
    err = TimeoutError("from the body")
    with pytest.raises(TimeoutError) as exc_info:
        async with deadline.step_deadline(1, 60.0):
            raise err
    assert exc_info.value is err

    with pytest.raises(TimeoutError) as exc_info:
        async with deadline.step_deadline(1, 60.0):
            async with asyncio.timeout(0.01):
                await asyncio.sleep(30)
    assert not isinstance(exc_info.value, StepTimeoutError)


@pytest.mark.asyncio
async def test_cleanup_runs_after_expiry_and_primary_is_the_step_timeout():
    class FakeClient:
        def __init__(self):
            self.calls = []

        async def pause_generation(self):
            self.calls.append("pause")

        async def resume_generation(self):
            # Awaiting here would be cancelled if the deadline re-delivered its cancellation.
            await asyncio.sleep(0.01)
            self.calls.append("resume")

    client = FakeClient()
    with pytest.raises(StepTimeoutError) as exc_info:
        async with deadline.step_deadline(9, 0.05), Timer("sync_weights"):
            await client.pause_generation()
            async with cleanup_preserving_primary(client.resume_generation, "resume_generation"):
                await asyncio.sleep(30)
    assert client.calls == ["pause", "resume"]
    assert exc_info.value.stage == "sync_weights"
    assert not getattr(exc_info.value, "__notes__", None)
