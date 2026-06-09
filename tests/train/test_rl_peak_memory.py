"""CPU test: the RayPPOTrainer peak-memory consumer is gated and reduces correctly.

Mirrors ``tests/train/test_sft_peak_memory.py`` for the RL pathway. No GPUs /
no Ray cluster: the worker-group fan-out is mocked and ``ray.get`` (inside the
shared :func:`peak_memory_metrics` helper) is patched to return fabricated
per-rank dicts.

Drives the REAL ``RayPPOTrainer.train()`` loop (heavy mocking mirrors
``tests/train/test_rl_callbacks.py``) and asserts the per-step log assembly:

  * with ``log_peak_memory=False`` no ``memory/*`` keys reach ``tracker.log`` and
    the fan-out RPC is never issued (byte-identical to before the feature);
  * with ``log_peak_memory=True`` the per-group + cluster-headline keys appear,
    only the ACTIVE worker groups are fanned out (a None ref/critic is skipped),
    and the fan-out goes through ``async_run_ray_method('pass_through',
    'get_peak_cuda_memory')`` on each active group.

Pure reduce correctness (max over ranks, multi-group max headline, GB
conversion, None-group skipping, {} when empty) is covered by
``tests/train/test_trainer_utils.py::test_peak_memory_metrics_*``.

uv run --isolated --extra dev pytest tests/train/test_rl_peak_memory.py -v
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.fully_async_trainer import (
    FullyAsyncRayPPOTrainer,
    GeneratedOutputGroup,
)
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils.trainer_utils import ResumeMode
from tests.train.util import example_dummy_config

_GB = 1024**3


class _DummyDataset:
    """Single-batch dataset: one iteration -> one training step."""

    def __init__(self, size: int = 2):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return ([{"role": "user", "content": f"q{idx}"}], None)

    def collate_fn(self, batch):
        return batch


def _per_rank(*specs):
    """Build per-rank dicts from (max_allocated, max_reserved) byte pairs."""
    return [
        {"max_allocated": alloc, "max_reserved": reserved, "total": 80 * _GB, "rank": i}
        for i, (alloc, reserved) in enumerate(specs)
    ]


_DUMMY_PEAK = _per_rank((6 * _GB, 8 * _GB))

# The full set of memory/* keys an RL step emits when only the policy group is
# active (critic + ref are None here): per-group policy keys + the cluster-wide
# headline keys from the shared helper.
_EXPECTED_MEMORY_KEYS = {
    "memory/policy/peak_allocated_gb",
    "memory/policy/peak_reserved_gb",
    "memory/policy/peak_reserved_frac",
    "memory/peak_allocated_gb",
    "memory/peak_reserved_gb",
    "memory/peak_reserved_frac",
}


def _stub_training_input() -> TrainingInputBatch:
    """Minimal TrainingInputBatch that survives the keys the loop pops post-advantages."""
    batch = TrainingInputBatch(
        {
            "sequences": torch.zeros((1, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "loss_mask": torch.ones((1, 4), dtype=torch.long),
            "response_mask": torch.ones((1, 4), dtype=torch.long),
            "rewards": torch.zeros((1, 4)),
        }
    )
    batch.metadata = {
        "uids": ["uid-0"],
        "response_length": 4,
        "avg_response_length": 4.0,
    }
    return batch


def _build_test_cfg(log_peak_memory: bool):
    cfg = example_dummy_config()
    cfg.trainer.epochs = 1
    cfg.trainer.eval_interval = -1  # no eval; keep the loop focused on the log site
    cfg.trainer.eval_before_train = False
    cfg.trainer.ckpt_interval = 0
    cfg.trainer.hf_save_interval = 0
    cfg.trainer.ckpt_path = ""
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.dump_data_batch = False
    cfg.trainer.algorithm.use_kl_in_reward = False
    cfg.trainer.update_ref_every_epoch = False
    cfg.trainer.algorithm.dynamic_sampling.type = None
    cfg.generator.step_wise_trajectories = False
    cfg.generator.inference_engine.enable_ray_prometheus_stats = False
    # The background GPU monitor is unrelated and would only emit a "Ray not
    # initialized" warning here.
    cfg.trainer.enable_ray_gpu_monitor = False
    cfg.trainer.log_peak_memory = log_peak_memory
    return cfg


def _build_trainer(monkeypatch, log_peak_memory: bool):
    """A real RayPPOTrainer wired with mocked workers (mirrors test_rl_callbacks)."""
    cfg = _build_test_cfg(log_peak_memory)

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2

    trainer = RayPPOTrainer(
        cfg=cfg,
        tracker=MagicMock(),
        tokenizer=tokenizer,
        train_dataset=_DummyDataset(size=2),
        eval_dataset=None,
        inference_engine_client=None,
        generator=MagicMock(),
    )

    # Only the policy group is active; critic + ref stay None so the test also
    # asserts the helper skips None groups (no fan-out on them).
    trainer.policy_model = MagicMock()
    assert trainer.critic_model is None
    assert trainer.ref_model is None

    dispatch_mock = MagicMock()
    dispatch_mock.save_weights_for_sampler = AsyncMock(return_value=None)
    dispatch_mock.get_lcm_dp_size = MagicMock(return_value=1)
    trainer.dispatch = dispatch_mock

    monkeypatch.setattr(trainer, "init_weight_sync_state", lambda: None)
    monkeypatch.setattr(
        trainer,
        "generate",
        AsyncMock(return_value={"rollout_metrics": None, "response_ids": [[1]], "rewards": [0.0]}),
    )
    monkeypatch.setattr(trainer, "postprocess_generator_output", lambda gen_out, uids: (gen_out, uids))
    monkeypatch.setattr(trainer, "convert_to_training_input", lambda *_args, **_kw: _stub_training_input())
    monkeypatch.setattr(trainer, "fwd_logprobs_values_reward", lambda batch: batch)
    monkeypatch.setattr(trainer, "compute_advantages_and_returns", lambda batch: batch)
    monkeypatch.setattr(trainer, "train_critic_and_policy", lambda batch: {"policy_loss": 0.42})
    monkeypatch.setattr(
        "skyrl.train.trainer.prepare_generator_input",
        lambda *_args, **_kw: ({"prompts": [[{"role": "user", "content": "q"}]]}, ["uid-0"]),
    )
    monkeypatch.setattr(trainer, "load_checkpoints", lambda: (0, ""))
    return trainer


def _logged_memory_keys(tracker: MagicMock) -> set[str]:
    """Collect every ``memory/*`` key passed to any ``tracker.log`` call."""
    keys: set[str] = set()
    for call in tracker.log.call_args_list:
        log_dict = call.args[0] if call.args else call.kwargs.get("log_dict", {})
        keys.update(k for k in log_dict if k.startswith("memory/"))
    return keys


@pytest.mark.parametrize("log_peak_memory", [True, False])
def test_rl_train_loop_gates_peak_memory_at_real_log_site(monkeypatch, log_peak_memory):
    """Drive the real RL ``train()`` loop; memory/* keys appear iff the flag is on.

    Exercises the actual ``if self.cfg.trainer.log_peak_memory:`` guard at the
    RL per-step log site (not a re-implemented copy), so a dropped or inverted
    guard there would fail this test.
    """
    trainer = _build_trainer(monkeypatch, log_peak_memory)

    with patch("skyrl.train.utils.trainer_utils.ray.get", return_value=_DUMMY_PEAK) as mock_get:
        asyncio.run(trainer.train())

    logged = _logged_memory_keys(trainer.tracker)
    policy_fan_out = trainer.policy_model.async_run_ray_method
    if log_peak_memory:
        assert logged == _EXPECTED_MEMORY_KEYS
        assert policy_fan_out.call_args_list, "policy fan-out should be issued when enabled"
        for call in policy_fan_out.call_args_list:
            assert call.args == ("pass_through", "get_peak_cuda_memory")
        assert mock_get.call_count >= 1
    else:
        assert logged == set(), "no memory/* keys must reach tracker.log when disabled"
        policy_fan_out.assert_not_called()
        mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# Fully-async RL parity: FullyAsyncRayPPOTrainer.train() FULLY OVERRIDES the
# base RayPPOTrainer.train() with its own per-step log assembly and never calls
# into the base gate, so the peak-memory block has to be wired into the override
# too. This drives the override's per-step log site to prove that gap is closed
# (and stays closed): a dropped block here would regress to "fully-async RL
# silently logs nothing" even with the flag on.
# ---------------------------------------------------------------------------


def _build_fully_async_trainer(monkeypatch, log_peak_memory: bool) -> FullyAsyncRayPPOTrainer:
    """A FullyAsyncRayPPOTrainer wired to run exactly one step of train().

    ``__init__`` is bypassed (it requires a live placement group / inference
    engine); only the attributes the single-step train() loop and its log site
    touch are populated, and the heavy collaborators (generation worker loop,
    staleness manager, async dataloader, training) are mocked so one step runs.
    """
    cfg = _build_test_cfg(log_peak_memory)
    cfg.trainer.update_ref_every_epoch = False

    trainer = object.__new__(FullyAsyncRayPPOTrainer)
    trainer.cfg = cfg
    trainer.colocate_all = False  # fully async asserts colocate_all is False
    trainer.tracker = MagicMock()
    trainer.tokenizer = MagicMock()
    trainer.all_metrics = {}
    trainer.all_timings = {}
    trainer.global_step = 0
    trainer.resume_mode = ResumeMode.NONE
    trainer._vllm_metrics_scraper = None

    # Exactly one step, one epoch, mini-batch of one group.
    trainer.mini_batch_size = 1
    trainer.max_staleness_steps = 0
    trainer.num_parallel_generation_workers = 1
    trainer.num_steps_per_epoch = 1
    trainer.total_training_steps = 1

    # Only the policy group is active; critic + ref stay None so the helper's
    # None-skip is exercised here too.
    trainer.policy_model = MagicMock()
    trainer.critic_model = None
    trainer.ref_model = None

    dispatch_mock = MagicMock()
    dispatch_mock.save_weights_for_sampler = AsyncMock(return_value=None)
    trainer.dispatch = dispatch_mock

    # One ready-made generated group to pop from the buffer.
    group = GeneratedOutputGroup(
        generator_output={"response_ids": [[1]], "rewards": [0.0], "rollout_metrics": None},
        uid="uid-0",
        global_step_when_scheduled=1,
    )

    async def _fill_buffer_then_idle(buffer):
        await buffer.put(group)
        # Idle forever; the epoch teardown cancels this task.
        await asyncio.Event().wait()

    monkeypatch.setattr(trainer, "init_weight_sync_state", lambda: None)
    monkeypatch.setattr(trainer, "_run_generate_for_a_group_loop", _fill_buffer_then_idle)
    monkeypatch.setattr(
        trainer,
        "convert_generation_group_mini_batch_to_training_input",
        lambda groups: _stub_training_input(),
    )
    monkeypatch.setattr(trainer, "_run_training", AsyncMock(return_value={"policy_loss": 0.42}))

    # Async dataloader + staleness manager: just enough to satisfy the loop and
    # the end-of-epoch consumed-UID assertion (1 consumed == mini_batch_size * 1).
    async_dataloader = MagicMock()
    async_dataloader.mark_consumed_uids = AsyncMock(return_value=None)
    async_dataloader.get_consumed_uids_list = MagicMock(return_value=["uid-0"])
    async_dataloader.reset_at_epoch_end = AsyncMock(return_value=None)
    trainer.async_train_dataloader = async_dataloader

    staleness_manager = MagicMock()
    staleness_manager.notify_capacity_change = AsyncMock(return_value=None)
    staleness_manager.validate_state_at_epoch_end = AsyncMock(return_value=None)
    trainer._staleness_manager = staleness_manager

    return trainer


@pytest.mark.parametrize("log_peak_memory", [True, False])
def test_fully_async_rl_train_loop_gates_peak_memory_at_real_log_site(monkeypatch, log_peak_memory):
    """Drive FullyAsyncRayPPOTrainer.train(); memory/* keys appear iff the flag is on.

    The fully-async override never reaches the base trainer's gate, so this
    asserts the block was re-wired into the override's own per-step log assembly.
    """
    trainer = _build_fully_async_trainer(monkeypatch, log_peak_memory)

    with patch("skyrl.train.utils.trainer_utils.ray.get", return_value=_DUMMY_PEAK) as mock_get:
        asyncio.run(trainer.train())

    logged = _logged_memory_keys(trainer.tracker)
    policy_fan_out = trainer.policy_model.async_run_ray_method
    if log_peak_memory:
        assert logged == _EXPECTED_MEMORY_KEYS
        assert policy_fan_out.call_args_list, "policy fan-out should be issued when enabled"
        for call in policy_fan_out.call_args_list:
            assert call.args == ("pass_through", "get_peak_cuda_memory")
        assert mock_get.call_count >= 1
    else:
        assert logged == set(), "no memory/* keys must reach tracker.log when disabled"
        policy_fan_out.assert_not_called()
        mock_get.assert_not_called()
