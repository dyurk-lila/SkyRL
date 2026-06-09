"""CPU test: the SFTTrainer peak-memory consumer max-reduces across ranks.

Two layers, no GPUs / no Ray cluster:

  * Unit: exercises ``SFTTrainer._peak_memory_log`` in isolation by mocking the
    policy actor group's ``async_run_ray_method`` and patching ``ray.get`` to
    return fabricated per-rank dicts.
  * Integration: drives the REAL ``train()`` and ``_train_dummy()`` loops with a
    mocked dispatch (same approach as ``tests/train/test_sft_callbacks.py``) and
    asserts the ``log_peak_memory`` gating is honored at BOTH per-step log sites
    — i.e. with the flag off no ``memory/*`` keys reach ``tracker.log`` and the
    fan-out RPC is never issued (truly byte-identical to before the feature).

Worker-side GPU correctness of ``get_peak_cuda_memory`` is covered separately by
the CUDA-guarded tests in
``tests/backends/skyrl_train/workers/test_get_peak_cuda_memory.py``; the consumer
here is fully CPU-mockable.

uv run --isolated --extra dev --extra fsdp pytest tests/train/test_sft_peak_memory.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

from skyrl.train.config.sft_config import (
    SFTConfig,
    SFTPlacementConfig,
    build_skyrl_config_for_sft,
)
from skyrl.train.sft_trainer import SFTTrainer

_GB = 1024**3


def _make_trainer(log_peak_memory: bool = True) -> SFTTrainer:
    """A minimally-initialized trainer that never spins up workers.

    We bypass ``setup()`` and only populate the attributes ``_peak_memory_log``
    touches (``sft_cfg`` and ``dispatch``).
    """
    trainer = object.__new__(SFTTrainer)
    cfg = SFTConfig()
    cfg.strategy = "fsdp"
    cfg.placement = SFTPlacementConfig(num_nodes=1, num_gpus_per_node=2)
    cfg.log_peak_memory = log_peak_memory
    trainer.sft_cfg = cfg
    trainer.dispatch = MagicMock()
    return trainer


def _per_rank(*specs: tuple[int, int]) -> list[dict]:
    """Build fake per-rank dicts from (max_allocated, max_reserved) byte pairs."""
    return [
        {"max_allocated": alloc, "max_reserved": reserved, "total": 80 * _GB, "rank": i}
        for i, (alloc, reserved) in enumerate(specs)
    ]


def test_peak_memory_log_max_reduces_across_ranks():
    """The consumer takes the max over ranks for each counter and reports GB."""
    trainer = _make_trainer()
    # rank 0 has the larger allocated; rank 1 has the larger reserved.
    fake = _per_rank((3 * _GB, 5 * _GB), (4 * _GB, 4 * _GB))

    sentinel = object()
    trainer.dispatch.policy_actor_group.async_run_ray_method.return_value = sentinel
    with patch("skyrl.train.sft_trainer.ray.get", return_value=fake) as mock_get:
        out = trainer._peak_memory_log()

    # Fans out the right pass_through method on the policy group, then ray.gets it.
    trainer.dispatch.policy_actor_group.async_run_ray_method.assert_called_once_with(
        "pass_through", "get_peak_cuda_memory"
    )
    mock_get.assert_called_once_with(sentinel)

    assert out == {
        "memory/peak_allocated_gb": 4.0,  # max(3, 4)
        "memory/peak_reserved_gb": 5.0,  # max(5, 4)
        # reserved fraction uses the capacity of the peak-reserved rank (rank 0,
        # 80 GB card): max_reserved 5 GB / 80 GB.
        "memory/peak_reserved_frac": 5.0 / 80.0,
    }


def test_peak_memory_log_does_not_pass_reset_false():
    """The consumer relies on the worker's default ``reset=True`` per-step window.

    It fans out ``get_peak_cuda_memory`` with no extra args, so each logged step
    measures a fresh window rather than a cumulative high-water mark. Guards
    against someone wiring through ``reset=False`` (which would make the metric
    monotonically non-decreasing and meaningless per step).
    """
    trainer = _make_trainer()
    fake = _per_rank((1 * _GB, 1 * _GB))
    with patch("skyrl.train.sft_trainer.ray.get", return_value=fake):
        trainer._peak_memory_log()

    _, call_args, call_kwargs = trainer.dispatch.policy_actor_group.async_run_ray_method.mock_calls[0]
    assert call_args == ("pass_through", "get_peak_cuda_memory")
    assert call_kwargs == {}


def test_peak_memory_log_single_rank():
    """A single-rank result is reported as-is (in GB)."""
    trainer = _make_trainer()
    fake = _per_rank((7 * _GB, 9 * _GB))
    with patch("skyrl.train.sft_trainer.ray.get", return_value=fake):
        out = trainer._peak_memory_log()
    assert out == {
        "memory/peak_allocated_gb": 7.0,
        "memory/peak_reserved_gb": 9.0,
        "memory/peak_reserved_frac": 9.0 / 80.0,
    }


def test_peak_memory_log_empty_results_returns_empty_dict():
    """No per-rank results -> empty dict (defensive; never blocks the log dict)."""
    trainer = _make_trainer()
    with patch("skyrl.train.sft_trainer.ray.get", return_value=[]):
        assert trainer._peak_memory_log() == {}


# ---------------------------------------------------------------------------
# Integration: drive the REAL log-dict assembly at both per-step sites so a
# dropped/inverted ``if self.sft_cfg.log_peak_memory:`` guard is caught.
# ---------------------------------------------------------------------------

_DUMMY_PEAK = _per_rank((6 * _GB, 8 * _GB))


def _logged_memory_keys(tracker: MagicMock) -> set[str]:
    """Collect every ``memory/*`` key passed to any ``tracker.log`` call."""
    keys: set[str] = set()
    for call in tracker.log.call_args_list:
        log_dict = call.args[0] if call.args else call.kwargs.get("log_dict", {})
        keys.update(k for k in log_dict if k.startswith("memory/"))
    return keys


def _build_integration_trainer(log_peak_memory: bool) -> SFTTrainer:
    """A real SFTTrainer wired with a mocked dispatch (mirrors test_sft_callbacks).

    The dispatch's policy fan-out returns ``_DUMMY_PEAK`` via a patched
    ``ray.get`` (installed by the caller). Everything else needed by the two
    train loops is mocked: no model, no workers, no GPUs.
    """
    cfg = SFTConfig()
    cfg.strategy = "fsdp"
    cfg.model.path = "unused"
    # Single rank for the loop wiring (mirrors test_sft_callbacks). The per-rank
    # fan-out result (_DUMMY_PEAK) is mocked independently of placement, so this
    # still exercises the multi-rank max-reduce in the consumer.
    cfg.placement = SFTPlacementConfig(num_nodes=1, num_gpus_per_node=1)
    cfg.dataset_name = "unused-monkeypatched"
    cfg.dataset_split = "train"
    cfg.eval_dataset_name = ""
    cfg.eval_before_train = False
    cfg.num_steps = 2
    cfg.num_epochs = None
    cfg.batch_size = 2
    cfg.micro_train_batch_size_per_gpu = 1
    cfg.max_length = 8
    cfg.remove_microbatch_padding = False
    cfg.logger = "console"
    cfg.ckpt_path = ""
    cfg.ckpt_interval = 0
    cfg.hf_save_interval = 0
    # Keep the test focused on the peak-memory gating; the background GPU monitor
    # is unrelated and would only emit a "Ray not initialized" warning here.
    cfg.enable_ray_gpu_monitor = False
    # Default to the real data-driven train() loop. The _train_dummy test flips
    # this on (train() early-returns into _train_dummy when it is True).
    cfg.dummy_run_full_ctx = False
    cfg.dummy_run_max_steps = 2
    cfg.log_peak_memory = log_peak_memory

    skyrl_cfg = build_skyrl_config_for_sft(cfg)
    trainer = SFTTrainer(cfg, skyrl_cfg=skyrl_cfg)

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.vocab_size = 128
    trainer.tokenizer = tokenizer
    trainer.collator = trainer._build_collator(tokenizer)
    trainer.tracker = MagicMock()

    step_output = MagicMock()
    step_output.metrics = {"loss": 0.42, "final_loss": 0.42}
    dispatch_mock = MagicMock()
    dispatch_mock.forward_backward = MagicMock(return_value=step_output)
    dispatch_mock.optim_step = MagicMock(return_value=1.0)
    dispatch_mock.dp_size = MagicMock(return_value=1)
    # The peak-memory fan-out goes through the policy actor group; ray.get is
    # patched per-test to return _DUMMY_PEAK regardless of the (mocked) handle.
    trainer.dispatch = dispatch_mock
    return trainer


def _dummy_tokenized() -> list[dict]:
    example = {
        "input_ids": [1, 2, 3, 4, 5, 6, 7, 8],
        "attention_mask": [1] * 8,
        "num_actions": 4,
        "loss_mask": [1, 1, 1, 1],
    }
    return [example, example]


@pytest.mark.parametrize("log_peak_memory", [True, False])
def test_train_loop_gates_peak_memory_at_real_log_site(monkeypatch, log_peak_memory):
    """Drive the real ``train()`` loop; the memory/* keys appear iff the flag is on.

    This exercises the actual ``if self.sft_cfg.log_peak_memory:`` guard at the
    ``train()`` log site (not a re-implemented copy), so a dropped or inverted
    guard there would fail this test.
    """
    trainer = _build_integration_trainer(log_peak_memory)
    monkeypatch.setattr(trainer, "_load_and_tokenize", lambda *_a, **_k: _dummy_tokenized())
    monkeypatch.setattr(trainer, "load_checkpoint", lambda: 0)

    with patch("skyrl.train.sft_trainer.ray.get", return_value=_DUMMY_PEAK) as mock_get:
        trainer.train()

    logged = _logged_memory_keys(trainer.tracker)
    fan_out = trainer.dispatch.policy_actor_group.async_run_ray_method
    if log_peak_memory:
        assert logged == {
            "memory/peak_allocated_gb",
            "memory/peak_reserved_gb",
            "memory/peak_reserved_frac",
        }
        assert fan_out.call_args_list, "fan-out RPC should be issued when enabled"
        for call in fan_out.call_args_list:
            assert call.args == ("pass_through", "get_peak_cuda_memory")
        assert mock_get.call_count >= 1
    else:
        assert logged == set(), "no memory/* keys must reach tracker.log when disabled"
        fan_out.assert_not_called()
        mock_get.assert_not_called()


@pytest.mark.parametrize("log_peak_memory", [True, False])
def test_dummy_loop_gates_peak_memory_at_real_log_site(log_peak_memory):
    """Drive the real ``_train_dummy()`` loop; covers the SECOND gated log site.

    The two log-dict build sites are independently gated; this covers the
    benchmarking (``_train_dummy``) path while the test above covers ``train()``.
    """
    trainer = _build_integration_trainer(log_peak_memory)

    with patch("skyrl.train.sft_trainer.ray.get", return_value=_DUMMY_PEAK) as mock_get:
        trainer._train_dummy()

    logged = _logged_memory_keys(trainer.tracker)
    fan_out = trainer.dispatch.policy_actor_group.async_run_ray_method
    if log_peak_memory:
        assert logged == {
            "memory/peak_allocated_gb",
            "memory/peak_reserved_gb",
            "memory/peak_reserved_frac",
        }
        assert mock_get.call_count >= 1
    else:
        assert logged == set(), "no memory/* keys must reach tracker.log when disabled"
        fan_out.assert_not_called()
        mock_get.assert_not_called()
