"""CPU test: async data prefetch (double-buffering) is byte-identical to serial.

The SFT training loop optionally collates the NEXT step's batch on a background
thread while the current step's forward/backward runs on the GPU
(``SFTConfig.prefetch_data``). The prefetched batch MUST be byte-identical to
what the synchronous path would have produced for that same step — including
across epoch boundaries, where the data reshuffles in place.

Strategy: run the real ``SFTTrainer.train()`` loop twice over the same dummy
dataset/seed — once with prefetch OFF (the baseline) and once ON — capturing a
deep snapshot of every ``batch`` handed to ``train_step``. Then assert the two
batch sequences are tensor-equal step-for-step over a run that spans TWO epoch
boundaries. The examples have distinct token ids so a stale pre-shuffle batch
would differ from the baseline and fail the comparison.

Both collator paths are covered:
  * ``DefaultCollator`` (FSDP default left-pad path).
  * ``PackedDataCollator`` (Megatron FFD packing) — constructed directly,
    CPU-only, no Megatron runtime needed since the collate is pure numpy/torch.

Also unit-tests the ``BatchPrefetcher`` invariants (mismatch assertion,
exception propagation, single-slot guard).

Run:
  uv run --isolated --extra dev --extra fsdp pytest tests/train/test_sft_prefetch.py -v
"""

import copy
from typing import Optional
from unittest.mock import MagicMock

import pytest
import torch

from skyrl.train.config.sft_config import SFTConfig, SFTPlacementConfig
from skyrl.train.dataset.collators import DefaultCollator, PackedDataCollator
from skyrl.train.sft_trainer import SFTTrainer
from skyrl.train.utils.batch_prefetcher import BatchPrefetcher

# ---------------------------------------------------------------------------
# Helpers: dummy config, dataset, and a trainer wired for a CPU-only train()
# ---------------------------------------------------------------------------


def _build_test_sft_config(num_steps: int, batch_size: int) -> SFTConfig:
    cfg = SFTConfig()
    cfg.strategy = "fsdp"
    cfg.model.path = "unused"
    cfg.placement = SFTPlacementConfig(num_nodes=1, num_gpus_per_node=1)
    cfg.dataset_name = "unused-monkeypatched"
    cfg.dataset_split = "train"
    cfg.eval_dataset_name = ""  # no eval path
    cfg.eval_interval = 0
    cfg.eval_before_train = False
    cfg.num_steps = num_steps
    cfg.num_epochs = None
    cfg.batch_size = batch_size
    cfg.micro_train_batch_size_per_gpu = 1
    cfg.max_length = 64
    cfg.remove_microbatch_padding = False
    cfg.logger = "console"
    cfg.ckpt_path = ""  # no checkpointing
    cfg.ckpt_interval = 0
    cfg.hf_save_interval = 0
    cfg.enable_ray_gpu_monitor = False
    cfg.seed = 1234
    return cfg


def _distinct_tokenized(n: int) -> list[dict]:
    """``n`` examples with DISTINCT token ids and varying lengths.

    Distinct ids + varying lengths mean different orderings produce different
    collated batches, so a stale-shuffle prefetch bug cannot accidentally match
    the synchronous baseline.
    """
    examples = []
    for i in range(n):
        length = 6 + (i % 4)  # vary length 6..9 so packing/padding differs by order
        base = (i + 1) * 1000
        input_ids = [base + j for j in range(length)]
        num_actions = 1 + (i % 3)  # 1..3 response tokens
        examples.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * length,
                "num_actions": num_actions,
                "loss_mask": [1] * num_actions,
            }
        )
    return examples


def _make_trainer(cfg: SFTConfig, collator) -> SFTTrainer:
    # Pass a mock bridge config so we don't call ``build_skyrl_config_for_sft``,
    # which (via ``SkyRLTrainConfig.__post_init__``) eagerly imports the vllm
    # inference-server stack — unneeded for a CPU-only collate test.
    skyrl_cfg = MagicMock()
    trainer = SFTTrainer(cfg, skyrl_cfg=skyrl_cfg)

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    trainer.tokenizer = tokenizer
    trainer.collator = collator
    trainer.tracker = MagicMock()

    # Mock the worker dispatch: forward_backward / optim_step are the GPU touch
    # points in train_step; forward / dp_size are used by run_eval.
    step_output = MagicMock()
    step_output.metrics = {"loss": 0.5, "final_loss": 0.5}
    dispatch_mock = MagicMock()
    dispatch_mock.forward_backward = MagicMock(return_value=step_output)
    dispatch_mock.optim_step = MagicMock(return_value=1.0)
    dispatch_mock.dp_size = MagicMock(return_value=1)
    # run_eval reads a real float loss out of forward()'s output metrics.
    eval_output = MagicMock()
    eval_output.metrics = {"loss": 0.5}
    dispatch_mock.forward = MagicMock(return_value=eval_output)
    trainer.dispatch = dispatch_mock
    return trainer


def _snapshot_batch(batch) -> dict:
    """Deep, order-preserving snapshot of the tensor payload of a batch."""
    snap = {}
    for key in ("sequences", "attention_mask", "loss_mask"):
        if key in batch:
            snap[key] = batch[key].clone()
    # PackedDataCollator carries sub_seq_lengths (a TensorList of 1-D tensors).
    if "sub_seq_lengths" in batch:
        snap["sub_seq_lengths"] = [t.clone() for t in batch["sub_seq_lengths"]]
    return snap


def _capture_batches(trainer: SFTTrainer, monkeypatch, start_step: int = 0) -> list[dict]:
    """Run train() and capture a snapshot of every batch passed to train_step.

    ``start_step`` mocks ``load_checkpoint`` so the loop resumes from a nonzero
    step (the checkpoint-resume branch). The serial and prefetch runs must pass
    the SAME ``start_step`` for the byte-identity comparison to hold.
    """
    captured: list[dict] = []
    real_train_step = trainer.train_step

    def _spy_train_step(batch, step):
        captured.append({"step": step, "snap": _snapshot_batch(batch)})
        return real_train_step(batch, step)

    monkeypatch.setattr(trainer, "train_step", _spy_train_step)
    monkeypatch.setattr(trainer, "load_checkpoint", lambda: start_step)
    trainer.train()
    return captured


def _assert_batch_sequences_equal(serial: list[dict], prefetch: list[dict]):
    assert len(serial) == len(prefetch), f"step count differs: serial={len(serial)} prefetch={len(prefetch)}"
    for s, p in zip(serial, prefetch):
        assert s["step"] == p["step"], f"step mismatch: {s['step']} vs {p['step']}"
        s_snap, p_snap = s["snap"], p["snap"]
        assert s_snap.keys() == p_snap.keys(), f"step {s['step']} key mismatch: {s_snap.keys()} vs {p_snap.keys()}"
        for key in s_snap:
            sv, pv = s_snap[key], p_snap[key]
            if key == "sub_seq_lengths":
                assert len(sv) == len(pv), f"step {s['step']} sub_seq_lengths len differs"
                for a, b in zip(sv, pv):
                    assert torch.equal(a, b), f"step {s['step']} sub_seq_lengths differ"
            else:
                assert sv.shape == pv.shape, f"step {s['step']} {key} shape differs: {sv.shape} vs {pv.shape}"
                assert torch.equal(sv, pv), f"step {s['step']} {key} not byte-identical"


# ---------------------------------------------------------------------------
# Integration: prefetch ON == serial baseline across epoch boundaries
# ---------------------------------------------------------------------------


def _run_pair(
    monkeypatch,
    collator_factory,
    n_examples,
    batch_size,
    num_steps,
    *,
    start_step: int = 0,
    num_epochs: Optional[int] = None,
    eval_n_examples: int = 0,
    eval_interval: int = 0,
):
    """Run the loop serially then with prefetch; return (serial, prefetch) captures.

    Each run gets its OWN tokenized list (the loop shuffles in place) and its
    OWN collator so there is no cross-run state bleed.

    Keyword args exercise the non-default loop branches under prefetch:
      * ``start_step`` resumes from a (mocked) mid-epoch checkpoint.
      * ``num_epochs`` derives ``num_steps`` from epochs instead of pinning it
        (pass ``num_steps=None`` together with this).
      * ``eval_n_examples`` / ``eval_interval`` enable in-loop eval, which calls
        ``self.collator`` on the same instance the prefetch worker uses.
    """
    base = _distinct_tokenized(n_examples)
    eval_base = _distinct_tokenized(eval_n_examples) if eval_n_examples else None

    def _run(prefetch_on: bool):
        cfg = _build_test_sft_config(num_steps=num_steps, batch_size=batch_size)
        cfg.prefetch_data = prefetch_on
        if num_epochs is not None:
            cfg.num_epochs = num_epochs
        if eval_base is not None:
            cfg.eval_dataset_name = "unused-eval"
            cfg.eval_interval = eval_interval
        tokenized = copy.deepcopy(base)
        eval_tokenized = copy.deepcopy(eval_base) if eval_base is not None else None
        trainer = _make_trainer(cfg, collator_factory(cfg))
        monkeypatch.setattr(trainer, "load_dataset", lambda: tokenized)
        monkeypatch.setattr(trainer, "load_eval_dataset", lambda: eval_tokenized)
        return _capture_batches(trainer, monkeypatch, start_step=start_step)

    serial = _run(prefetch_on=False)
    prefetch = _run(prefetch_on=True)
    return serial, prefetch


def test_prefetch_matches_serial_default_collator(monkeypatch):
    """DefaultCollator: prefetched batches are byte-identical across 2 epoch boundaries.

    n_examples=6, batch_size=2 -> steps_per_epoch=3; num_steps=7 spans the
    boundaries after step 3 and step 6 (3 epochs touched).
    """
    n_examples, batch_size, num_steps = 6, 2, 7

    def factory(cfg):
        return DefaultCollator(MagicMock(pad_token_id=0), micro_train_batch_size_per_gpu=1)

    serial, prefetch = _run_pair(monkeypatch, factory, n_examples, batch_size, num_steps)
    assert len(serial) == num_steps, f"expected {num_steps} steps, got {len(serial)}"
    # Confirm the run actually crossed >=2 epoch boundaries (steps_per_epoch=3).
    assert num_steps // (n_examples // batch_size) >= 2
    _assert_batch_sequences_equal(serial, prefetch)


def test_prefetch_matches_serial_packed_collator(monkeypatch):
    """PackedDataCollator (FFD packing): prefetched batches byte-identical across boundaries.

    Constructed CPU-only (dp_size=1, tp/pp/cp=1) — the collate is pure
    numpy/torch and needs no Megatron runtime.
    """
    n_examples, batch_size, num_steps = 6, 2, 7

    def factory(cfg):
        return PackedDataCollator(
            tokenizer=MagicMock(pad_token_id=0),
            max_tokens_per_microbatch=64,
            tp_size=1,
            pp_size=1,
            cp_size=1,
            dp_size=1,
            batch_size=batch_size,
            micro_train_batch_size_per_gpu=1,
        )

    serial, prefetch = _run_pair(monkeypatch, factory, n_examples, batch_size, num_steps)
    assert len(serial) == num_steps
    # sub_seq_lengths must be present on the packed path (proves packing fired).
    assert all("sub_seq_lengths" in c["snap"] for c in prefetch)
    _assert_batch_sequences_equal(serial, prefetch)


def test_prefetch_matches_serial_uneven_wraparound(monkeypatch):
    """Wrap-around + uneven epoch: dataset not divisible by batch_size.

    n_examples=5, batch_size=2 -> the slice wraps around the dataset end inside
    an epoch and epoch boundaries land mid-list. Exercises the
    ``end_idx > len(tokenized)`` branch under prefetch.
    """
    n_examples, batch_size, num_steps = 5, 2, 9

    def factory(cfg):
        return DefaultCollator(MagicMock(pad_token_id=0), micro_train_batch_size_per_gpu=1)

    serial, prefetch = _run_pair(monkeypatch, factory, n_examples, batch_size, num_steps)
    assert len(serial) == num_steps
    _assert_batch_sequences_equal(serial, prefetch)


def test_prefetch_matches_serial_resume_mid_epoch(monkeypatch):
    """Resume from a nonzero (mid-epoch) checkpoint step under prefetch.

    Every other integration test pins ``load_checkpoint -> 0`` so the resume
    branch (``start_step > 0``) is never exercised under prefetch. Here both the
    serial and prefetch runs resume from the SAME mid-epoch ``start_step`` (the
    loop begins at ``start_step + 1``) and must still produce byte-identical
    batches. ``start_step`` is also chosen to span an epoch boundary so the
    resume-time shuffle replay + the first post-resume reshuffle both fire.

    n_examples=6, batch_size=2 -> steps_per_epoch=3. Resume from step=2 (epoch 0,
    mid-epoch) running to step 7 crosses the boundaries after steps 3 and 6.
    """
    n_examples, batch_size, num_steps, start_step = 6, 2, 7, 2

    def factory(cfg):
        return DefaultCollator(MagicMock(pad_token_id=0), micro_train_batch_size_per_gpu=1)

    serial, prefetch = _run_pair(monkeypatch, factory, n_examples, batch_size, num_steps, start_step=start_step)
    # The loop resumes at start_step + 1, so steps start_step+1..num_steps run.
    assert len(serial) == num_steps - start_step
    assert serial[0]["step"] == start_step + 1
    # Confirm we actually exercised the resume branch (mid-epoch start).
    assert start_step % (n_examples // batch_size) != 0
    _assert_batch_sequences_equal(serial, prefetch)


def test_prefetch_matches_serial_with_eval(monkeypatch):
    """Eval enabled (eval_interval > 0) under prefetch is byte-identical to serial.

    ``run_eval`` calls ``self.collator`` on the SAME collator instance the
    prefetch worker uses while a prefetch may be in flight; this confirms the
    interaction is safe and leaves training batches unchanged. All other
    integration tests disable eval.

    n_examples=6, batch_size=2 -> steps_per_epoch=3; eval fires at steps 2 and 4.
    """
    n_examples, batch_size, num_steps = 6, 2, 5

    def factory(cfg):
        return DefaultCollator(MagicMock(pad_token_id=0), micro_train_batch_size_per_gpu=1)

    serial, prefetch = _run_pair(
        monkeypatch,
        factory,
        n_examples,
        batch_size,
        num_steps,
        eval_n_examples=4,
        eval_interval=2,
    )
    assert len(serial) == num_steps
    _assert_batch_sequences_equal(serial, prefetch)


def test_prefetch_matches_serial_num_epochs_derived(monkeypatch):
    """``num_steps`` DERIVED from ``num_epochs`` (the common production path).

    The other integration configs set ``num_steps`` explicitly. Here ``num_steps``
    is None and ``num_epochs`` drives the loop length; assert the loop terminates
    at the derived count and is byte-identical to serial across the boundaries.

    n_examples=6, batch_size=2 -> steps_per_epoch=3; num_epochs=2 -> 6 steps.
    """
    n_examples, batch_size, num_epochs = 6, 2, 2
    expected_steps = num_epochs * (n_examples // batch_size)  # 6

    def factory(cfg):
        return DefaultCollator(MagicMock(pad_token_id=0), micro_train_batch_size_per_gpu=1)

    serial, prefetch = _run_pair(monkeypatch, factory, n_examples, batch_size, num_steps=None, num_epochs=num_epochs)
    assert len(serial) == expected_steps
    assert serial[-1]["step"] == expected_steps
    _assert_batch_sequences_equal(serial, prefetch)


def test_prefetch_worker_exception_teardown_propagates_original(monkeypatch):
    """A mid-loop failure with a batch in flight propagates the ORIGINAL error.

    ``train()`` wraps the loop in ``try/finally: prefetcher.shutdown()``, and
    ``shutdown() -> clear() -> future.result()`` can itself re-raise the
    background producer's exception. This asserts the exception the loop body
    raised (here from ``train_step``) is what propagates — i.e. it is not masked
    by a re-raise from the finally-block teardown — while a prefetch is in flight.

    Mirrors ``test_prefetcher_worker_exception_propagates`` but at the train()
    level. n_examples=6, batch_size=2 so prefetch is live by step 2.
    """
    n_examples, batch_size, num_steps = 6, 2, 5
    base = _distinct_tokenized(n_examples)

    cfg = _build_test_sft_config(num_steps=num_steps, batch_size=batch_size)
    cfg.prefetch_data = True
    tokenized = copy.deepcopy(base)
    collator = DefaultCollator(MagicMock(pad_token_id=0), micro_train_batch_size_per_gpu=1)
    trainer = _make_trainer(cfg, collator)
    monkeypatch.setattr(trainer, "load_dataset", lambda: tokenized)
    monkeypatch.setattr(trainer, "load_eval_dataset", lambda: None)
    monkeypatch.setattr(trainer, "load_checkpoint", lambda: 0)

    real_train_step = trainer.train_step

    def _failing_train_step(batch, step):
        # Fail on step 2: by now step 1 has already submitted the prefetch for
        # step 2, so a batch is genuinely in flight when the exception unwinds.
        if step == 2:
            raise RuntimeError("train-step boom")
        return real_train_step(batch, step)

    monkeypatch.setattr(trainer, "train_step", _failing_train_step)

    # The ORIGINAL train_step error must surface, not a teardown re-raise.
    with pytest.raises(RuntimeError, match="train-step boom"):
        trainer.train()


# ---------------------------------------------------------------------------
# Unit tests for the BatchPrefetcher invariants
# ---------------------------------------------------------------------------


def test_prefetcher_basic_submit_get():
    pf = BatchPrefetcher(lambda step: f"batch-{step}")
    try:
        assert pf.pending_step() is None
        pf.submit(5)
        assert pf.pending_step() == 5
        assert pf.get(5) == "batch-5"
        assert pf.pending_step() is None
    finally:
        pf.shutdown()


def test_prefetcher_step_mismatch_raises():
    pf = BatchPrefetcher(lambda step: step)
    try:
        pf.submit(3)
        with pytest.raises(AssertionError, match="!= expected step"):
            pf.get(4)  # asking for a different step than was submitted
    finally:
        pf.shutdown()


def test_prefetcher_double_submit_raises():
    pf = BatchPrefetcher(lambda step: step)
    try:
        pf.submit(1)
        with pytest.raises(AssertionError, match="slot already occupied"):
            pf.submit(2)  # single-slot invariant
    finally:
        pf.shutdown()


def test_prefetcher_worker_exception_propagates():
    def _boom(step):
        raise RuntimeError("producer failed")

    pf = BatchPrefetcher(_boom)
    try:
        pf.submit(1)
        with pytest.raises(RuntimeError, match="producer failed"):
            pf.get(1)
    finally:
        pf.shutdown()


def test_prefetcher_clear_drains_in_flight():
    pf = BatchPrefetcher(lambda step: step)
    try:
        pf.submit(7)
        pf.clear()
        assert pf.pending_step() is None
        # A fresh submit after clear works normally.
        pf.submit(8)
        assert pf.get(8) == 8
    finally:
        pf.shutdown()
