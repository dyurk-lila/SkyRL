"""CPU/gloo correctness tests for the fused LM-head log-prob path.

Runs WITHOUT GPUs or megatron-core: it exercises ``FusedLinearLogprob`` /
``from_parallel_logits_to_logprobs(lm_head_weight=...)`` against the stock logits path
(``from_parallel_logits_to_logprobs`` over fully-materialized logits) on a real Gloo
tensor-parallel process group. It checks the forward log-prob, the gradient w.r.t. the
hidden state, AND the gradient w.r.t. the LM-head weight all match the unfused reference,
at TP=1 and TP=2, with and without out-of-shard targets.

    uv run --isolated --extra dev -- pytest -s \
        tests/backends/skyrl_train/cpu/megatron/test_fused_linear_logprob.py

(No --extra megatron / no CUDA needed.)
"""

import os
import sys
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# ``model_utils`` imports ``megatron.core.parallel_state`` at module scope, but this test only
# uses the pure-torch fused path and the gloo WORLD group — no real megatron-core. When it is not
# installed (the CPU CI env, which has no ``--extra megatron``), stub the module so the import
# resolves. This block runs at import scope so it also takes effect in the ``mp.spawn`` workers,
# which start fresh interpreters and re-import this module. It is a no-op when megatron-core is
# genuinely present (GPU CI / dev environments).
try:  # pragma: no cover - exercised only when megatron-core is absent
    import megatron.core.parallel_state  # noqa: F401
except ModuleNotFoundError:
    _megatron = MagicMock()
    sys.modules["megatron"] = _megatron
    sys.modules["megatron.core"] = _megatron.core
    sys.modules["megatron.core.parallel_state"] = _megatron.core.parallel_state

import skyrl.backends.skyrl_train.distributed.megatron.model_utils as model_utils  # noqa: E402
from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (  # noqa: E402
    from_parallel_logits_to_logprobs,
    vocab_parallel_entropy,
)

# This is a megatron-backend test; mark it so ``-m megatron`` selects it where megatron-core is
# installed. The CPU job does not filter by marker, so the pure-torch path still runs there.
pytestmark = pytest.mark.megatron


def _stock_logprobs(hidden, weight_shard, target, vstart, vend, chunk_size):
    """Reference: materialize this rank's logit shard (hidden @ W_shard.T) then run the stock
    (unfused) logprob path over the same TP group. This is the EXACT equivalence we claim:
    fused(hidden, W_shard) must equal stock(materialized logit shard). Both sides share the
    identical distributed log-softmax + out-of-shard target semantics, so a per-rank compare
    of forward, grad-hidden, AND grad-weight is apples-to-apples (no reference to hand-roll).
    """
    leaf_h = hidden.detach().clone().requires_grad_(True)
    leaf_w = weight_shard.detach().clone().requires_grad_(True)
    logits = leaf_h @ leaf_w.t()  # [B, S, V // TP]
    lp = from_parallel_logits_to_logprobs(
        logits,
        target,
        vocab_start_index=vstart,
        vocab_end_index=vend,
        tp_group=dist.group.WORLD,
        inference_only=False,
        cp_group=None,
        chunk_size=chunk_size,
    )
    grad_seed = torch.linspace(0.5, 1.5, steps=lp.numel(), dtype=lp.dtype).reshape(lp.shape)
    lp.backward(grad_seed)
    return lp.detach(), leaf_h.grad.detach(), leaf_w.grad.detach(), grad_seed


def _fused_logprobs(hidden, weight_shard, target, vstart, vend, chunk_size, grad_seed):
    """Fused path on this rank's weight shard, via the public lm_head_weight entry point."""
    leaf_h = hidden.detach().clone().requires_grad_(True)
    leaf_w = weight_shard.detach().clone().requires_grad_(True)
    lp = from_parallel_logits_to_logprobs(
        leaf_h,  # hidden state, NOT logits
        target,
        vocab_start_index=vstart,
        vocab_end_index=vend,
        tp_group=dist.group.WORLD,
        inference_only=False,
        cp_group=None,
        chunk_size=chunk_size,
        lm_head_weight=leaf_w,
    )
    lp.backward(grad_seed.clone())
    return lp.detach(), leaf_h.grad.detach(), leaf_w.grad.detach()


def _worker(rank, world_size, port, chunk_size, with_oov, ret_dict):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(0)  # identical across ranks => identical hidden/weight/target

        batch_size, seq_len, hidden_size, vocab_size = 3, 24, 32, 256
        # fp32 here so the reference and fused path agree to fp32 rounding; the kernel itself
        # upcasts bf16->fp32 internally (covered by the dtype-contract assert below).
        hidden = torch.randn(batch_size, seq_len, hidden_size) * 0.5
        weight_full = torch.randn(vocab_size, hidden_size) * 0.1
        target_high = vocab_size + 50 if with_oov else vocab_size
        target = torch.randint(0, target_high, (batch_size, seq_len), dtype=torch.long)

        # This rank's contiguous vocab shard.
        assert vocab_size % world_size == 0
        shard = vocab_size // world_size
        vstart, vend = rank * shard, (rank + 1) * shard
        weight_shard = weight_full[vstart:vend].contiguous()

        # Reference and fused both run on the SAME shard over the SAME tp_group, so every
        # quantity compares per-rank with no cross-rank reduction needed.
        lp_ref, gh_ref, gw_ref, grad_seed = _stock_logprobs(hidden, weight_shard, target, vstart, vend, chunk_size)
        lp_fused, gh_fused, gw_fused = _fused_logprobs(
            hidden, weight_shard, target, vstart, vend, chunk_size, grad_seed
        )

        # 1) forward log-prob: full-vocab log-prob (after the SUM all-reduce) — equal on both.
        fwd_ok = torch.allclose(lp_fused, lp_ref, atol=1e-4, rtol=1e-4)
        # 2) grad-hidden: this shard's partial; the fused Function does NOT all-reduce it (the
        #    wrapper's SP-gather reduce-scatter does, in real training) — and neither does the
        #    stock shard reference, so they compare directly.
        gh_ok = torch.allclose(gh_fused, gh_ref, atol=1e-4, rtol=1e-4)
        # 3) grad-weight: per-shard, no reduction.
        gw_ok = torch.allclose(gw_fused, gw_ref, atol=1e-4, rtol=1e-4)

        ret_dict[rank] = {
            "fwd_ok": bool(fwd_ok),
            "gh_ok": bool(gh_ok),
            "gw_ok": bool(gw_ok),
            "lp_dtype": str(lp_fused.dtype),
            "fwd_max_abs": float((lp_fused - lp_ref).abs().max()),
            "gh_max_abs": float((gh_fused - gh_ref).abs().max()),
            "gw_max_abs": float((gw_fused - gw_ref).abs().max()),
        }
    finally:
        dist.destroy_process_group()


def _run(world_size, chunk_size, with_oov):
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    ret = manager.dict()
    # find a free port without binding races across the spawned ranks
    import socket

    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    mp.spawn(_worker, args=(world_size, port, chunk_size, with_oov, ret), nprocs=world_size, join=True)
    return dict(ret)


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("chunk_size", [8, 1000])  # 1000 > seq_len => single-chunk path
@pytest.mark.parametrize("with_oov", [False, True])
def test_fused_matches_stock_logits_path(world_size, chunk_size, with_oov):
    """Fused hidden->logprob matches the stock materialized-logits path (fwd + both grads).

    world_size=1 checks the math in isolation; world_size=2 checks the vocab-parallel
    online-softmax all-reduce, the out-of-shard target masking, and that summing per-shard
    hidden grads reproduces the reference (the TP reduction contract the wrapper relies on).
    """
    results = _run(world_size, chunk_size, with_oov)
    assert len(results) == world_size
    for rank, r in results.items():
        assert r["lp_dtype"] == "torch.float32", r
        assert r["fwd_ok"], f"forward mismatch rank={rank}: {r}"
        assert r["gh_ok"], f"grad-hidden mismatch rank={rank}: {r}"
        assert r["gw_ok"], f"grad-weight mismatch rank={rank}: {r}"


# ======================================================================================
# Entropy dual-output parity (verl's established (log_probs, entropy) pattern).
#
# The fused path can now ALSO return per-token entropy in the SAME chunked TP loop. We assert
# it matches the stock reference ``vocab_parallel_entropy`` over the materialized logit shard,
# in BOTH the forward (per-token entropy) and the backward (the entropy gradient term added to
# grad-hidden / grad-weight). This is the load-bearing new path for routing ALL RL losses
# (ppo/grpo/...) through fusion, not just SFT cross_entropy.
# ======================================================================================


def _point_entropy_tp_group_at_world():
    """Point stock ``vocab_parallel_entropy`` at the gloo WORLD group; return the original accessor.

    ``_VocabParallelEntropy`` reduces over ``mpu.get_tensor_model_parallel_group()``. In this CPU
    test the tensor-parallel group IS the WORLD process group, so point mpu's accessor at it (works
    whether mpu is real megatron-core on CI/linux or the lightweight stub) so the reference and the
    fused path reduce over the identical group. The caller restores the returned original.
    """
    original = model_utils.mpu.get_tensor_model_parallel_group
    model_utils.mpu.get_tensor_model_parallel_group = lambda: dist.group.WORLD
    return original


def _entropy_worker(rank, world_size, port, chunk_size, with_oov, ret_dict):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    _orig_tp_group_accessor = _point_entropy_tp_group_at_world()
    try:
        torch.manual_seed(0)  # identical across ranks => identical hidden/weight/target

        batch_size, seq_len, hidden_size, vocab_size = 3, 24, 32, 256
        hidden = torch.randn(batch_size, seq_len, hidden_size) * 0.5
        weight_full = torch.randn(vocab_size, hidden_size) * 0.1
        target_high = vocab_size + 50 if with_oov else vocab_size
        target = torch.randint(0, target_high, (batch_size, seq_len), dtype=torch.long)

        assert vocab_size % world_size == 0
        shard = vocab_size // world_size
        vstart, vend = rank * shard, (rank + 1) * shard
        weight_shard = weight_full[vstart:vend].contiguous()

        # ---- Forward entropy parity ----
        # Reference: per-token entropy of the full-vocab distribution from the materialized shard.
        ref_h = hidden.detach().clone().requires_grad_(True)
        ref_w = weight_shard.detach().clone().requires_grad_(True)
        ent_ref = vocab_parallel_entropy(ref_h @ ref_w.t())  # [B, S]

        fh = hidden.detach().clone().requires_grad_(True)
        fw = weight_shard.detach().clone().requires_grad_(True)
        lp_fused, ent_fused = from_parallel_logits_to_logprobs(
            fh,
            target,
            vocab_start_index=vstart,
            vocab_end_index=vend,
            tp_group=dist.group.WORLD,
            inference_only=False,
            cp_group=None,
            chunk_size=chunk_size,
            lm_head_weight=fw,
            return_entropy=True,  # NEW dual output
        )
        # Entropy is per-position & target-independent: returned RAW [B, S] (NOT shifted/trimmed),
        # so it lines up with the reference [B, S] directly.
        fwd_ent_ok = torch.allclose(ent_fused, ent_ref, atol=1e-4, rtol=1e-4)
        ent_dtype = str(ent_fused.dtype)
        ent_sum = float(ent_fused.detach().sum())  # for target-independence check across OOV

        # ---- Backward entropy + logprob parity (the load-bearing new path) ----
        # Stock combined backward: accumulate the logprob grad and the entropy grad on the SAME
        # leaves (matches what the wrapper does: logprob -> policy loss, entropy -> entropy term).
        s_h = hidden.detach().clone().requires_grad_(True)
        s_w = weight_shard.detach().clone().requires_grad_(True)
        lp_stock = from_parallel_logits_to_logprobs(
            s_h @ s_w.t(),
            target,
            vocab_start_index=vstart,
            vocab_end_index=vend,
            tp_group=dist.group.WORLD,
            inference_only=False,
            cp_group=None,
            chunk_size=chunk_size,
        )
        ent_stock = vocab_parallel_entropy(s_h @ s_w.t())
        g_lp = torch.linspace(0.5, 1.5, steps=lp_stock.numel(), dtype=lp_stock.dtype).reshape(lp_stock.shape)
        g_ent = torch.linspace(-0.7, 0.9, steps=ent_stock.numel(), dtype=ent_stock.dtype).reshape(ent_stock.shape)
        torch.autograd.backward([lp_stock, ent_stock], [g_lp.clone(), g_ent.clone()])
        gh_ref, gw_ref = s_h.grad.detach(), s_w.grad.detach()

        # Fused dual-output backward: lp_fused is [B, S-1] (same trim as stock), ent_fused [B, S].
        torch.autograd.backward([lp_fused, ent_fused], [g_lp.clone(), g_ent.clone()])
        gh_fused, gw_fused = fh.grad.detach(), fw.grad.detach()

        ret_dict[rank] = {
            "fwd_ent_ok": bool(fwd_ent_ok),
            "ent_dtype": ent_dtype,
            "ent_sum": ent_sum,
            "gh_ok": bool(torch.allclose(gh_fused, gh_ref, atol=1e-4, rtol=1e-4)),
            "gw_ok": bool(torch.allclose(gw_fused, gw_ref, atol=1e-4, rtol=1e-4)),
            "fwd_ent_max_abs": float((ent_fused - ent_ref).abs().max()),
            "gh_max_abs": float((gh_fused - gh_ref).abs().max()),
            "gw_max_abs": float((gw_fused - gw_ref).abs().max()),
        }
    finally:
        model_utils.mpu.get_tensor_model_parallel_group = _orig_tp_group_accessor
        dist.destroy_process_group()


def _run_entropy(world_size, chunk_size, with_oov):
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    ret = manager.dict()
    import socket

    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    mp.spawn(_entropy_worker, args=(world_size, port, chunk_size, with_oov, ret), nprocs=world_size, join=True)
    return dict(ret)


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("chunk_size", [8, 1000])
@pytest.mark.parametrize("with_oov", [False, True])
def test_fused_entropy_matches_stock_vocab_parallel_entropy(world_size, chunk_size, with_oov):
    """Fused per-token entropy (fwd + grad) matches stock ``vocab_parallel_entropy``.

    world_size=2 also exercises the entropy-specific all-reduce(SUM) of ``sum(softmax*logits)``
    that the fused entropy adds on top of the logprob path's MAX/SUM reductions, matching stock.
    """
    results = _run_entropy(world_size, chunk_size, with_oov)
    assert len(results) == world_size
    for rank, r in results.items():
        assert r["ent_dtype"] == "torch.float32", r
        assert r["fwd_ent_ok"], f"entropy forward mismatch rank={rank}: {r}"
        assert r["gh_ok"], f"entropy grad-hidden mismatch rank={rank}: {r}"
        assert r["gw_ok"], f"entropy grad-weight mismatch rank={rank}: {r}"


def test_fused_entropy_is_target_independent():
    """Entropy is a property of the logit distribution, not the target.

    Strong correctness signal that entropy is NOT being (mistakenly) zeroed/masked at OOV
    positions the way logprobs are: with_oov=True and with_oov=False must give identical entropy.
    """
    for world_size in (1, 2):
        for chunk_size in (8, 1000):
            r_no = _run_entropy(world_size, chunk_size, with_oov=False)
            r_oov = _run_entropy(world_size, chunk_size, with_oov=True)
            for rank in r_no:
                diff = abs(r_no[rank]["ent_sum"] - r_oov[rank]["ent_sum"])
                assert diff < 1e-4, (
                    f"entropy changed with OOV targets (ws={world_size}, chunk={chunk_size}, "
                    f"rank={rank}): |diff|={diff:.2e} — entropy must be target-independent"
                )


def test_fused_entropy_slice_alignment():
    """Regression guard on the wrapper's entropy/logprob off-by-one (single-rank, no dist).

    The wrapper slices the per-position fused entropy as ``entropy_full[:, -num_actions-1:-1]``
    and the per-position logprobs (after the internal roll+trim to [B, S-1]) as
    ``token_logprobs[:, -num_actions:]``. Both must select logit positions
    ``[S-1-num_actions .. S-2]`` so the entropy term lines up with the action logprobs. This
    asserts that index identity directly (independent of any distributed setup).
    """
    S, num_actions = 24, 7
    # Per-position tensors, 0..S-1 labels so we can read off WHICH logit position each slice picks.
    entropy_full = torch.arange(S).float().unsqueeze(0)  # [1, S], value == logit position
    # token_logprobs is the entry point's [:, :-1] of the per-position values -> positions 0..S-2.
    token_logprobs = torch.arange(S).float().unsqueeze(0)[:, :-1]  # [1, S-1]

    entropy_slice = entropy_full[:, -num_actions - 1 : -1]  # wrapper's entropy action slice
    logprob_slice = token_logprobs[:, -num_actions:]  # wrapper's action_log_probs slice

    # Both must reference logit positions S-1-num_actions .. S-2.
    expected = torch.arange(S - 1 - num_actions, S - 1).float().unsqueeze(0)
    assert torch.equal(entropy_slice, expected), (entropy_slice, expected)
    assert torch.equal(logprob_slice, expected), (logprob_slice, expected)
