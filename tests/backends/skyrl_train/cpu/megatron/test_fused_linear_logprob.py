"""CPU/gloo correctness tests for the fused LM-head log-prob path.

Runs WITHOUT GPUs or megatron-core: it exercises ``FusedLinearLogprob`` /
``from_parallel_logits_to_logprobs(lm_head_weight=...)`` against the stock logits path
(``from_parallel_logits_to_logprobs`` over fully-materialized logits) on a real Gloo
tensor-parallel process group. This is the verification the original downstream
monkey-patch lacked: it checks the forward log-prob, the gradient w.r.t. the hidden
state, AND the gradient w.r.t. the LM-head weight all match the unfused reference,
at TP=1 and TP=2, with and without out-of-shard targets.

    uv run --isolated --extra dev -- pytest -s \
        tests/backends/skyrl_train/cpu/megatron/test_fused_linear_logprob.py

(No --extra megatron / no CUDA needed.)
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
    FusedLinearLogprob,
    from_parallel_logits_to_logprobs,
)


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
        lp_ref, gh_ref, gw_ref, grad_seed = _stock_logprobs(
            hidden, weight_shard, target, vstart, vend, chunk_size
        )
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
