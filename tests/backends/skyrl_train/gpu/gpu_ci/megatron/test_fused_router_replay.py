"""Fused router-replay numerics, fallback, and checkpoint-recompute tests."""

import multiprocessing

import pytest
import torch

from skyrl.backends.skyrl_train.kernels import replay_router
from skyrl.backends.skyrl_train.utils import replay_utils

pytestmark = [
    pytest.mark.h100,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]

# Nemotron-3-Super-120B-A12B per-rank router shape (32768 seqlen / CP=2 / TP=4).
NUM_TOKENS = 512
NUM_EXPERTS = 512
TOPK = 22

PROB_ATOL = {torch.float32: 1e-6, torch.float16: 2e-4, torch.bfloat16: 1e-3}
GRAD_ATOL = {torch.float32: 1e-6, torch.float16: 2e-4, torch.bfloat16: 1e-3}
ROUTER_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _inputs(num_tokens=NUM_TOKENS, num_experts=NUM_EXPERTS, topk=TOPK, seed=0, dtype=torch.float32):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(num_tokens, num_experts, device="cuda", generator=generator).to(dtype)
    # Distinct experts per token, as vLLM's top-k always produces.
    indices = torch.argsort(torch.rand(num_tokens, num_experts, device="cuda", generator=generator), dim=1)[
        :, :topk
    ].to(torch.int32)
    expert_bias = torch.randn(num_experts, device="cuda", generator=generator)
    return logits, indices, expert_bias


def _fresh_replay(indices, action=None):
    """Create an isolated ``RouterReplay`` holding ``indices``."""
    from megatron.core.transformer.moe.router_replay import (
        RouterReplay,
        RouterReplayAction,
    )

    RouterReplay.clear_global_router_replay_instances()
    replay = RouterReplay()
    replay.set_target_indices(indices)
    replay.set_router_replay_action(action or RouterReplayAction.REPLAY_FORWARD)
    return replay


@pytest.fixture
def unpatched_routing():
    """Megatron's own ``topk_routing_with_score_function``, before any SkyRL patch."""
    from megatron.core.transformer.moe import moe_utils

    assert not getattr(
        moe_utils, "_fused_replay_patched", False
    ), "another test installed the fused-replay patch; it is process-global"
    return moe_utils.topk_routing_with_score_function


@pytest.fixture
def patched_routing(monkeypatch, unpatched_routing):
    """Install the fused-replay patch for one test and restore both bindings after."""
    from megatron.core.transformer.moe import moe_utils, router

    monkeypatch.setattr(moe_utils, "topk_routing_with_score_function", unpatched_routing)
    monkeypatch.setattr(router, "topk_routing_with_score_function", unpatched_routing)
    monkeypatch.setattr(moe_utils, "_fused_replay_patched", False, raising=False)
    monkeypatch.setattr(replay_utils, "_logged_fallback_reasons", set())

    def install(enable_fused_kernel=True):
        replay_utils.patch_topk_router_fused_replay(enable_fused_kernel=enable_fused_kernel)
        return moe_utils.topk_routing_with_score_function

    return install


def _routed_experts(routing_map):
    """Sorted expert ids per token, from a dense [tokens, experts] boolean map."""
    counts = routing_map.sum(dim=1)
    assert counts.min().item() == counts.max().item(), "ragged routing map"
    return routing_map.nonzero()[:, 1].view(routing_map.shape[0], -1).sort(dim=1).values


def _run_invalid_index(result_queue):
    """Exercise the device assertion in a child because it poisons that CUDA context."""
    logits = torch.randn(8, 16, device="cuda")
    indices = torch.zeros(8, 2, device="cuda", dtype=torch.int32)
    indices[0, 0] = 16
    try:
        assert replay_router.is_available(), replay_router.unavailable_reason()
        replay_router.fused_replay_routing_dense(logits, indices)
        torch.cuda.synchronize()
    except RuntimeError as exc:
        result_queue.put(str(exc))
        return
    result_queue.put("")


def test_extension_builds():
    assert replay_router.is_available(), replay_router.unavailable_reason()


def test_out_of_range_replay_index_fails_on_device(monkeypatch):
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    monkeypatch.delenv(replay_router.VALIDATE_INDICES_ENV_VAR, raising=False)
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=_run_invalid_index, args=(result_queue,))
    process.start()
    process.join(timeout=300)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("invalid-index CUDA child hung")
    assert process.exitcode == 0
    error = result_queue.get(timeout=10)
    assert "assert" in error.lower(), error


@pytest.mark.parametrize("dtype", ROUTER_DTYPES)
def test_forward_matches_unfused_replay(patched_routing, unpatched_routing, dtype):
    logits, indices, expert_bias = _inputs(dtype=dtype)

    reference = unpatched_routing(
        logits,
        TOPK,
        score_function=replay_utils.SIGMOID_SCORE_FUNCTION,
        expert_bias=expert_bias,
        fused=False,
        router_replay=_fresh_replay(indices.long()),
    )
    routing = patched_routing(enable_fused_kernel=True)
    got = routing(
        logits,
        TOPK,
        score_function=replay_utils.SIGMOID_SCORE_FUNCTION,
        expert_bias=expert_bias,
        fused=False,
        router_replay=_fresh_replay(indices),
    )

    max_delta = (got[0] - reference[0]).abs().max().item()
    print(f"forward max|delta probs| = {max_delta:.3e}")
    assert got[0].dtype == dtype
    assert not got[0].requires_grad, "frozen/no-grad routers must not acquire an autograd edge"
    assert max_delta < PROB_ATOL[dtype]
    assert torch.equal(got[1], reference[1])


@pytest.mark.parametrize("dtype", ROUTER_DTYPES)
def test_backward_matches_unfused_replay(patched_routing, unpatched_routing, dtype):
    logits, indices, expert_bias = _inputs(seed=1, dtype=dtype)
    cotangent = torch.randn_like(logits)

    def grad_of(routing, replay_indices):
        leaf = logits.detach().clone().requires_grad_(True)
        probs, _ = routing(
            leaf,
            TOPK,
            score_function=replay_utils.SIGMOID_SCORE_FUNCTION,
            expert_bias=expert_bias,
            fused=False,
            router_replay=_fresh_replay(replay_indices),
        )
        probs.backward(cotangent)
        return leaf.grad

    reference = grad_of(unpatched_routing, indices.long())
    got = grad_of(patched_routing(enable_fused_kernel=True), indices)

    max_delta = (got - reference).abs().max().item()
    print(f"backward max|delta grad| = {max_delta:.3e} (ref max|grad| {reference.abs().max():.3e})")
    assert got.dtype == dtype
    assert max_delta < GRAD_ATOL[dtype]


@pytest.mark.parametrize("dtype", ROUTER_DTYPES)
def test_backward_accumulates_duplicate_replay_indices(patched_routing, unpatched_routing, dtype):
    logits, indices, expert_bias = _inputs(num_tokens=64, seed=2, dtype=dtype)
    indices[:, 1] = indices[:, 0]
    cotangent = torch.randn_like(logits)

    def grad_of(routing, replay_indices):
        leaf = logits.detach().clone().requires_grad_(True)
        probs, _ = routing(
            leaf,
            TOPK,
            score_function=replay_utils.SIGMOID_SCORE_FUNCTION,
            expert_bias=expert_bias,
            fused=False,
            router_replay=_fresh_replay(replay_indices),
        )
        probs.backward(cotangent)
        return leaf.grad

    reference = grad_of(unpatched_routing, indices.long())
    got = grad_of(patched_routing(enable_fused_kernel=True), indices)

    assert (got - reference).abs().max().item() < GRAD_ATOL[dtype]


@pytest.mark.parametrize("enable_fused_kernel", [False, True])
def test_patch_never_lets_fusion_bypass_replay(patched_routing, enable_fused_kernel):
    logits, indices, expert_bias = _inputs(seed=3)
    routing = patched_routing(enable_fused_kernel=enable_fused_kernel)

    _, routing_map = routing(
        logits,
        TOPK,
        score_function=replay_utils.SIGMOID_SCORE_FUNCTION,
        expert_bias=expert_bias,
        fused=True,
        router_replay=_fresh_replay(indices),
    )

    assert routing_map.sum().item() == indices.numel()
    assert torch.equal(_routed_experts(routing_map), indices.long().sort(dim=1).values)


def test_unsupported_score_function_falls_back(patched_routing, unpatched_routing):
    logits, indices, _ = _inputs(seed=4)
    routing = patched_routing(enable_fused_kernel=True)

    reference = unpatched_routing(
        logits, TOPK, score_function="softmax", fused=False, router_replay=_fresh_replay(indices.long())
    )
    got = routing(logits, TOPK, score_function="softmax", fused=False, router_replay=_fresh_replay(indices))

    assert torch.equal(got[0], reference[0])
    assert any("score_function" in reason for reason in replay_utils._logged_fallback_reasons)


def test_backward_fifo_is_consumed_in_microbatch_order(patched_routing):
    """Consume replayed routes in microbatch order."""
    from megatron.core.transformer.moe.router_replay import (
        RouterReplay,
        RouterReplayAction,
    )

    routing = patched_routing(enable_fused_kernel=True)
    microbatches = [_inputs(num_tokens=64 + 16 * i, seed=10 + i) for i in range(4)]

    RouterReplay.clear_global_router_replay_instances()
    replay = RouterReplay()
    for _, indices, _ in microbatches:
        replay.set_target_indices(indices)

    replay.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
    for logits, indices, expert_bias in microbatches:
        _, routing_map = routing(
            logits,
            TOPK,
            score_function=replay_utils.SIGMOID_SCORE_FUNCTION,
            expert_bias=expert_bias,
            fused=False,
            router_replay=replay,
        )
        assert routing_map.shape == logits.shape
        assert torch.equal(_routed_experts(routing_map), indices.long().sort(dim=1).values)

    assert replay.replay_backward_list == []


@pytest.mark.parametrize("enable_fused_kernel", [False, True])
def test_selective_moe_recompute_replays_matching_routes(patched_routing, enable_fused_kernel):
    """Replay matching routes during selective MoE checkpoint recomputation."""
    from megatron.core import tensor_parallel
    from megatron.core.transformer.moe.router_replay import (
        RouterReplay,
        RouterReplayAction,
    )

    routing = patched_routing(enable_fused_kernel=enable_fused_kernel)
    microbatches = [_inputs(num_tokens=64 + 16 * i, seed=20 + i) for i in range(4)]
    recompute_routes = []

    RouterReplay.clear_global_router_replay_instances()
    replay = RouterReplay()

    def make_forward(expert_bias, cotangent):
        def moe_like_forward(hidden):
            routing_probs, routing_map = routing(
                hidden,
                TOPK,
                score_function=replay_utils.SIGMOID_SCORE_FUNCTION,
                expert_bias=expert_bias,
                fused=False,
                router_replay=replay,
            )
            if replay.router_replay_action == RouterReplayAction.REPLAY_BACKWARD:
                recompute_routes.append(_routed_experts(routing_map))
            # Avoid the near-zero gradient of normalized probabilities summed by row.
            return (routing_probs * cotangent).sum()

        return moe_like_forward

    outputs = []
    leaves = []
    forwards = []
    for logits, indices, expert_bias in microbatches:
        leaf = logits.detach().clone().requires_grad_(True)
        forward = make_forward(expert_bias, torch.randn_like(logits))
        leaves.append(leaf)
        forwards.append(forward)
        replay.set_target_indices(indices)
        replay.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
        outputs.append(tensor_parallel.checkpoint(forward, False, leaf))
        replay.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)

    assert len(replay.replay_backward_list) == len(microbatches), "forwards must not pop"
    assert recompute_routes == [], "no recompute should have happened yet"

    for output in outputs:
        output.backward()

    assert replay.replay_backward_list == [], "backward left routes unconsumed"
    assert len(recompute_routes) == len(microbatches)
    for consumed, (_, indices, _) in zip(recompute_routes, microbatches, strict=True):
        assert torch.equal(consumed, indices.long().sort(dim=1).values), "FIFO order desync"

    for leaf, forward, (logits, indices, _) in zip(leaves, forwards, microbatches, strict=True):
        direct_leaf = logits.detach().clone().requires_grad_(True)
        replay.set_target_indices(indices)
        replay.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
        forward(direct_leaf).backward()
        assert (leaf.grad - direct_leaf.grad).abs().max().item() < GRAD_ATOL[torch.float32]
    replay.clear_indices()
