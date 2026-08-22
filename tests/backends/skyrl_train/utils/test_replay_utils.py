import sys
import types
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import (
    build_token_metadata_layout,
)
from skyrl.backends.skyrl_train.kernels.replay_router import _FusedReplayRoutingDense
from skyrl.backends.skyrl_train.utils import replay_utils
from skyrl.backends.skyrl_train.utils.packed_tensor import PackedTensor
from skyrl.backends.skyrl_train.utils.replay_utils import make_replay_padding_indices


def _pack_routes(routes: torch.Tensor, attention_mask: torch.Tensor) -> PackedTensor:
    """Pack a ``[batch, seq_len, layers, topk]`` fixture to its real tokens."""
    return PackedTensor.from_segments([routes[row][attention_mask[row].bool()] for row in range(routes.shape[0])])


@pytest.fixture
def replay_logs():
    """Collect ``(level, message)`` for every loguru record a test emits."""
    records: list[tuple[str, str]] = []
    sink_id = logger.add(
        lambda message: records.append((message.record["level"].name, message.record["message"])),
        level="INFO",
        enqueue=False,
    )
    yield records
    logger.remove(sink_id)


@pytest.fixture
def parallel_state(monkeypatch):
    try:
        import megatron.core.parallel_state as mpu
    except ModuleNotFoundError:
        megatron = types.ModuleType("megatron")
        core = types.ModuleType("megatron.core")
        mpu = types.ModuleType("megatron.core.parallel_state")
        megatron.core = core
        core.parallel_state = mpu
        monkeypatch.setitem(sys.modules, "megatron", megatron)
        monkeypatch.setitem(sys.modules, "megatron.core", core)
        monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", mpu)

    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0, raising=False)
    return mpu


def test_patch_topk_router_expert_bias_excludes_padding(monkeypatch):
    router_module = types.ModuleType("megatron.core.transformer.moe.router")

    class TopKRouter:
        def __init__(self):
            self.local_tokens_per_expert = torch.zeros(3, dtype=torch.int64)

        def _apply_expert_bias(self, routing_map, padding_mask=None):
            if padding_mask is not None:
                routing_map = routing_map & (~padding_mask)
            self.local_tokens_per_expert += routing_map.sum(dim=0)

    router_module.TopKRouter = TopKRouter
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.moe.router", router_module)

    replay_utils.patch_topk_router_expert_bias_padding_mask()
    router = TopKRouter()
    router._apply_expert_bias(
        torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.bool),
        torch.tensor([False, True]),
    )

    assert torch.equal(router.local_tokens_per_expert, torch.tensor([1, 0, 1]))


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int16, torch.int32])
def test_replay_padding_indices_are_unique(dtype):
    padding = make_replay_padding_indices((2, 3, 4, 3), dtype=dtype)

    assert padding.shape == (2, 3, 4, 3)
    assert torch.equal(padding, torch.tensor([0, 1, 2], dtype=dtype).expand_as(padding))


@pytest.mark.parametrize("shape", [(), (2, 3, 4, 0)])
def test_replay_padding_rejects_missing_topk(shape):
    with pytest.raises(ValueError, match="positive topk"):
        make_replay_padding_indices(shape, dtype=torch.uint8)


@pytest.mark.parametrize("route_dtype", [torch.uint8, torch.int16, torch.int32])
def test_setup_replay_installs_indices_and_returns_model_mask(monkeypatch, parallel_state, route_dtype):
    router_replay_module = types.ModuleType("megatron.core.transformer.moe.router_replay")

    class RouterReplay:
        global_router_replay_instances = [SimpleNamespace(layer_number=2)]
        replay_data = None
        action = None

        @classmethod
        def set_replay_data(cls, replay_data):
            cls.replay_data = replay_data

        @classmethod
        def set_global_router_replay_action(cls, action):
            cls.action = action

    class RouterReplayAction:
        REPLAY_FORWARD = "replay_forward"

    router_replay_module.RouterReplay = RouterReplay
    router_replay_module.RouterReplayAction = RouterReplayAction
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.moe.router_replay", router_replay_module)
    monkeypatch.setattr(replay_utils, "_get_current_pp_stage_layer_range", lambda model_config: (1, 1))
    monkeypatch.setattr(
        replay_utils,
        "scatter_router_padding_mask_for_model",
        lambda mask, model, model_config: mask,
    )
    monkeypatch.setattr(replay_utils, "_replayed_layer_count", None)
    apply_layout = replay_utils.align_packed_token_metadata
    routed_layer_counts = []

    def record_routed_layer_count(metadata, layout, padding_value):
        routed_layer_counts.append(metadata.row_shape[0])
        return apply_layout(metadata, layout, padding_value)

    monkeypatch.setattr(replay_utils, "align_packed_token_metadata", record_routed_layer_count)

    routes = torch.tensor(
        [
            [
                [[0, 1], [0, 1], [0, 1]],
                [[10, 11], [1, 2], [20, 21]],
                [[12, 13], [3, 4], [22, 23]],
                [[14, 15], [5, 6], [24, 25]],
            ]
        ],
        dtype=route_dtype,
    )
    attention_mask = torch.tensor([[0, 1, 1, 1]])
    router_padding_mask = torch.tensor([[1, 0, 0, 1]], dtype=torch.bool)
    metadata_layout = build_token_metadata_layout(
        attention_mask,
        routes.device,
        packed=False,
        fp8_enabled=False,
    )

    model_kwargs = replay_utils.setup_per_microbatch_replay_forward(
        _pack_routes(routes, attention_mask),
        (0, 1, 2),
        router_padding_mask,
        attention_mask,
        model=object(),
        model_config=SimpleNamespace(fp8=None),
        metadata_layout=metadata_layout,
    )

    assert RouterReplay.replay_data[0].tolist() == [[1, 2], [3, 4], [5, 6]]
    assert RouterReplay.replay_data[0].dtype == torch.int32
    assert RouterReplay.action == RouterReplayAction.REPLAY_FORWARD
    assert model_kwargs["padding_mask"].tolist() == [[False, False, True]]
    assert routed_layer_counts == [1]
    # What the path log reports: layers replayed on this rank, not the data's layer count.
    assert replay_utils._replayed_layer_count == 1


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_replay_indices_are_dtype_independent(monkeypatch, parallel_state, packed, tp_size):
    """Compact int16 routes produce the same int32 replay data as int32 routes."""
    router_replay_module = types.ModuleType("megatron.core.transformer.moe.router_replay")

    class RouterReplay:
        global_router_replay_instances = [SimpleNamespace(layer_number=2), SimpleNamespace(layer_number=3)]
        replay_data = None

        @classmethod
        def set_replay_data(cls, replay_data):
            cls.replay_data = replay_data

        @classmethod
        def set_global_router_replay_action(cls, action):
            pass

    router_replay_module.RouterReplay = RouterReplay
    router_replay_module.RouterReplayAction = SimpleNamespace(REPLAY_FORWARD="replay_forward")
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.moe.router_replay", router_replay_module)
    monkeypatch.setattr(replay_utils, "_get_current_pp_stage_layer_range", lambda model_config: (1, 2))
    monkeypatch.setattr(
        replay_utils,
        "scatter_router_padding_mask_for_model",
        lambda mask, model, model_config: mask,
    )
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: tp_size, raising=False)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: tp_size - 1, raising=False)

    batch, seq_len, num_layers, topk = 2, 8, 4, 2
    base = torch.arange(batch * seq_len * num_layers * topk, dtype=torch.int32) % 4096 + 256
    routes_int32 = base.reshape(batch, seq_len, num_layers, topk)
    routes_int16 = routes_int32.to(torch.int16)

    attention_mask = torch.ones((batch, seq_len), dtype=torch.long)
    attention_mask[0, 0] = 0
    router_padding_mask = torch.zeros((batch, seq_len), dtype=torch.bool)
    router_padding_mask[1, -1] = True

    def run(routes):
        layout = build_token_metadata_layout(
            attention_mask,
            routes.device,
            packed=packed,
            fp8_enabled=False,
        )
        replay_utils.setup_per_microbatch_replay_forward(
            _pack_routes(routes, attention_mask),
            range(num_layers),
            router_padding_mask,
            attention_mask,
            model=object(),
            model_config=SimpleNamespace(fp8=None, sequence_parallel=False),
            metadata_layout=layout,
            remove_microbatch_padding=packed,
        )
        return [tensor.clone() for tensor in RouterReplay.replay_data]

    from_int16 = run(routes_int16)
    from_int32 = run(routes_int32)

    assert len(from_int16) == len(from_int32) == 2
    for narrow, wide in zip(from_int16, from_int32, strict=True):
        assert narrow.dtype == wide.dtype == torch.int32
        assert torch.equal(narrow, wide)


_UNPATCHED = object()


def _install_router_replay(monkeypatch, layer_numbers):
    module = types.ModuleType("megatron.core.transformer.moe.router_replay")

    class RouterReplay:
        global_router_replay_instances = [
            object() if n is _UNPATCHED else SimpleNamespace(layer_number=n) for n in layer_numbers
        ]
        replay_data = None

        @classmethod
        def set_replay_data(cls, replay_data):
            cls.replay_data = replay_data

        @classmethod
        def set_global_router_replay_action(cls, action):
            pass

    module.RouterReplay = RouterReplay
    module.RouterReplayAction = SimpleNamespace(REPLAY_FORWARD="replay_forward")
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.moe.router_replay", module)
    return RouterReplay


def _routes_tagged_by_layer(captured_layer_indices, *, seq_len=2, topk=2):
    routes = torch.zeros((1, seq_len, len(captured_layer_indices), topk), dtype=torch.int16)
    for slot, layer_index in enumerate(captured_layer_indices):
        routes[:, :, slot, :] = 300 + layer_index
    return routes


def _run_replay_setup(monkeypatch, routes, captured_layer_indices, *, stage_range):
    monkeypatch.setattr(replay_utils, "_get_current_pp_stage_layer_range", lambda model_config: stage_range)
    monkeypatch.setattr(
        replay_utils,
        "scatter_router_padding_mask_for_model",
        lambda mask, model, model_config: mask,
    )
    batch, seq_len = routes.shape[0], routes.shape[1]
    attention_mask = torch.ones((batch, seq_len), dtype=torch.long)
    router_padding_mask = torch.zeros((batch, seq_len), dtype=torch.bool)
    layout = build_token_metadata_layout(attention_mask, routes.device, packed=False, fp8_enabled=False)
    return replay_utils.setup_per_microbatch_replay_forward(
        _pack_routes(routes, attention_mask),
        captured_layer_indices,
        router_padding_mask,
        attention_mask,
        model=object(),
        model_config=SimpleNamespace(fp8=None, sequence_parallel=False),
        metadata_layout=layout,
    )


@pytest.mark.parametrize(
    ("captured_layer_indices", "moe_layers", "stage_range"),
    [
        ((1, 3, 5, 7), (1, 3, 5, 7), (0, 8)),
        (tuple(range(4)), tuple(range(4)), (0, 4)),
        ((1, 3, 5, 7), (5, 7), (4, 4)),
        ((1, 2, 3), (1, 2, 3), (0, 4)),
    ],
    ids=["interleaved", "all-moe", "later-pp-stage", "leading-dense"],
)
def test_replay_maps_routers_by_carried_layer(
    monkeypatch, parallel_state, captured_layer_indices, moe_layers, stage_range
):
    router_replay = _install_router_replay(monkeypatch, [layer + 1 for layer in moe_layers])
    routes = _routes_tagged_by_layer(captured_layer_indices)

    _run_replay_setup(monkeypatch, routes, captured_layer_indices, stage_range=stage_range)

    assert len(router_replay.replay_data) == len(moe_layers)
    for position, layer_index in enumerate(moe_layers):
        assert torch.all(router_replay.replay_data[position] == 300 + layer_index)


def test_replay_raises_when_a_router_layer_was_not_captured(monkeypatch, parallel_state):
    captured_layer_indices = (1, 3)
    _install_router_replay(monkeypatch, [2, 6])
    routes = _routes_tagged_by_layer(captured_layer_indices)

    with pytest.raises(ValueError, match="has no captured rollout routes"):
        _run_replay_setup(monkeypatch, routes, captured_layer_indices, stage_range=(0, 8))


def test_replay_raises_when_a_captured_layer_in_this_stage_owns_no_router(monkeypatch, parallel_state):
    captured_layer_indices = (1, 3, 5, 7)
    _install_router_replay(monkeypatch, [2, 4])
    routes = _routes_tagged_by_layer(captured_layer_indices)

    with pytest.raises(ValueError, match=r"captured routes for MoE layers \[5\]"):
        _run_replay_setup(monkeypatch, routes, captured_layer_indices, stage_range=(0, 6))


@pytest.mark.parametrize("layer_numbers", [[None, None], [_UNPATCHED, _UNPATCHED]])
def test_replay_raises_when_a_router_layer_number_is_unset(monkeypatch, parallel_state, layer_numbers):
    captured_layer_indices = tuple(range(4))
    _install_router_replay(monkeypatch, layer_numbers)
    routes = _routes_tagged_by_layer(captured_layer_indices)

    with pytest.raises(ValueError, match="no layer_number"):
        _run_replay_setup(monkeypatch, routes, captured_layer_indices, stage_range=(0, 4))


def test_replay_raises_on_layer_count_mismatch(monkeypatch, parallel_state):
    _install_router_replay(monkeypatch, [2])
    routes = _routes_tagged_by_layer((0, 1))

    with pytest.raises(ValueError, match="captured layer indices were carried"):
        _run_replay_setup(monkeypatch, routes, (1,), stage_range=(0, 2))


def test_replay_raises_on_duplicate_captured_layers(monkeypatch, parallel_state):
    _install_router_replay(monkeypatch, [2])
    routes = _routes_tagged_by_layer((0, 1))

    with pytest.raises(ValueError, match="duplicates"):
        _run_replay_setup(monkeypatch, routes, (1, 1), stage_range=(0, 2))


@pytest.mark.parametrize(
    ("model_kind", "pre_process", "expected"),
    [
        ("gpt", True, [[False, False, True, True]]),
        ("gpt", False, [[True, True]]),
        ("hybrid", True, [[True, True]]),
    ],
)
def test_sequence_parallel_mask_layout(monkeypatch, model_kind, pre_process, expected):
    hybrid_model = types.ModuleType("megatron.core.models.hybrid.hybrid_model")
    tensor_parallel = types.ModuleType("megatron.core.tensor_parallel")
    utils = types.ModuleType("megatron.core.utils")

    class HybridModel:
        def __init__(self):
            self.pre_process = pre_process

    class GPTModel:
        def __init__(self):
            self.pre_process = pre_process

    hybrid_model.HybridModel = HybridModel
    tensor_parallel.scatter_to_sequence_parallel_region = lambda value: value.chunk(2, dim=0)[1]
    utils.unwrap_model = lambda model: model
    monkeypatch.setitem(sys.modules, "megatron.core.models.hybrid.hybrid_model", hybrid_model)
    monkeypatch.setitem(sys.modules, "megatron.core.tensor_parallel", tensor_parallel)
    monkeypatch.setitem(sys.modules, "megatron.core.utils", utils)

    mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.bool)
    model = HybridModel() if model_kind == "hybrid" else GPTModel()
    scattered = replay_utils.scatter_router_padding_mask_for_model(
        mask,
        model,
        SimpleNamespace(sequence_parallel=True),
    )

    assert scattered.tolist() == expected


@pytest.fixture
def router_replay_module(monkeypatch):
    module = types.ModuleType("megatron.core.transformer.moe.router_replay")
    router = SimpleNamespace(replay_backward_list=[], action=None)

    class RouterReplay:
        global_router_replay_instances = [router]

        @classmethod
        def clear_global_indices(cls):
            for instance in cls.global_router_replay_instances:
                instance.replay_backward_list = []

        @classmethod
        def clear_global_router_replay_action(cls):
            for instance in cls.global_router_replay_instances:
                instance.action = None

    module.RouterReplay = RouterReplay
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.moe.router_replay", module)
    return router


def test_router_replay_schedule_clears_stale_forward_only_fifo(router_replay_module):
    router_replay_module.replay_backward_list = ["stale-forward-only"]

    with replay_utils.router_replay_schedule(enabled=True):
        assert router_replay_module.replay_backward_list == []
        router_replay_module.replay_backward_list.extend(["microbatch-0", "microbatch-1"])
        assert router_replay_module.replay_backward_list.pop(0) == "microbatch-0"
        assert router_replay_module.replay_backward_list.pop(0) == "microbatch-1"

    assert router_replay_module.replay_backward_list == []
    assert router_replay_module.action is None


def test_router_replay_schedule_clears_after_exception(router_replay_module):
    with pytest.raises(RuntimeError, match="schedule failed"):
        with replay_utils.router_replay_schedule(enabled=True):
            router_replay_module.replay_backward_list.append("partially-consumed-schedule")
            router_replay_module.action = "replay-backward"
            raise RuntimeError("schedule failed")

    assert router_replay_module.replay_backward_list == []
    assert router_replay_module.action is None


# Fused dispatch and FIFO accounting with a pure-torch kernel stub.


def _reference_dense_routing(logits, indices, scaling=None):
    """Megatron's sigmoid replay contract in plain torch."""
    scores = torch.sigmoid(torch.gather(logits.float(), 1, indices.long()))
    probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    if scaling:
        probs = probs * scaling
    probs = probs.type_as(logits)
    routing_probs = torch.zeros_like(logits).scatter(1, indices.long(), probs)
    routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter(1, indices.long(), True)
    return routing_probs, routing_map


def _ensure_module(monkeypatch, name):
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    parent_name, _, leaf = name.rpartition(".")
    if parent_name:
        monkeypatch.setattr(_ensure_module(monkeypatch, parent_name), leaf, module, raising=False)
    return module


class _FusedReplayHarness:
    """Fake Megatron seam plus recorders for both dispatch outcomes."""

    def __init__(self, moe_utils, router, actions):
        self.moe_utils = moe_utils
        self.router = router
        self.actions = actions
        self.unfused_calls = []
        self.kernel_calls = []

    def original(self, logits, topk, **kwargs):
        self.unfused_calls.append(kwargs)
        indices = self.replay.target_topk_idx
        if self.replay.router_replay_action == self.actions.REPLAY_BACKWARD:
            indices = self.replay.replay_backward_list.pop(0)
        return _reference_dense_routing(logits, indices, kwargs.get("scaling_factor"))

    def kernel(self, logits, indices, scaling=None):
        self.kernel_calls.append(indices)
        return _reference_dense_routing(logits, indices, scaling)

    def route(self, logits, topk, **kwargs):
        kwargs.setdefault("score_function", replay_utils.SIGMOID_SCORE_FUNCTION)
        kwargs.setdefault("router_replay", self.replay)
        return self.moe_utils.topk_routing_with_score_function(logits, topk, **kwargs)


@pytest.fixture
def fused_replay(monkeypatch):
    from skyrl.backends.skyrl_train.kernels import replay_router

    moe_utils = _ensure_module(monkeypatch, "megatron.core.transformer.moe.moe_utils")
    router = _ensure_module(monkeypatch, "megatron.core.transformer.moe.router")
    replay_module = _ensure_module(monkeypatch, "megatron.core.transformer.moe.router_replay")

    if not hasattr(replay_module, "RouterReplayAction"):
        actions = SimpleNamespace(RECORD="record", REPLAY_FORWARD="replay_forward", REPLAY_BACKWARD="replay_backward")
        monkeypatch.setattr(replay_module, "RouterReplayAction", actions, raising=False)
    actions = replay_module.RouterReplayAction

    harness = _FusedReplayHarness(moe_utils, router, actions)
    harness.replay = SimpleNamespace(router_replay_action=None, target_topk_idx=None, replay_backward_list=[])
    monkeypatch.setattr(moe_utils, "topk_routing_with_score_function", harness.original, raising=False)
    monkeypatch.setattr(router, "topk_routing_with_score_function", harness.original, raising=False)
    monkeypatch.setattr(moe_utils, "_fused_replay_patched", False, raising=False)
    monkeypatch.setattr(replay_router, "unavailable_reason", lambda: None)
    monkeypatch.setattr(replay_router, "fused_replay_routing_dense", harness.kernel)
    monkeypatch.setattr(replay_utils, "_fused_replay_kernel_enabled", False)
    monkeypatch.setattr(replay_utils, "_logged_fallback_reasons", set())
    # One-shot log state is per process; monkeypatch restores it between tests.
    monkeypatch.setattr(replay_utils, "_logged_side_channel_path", False)
    monkeypatch.setattr(replay_utils, "_replayed_layer_count", None)
    return harness


def _logits_and_indices(num_tokens=6, num_experts=8, topk=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(num_tokens, num_experts, generator=generator)
    indices = torch.stack([torch.randperm(num_experts, generator=generator)[:topk] for _ in range(num_tokens)]).to(
        torch.int32
    )
    return logits, indices


def _routed_experts(routing_map):
    """Sorted expert ids per token, from a dense ``[tokens, experts]`` boolean map."""
    return routing_map.nonzero()[:, 1].view(routing_map.shape[0], -1).sort(dim=1).values


def test_fused_replay_patch_installs_on_both_bindings(fused_replay):
    replay_utils.patch_topk_router_fused_replay(enable_fused_kernel=True)

    assert fused_replay.moe_utils.topk_routing_with_score_function is not fused_replay.original
    assert (
        fused_replay.router.topk_routing_with_score_function is fused_replay.moe_utils.topk_routing_with_score_function
    )


@pytest.mark.parametrize("action", [None, "RECORD"])
def test_non_replay_calls_are_untouched(fused_replay, action):
    replay_utils.patch_topk_router_fused_replay(enable_fused_kernel=True)
    logits, indices = _logits_and_indices()
    fused_replay.replay.router_replay_action = None if action is None else fused_replay.actions.RECORD
    fused_replay.replay.target_topk_idx = indices

    fused_replay.route(logits, indices.shape[1], fused=True)

    # Recording/off is Megatron's business: fusion must not be second-guessed there.
    assert fused_replay.kernel_calls == []
    assert fused_replay.unfused_calls[0]["fused"] is True


def test_fast_path_serves_replay_forward_without_consuming_fifo(fused_replay):
    replay_utils.patch_topk_router_fused_replay(enable_fused_kernel=True)
    logits, indices = _logits_and_indices()
    fused_replay.replay.router_replay_action = fused_replay.actions.REPLAY_FORWARD
    fused_replay.replay.target_topk_idx = indices
    fused_replay.replay.replay_backward_list = [indices]

    routing_probs, routing_map = fused_replay.route(logits, indices.shape[1])

    assert fused_replay.unfused_calls == []
    assert len(fused_replay.kernel_calls) == 1
    assert torch.equal(_routed_experts(routing_map), indices.long().sort(dim=1).values)
    assert routing_map.sum().item() == indices.numel()
    assert torch.allclose(routing_probs.sum(dim=-1), torch.ones(indices.shape[0]), atol=1e-6)
    assert fused_replay.replay.replay_backward_list == [indices]


def test_fast_path_consumes_backward_fifo_in_microbatch_order(fused_replay):
    replay_utils.patch_topk_router_fused_replay(enable_fused_kernel=True)
    microbatches = [_logits_and_indices(num_tokens=4 + i, seed=i) for i in range(3)]

    fused_replay.replay.router_replay_action = fused_replay.actions.REPLAY_BACKWARD
    fused_replay.replay.replay_backward_list = [indices for _, indices in microbatches]

    for logits, indices in microbatches:
        _, routing_map = fused_replay.route(logits, indices.shape[1])
        assert routing_map.sum().item() == indices.numel()

    assert fused_replay.replay.replay_backward_list == []
    for consumed, (_, expected) in zip(fused_replay.kernel_calls, microbatches, strict=True):
        assert torch.equal(consumed, expected)


def test_fallback_does_not_consume_backward_fifo(fused_replay):
    """A rejected fast path must not pop: Megatron's own path pops right after."""
    replay_utils.patch_topk_router_fused_replay(enable_fused_kernel=False)
    logits, indices = _logits_and_indices()
    fused_replay.replay.router_replay_action = fused_replay.actions.REPLAY_BACKWARD
    fused_replay.replay.replay_backward_list = [indices]

    fused_replay.route(logits, indices.shape[1])

    assert fused_replay.kernel_calls == []
    assert len(fused_replay.unfused_calls) == 1
    # popped exactly once, by the unfused path
    assert fused_replay.replay.replay_backward_list == []


@pytest.mark.parametrize(
    ("kwargs", "expected_fragment"),
    [
        ({"score_function": "softmax"}, "score_function"),
        ({"dense_output": True}, "dense_output"),
        ({"unknown_future_arg": 1}, "unrecognized routing arguments"),
    ],
)
def test_unsupported_shapes_fall_back(fused_replay, kwargs, expected_fragment):
    replay_utils.patch_topk_router_fused_replay(enable_fused_kernel=True)
    logits, indices = _logits_and_indices()
    fused_replay.replay.router_replay_action = fused_replay.actions.REPLAY_FORWARD
    fused_replay.replay.target_topk_idx = indices

    fused_replay.route(logits, indices.shape[1], **kwargs)

    assert fused_replay.kernel_calls == []
    assert len(fused_replay.unfused_calls) == 1
    assert any(expected_fragment in reason for reason in replay_utils._logged_fallback_reasons)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_router_dtype_uses_kernel(fused_replay, dtype):
    replay_utils.patch_topk_router_fused_replay(enable_fused_kernel=True)
    logits, indices = _logits_and_indices()
    fused_replay.replay.router_replay_action = fused_replay.actions.REPLAY_FORWARD
    fused_replay.replay.target_topk_idx = indices

    routing_probs, _ = fused_replay.route(logits.to(dtype), indices.shape[1])

    assert len(fused_replay.kernel_calls) == 1
    assert routing_probs.dtype == dtype


def test_unsupported_router_dtype_falls_back(fused_replay):
    replay_utils.patch_topk_router_fused_replay(enable_fused_kernel=True)
    logits, indices = _logits_and_indices()
    fused_replay.replay.router_replay_action = fused_replay.actions.REPLAY_FORWARD
    fused_replay.replay.target_topk_idx = indices

    fused_replay.route(logits.to(torch.float64), indices.shape[1])

    assert fused_replay.kernel_calls == []
    assert any("router dtype" in reason for reason in replay_utils._logged_fallback_reasons)


def test_fused_replay_backward_accepts_missing_probability_gradient():
    assert _FusedReplayRoutingDense.backward(SimpleNamespace(), None, None) == (None, None, None)


@pytest.mark.parametrize("topk", [1, 33])
def test_unsupported_topk_falls_back(fused_replay, topk):
    replay_utils.patch_topk_router_fused_replay(enable_fused_kernel=True)
    logits, indices = _logits_and_indices(num_experts=64, topk=topk)
    fused_replay.replay.router_replay_action = fused_replay.actions.REPLAY_FORWARD
    fused_replay.replay.target_topk_idx = indices

    fused_replay.route(logits, topk)

    assert fused_replay.kernel_calls == []


def test_unavailable_extension_falls_back(fused_replay, monkeypatch):
    """A CPU-only host, or any toolchain that cannot build the extension, must degrade."""
    from skyrl.backends.skyrl_train.kernels import replay_router

    monkeypatch.setattr(replay_router, "unavailable_reason", lambda: "extension build failed (fake)")
    replay_utils.patch_topk_router_fused_replay(enable_fused_kernel=True)
    logits, indices = _logits_and_indices()
    fused_replay.replay.router_replay_action = fused_replay.actions.REPLAY_FORWARD
    fused_replay.replay.target_topk_idx = indices

    fused_replay.route(logits, indices.shape[1])

    assert fused_replay.kernel_calls == []
    assert len(fused_replay.unfused_calls) == 1


@pytest.mark.parametrize("enable_fused_kernel", [False, True])
def test_router_fusion_is_forced_off_while_replaying(fused_replay, enable_fused_kernel, monkeypatch):
    """The live silent-mis-train bug: Megatron's ``if fused:`` early return drops
    ``router_replay`` outright, so fused+replay trains against TE-selected experts at
    chance-level index overlap while every loss curve stays plausible."""
    from skyrl.backends.skyrl_train.kernels import replay_router

    if not enable_fused_kernel:
        monkeypatch.setattr(replay_router, "unavailable_reason", lambda: "no toolchain")
    replay_utils.patch_topk_router_fused_replay(enable_fused_kernel=enable_fused_kernel)
    logits, indices = _logits_and_indices()
    fused_replay.replay.router_replay_action = fused_replay.actions.REPLAY_FORWARD
    fused_replay.replay.target_topk_idx = indices

    _, routing_map = fused_replay.route(logits, indices.shape[1], fused=True)

    # Whichever path served it, the replayed experts are the ones that got routed.
    assert routing_map.sum().item() == indices.numel()
    assert torch.equal(_routed_experts(routing_map), indices.long().sort(dim=1).values)
    if enable_fused_kernel:
        assert len(fused_replay.kernel_calls) == 1
    else:
        # Forced unfused, and loudly: never hand fused=True to Megatron under replay.
        assert fused_replay.unfused_calls[0]["fused"] is False
        assert replay_utils._logged_fallback_reasons


def test_scaling_factor_reaches_the_kernel(fused_replay):
    replay_utils.patch_topk_router_fused_replay(enable_fused_kernel=True)
    logits, indices = _logits_and_indices()
    fused_replay.replay.router_replay_action = fused_replay.actions.REPLAY_FORWARD
    fused_replay.replay.target_topk_idx = indices

    scaled, _ = fused_replay.route(logits, indices.shape[1], scaling_factor=2.5)
    unscaled, _ = fused_replay.route(logits, indices.shape[1])

    assert torch.allclose(scaled, unscaled * 2.5, atol=1e-6)


def test_patch_is_idempotent(fused_replay):
    replay_utils.patch_topk_router_fused_replay(enable_fused_kernel=False)
    once = fused_replay.moe_utils.topk_routing_with_score_function
    replay_utils.patch_topk_router_fused_replay(enable_fused_kernel=True)

    assert fused_replay.moe_utils.topk_routing_with_score_function is once
    # A second call still updates the opt-in flag rather than leaving a stale wrapper.
    logits, indices = _logits_and_indices()
    fused_replay.replay.router_replay_action = fused_replay.actions.REPLAY_FORWARD
    fused_replay.replay.target_topk_idx = indices
    fused_replay.route(logits, indices.shape[1])

    assert len(fused_replay.kernel_calls) == 1


# Side-channel path observability.


@pytest.mark.parametrize(
    ("enable_fused_kernel", "expected_path"),
    [
        (True, replay_utils.SideChannelPath.FUSED_KERNEL),
        (False, replay_utils.SideChannelPath.UNFUSED),
    ],
)
def test_side_channel_path_is_logged_once_per_process(
    fused_replay, replay_logs, monkeypatch, enable_fused_kernel, expected_path
):
    """One line naming the path, layer count, topk and token count -- not one per call."""
    monkeypatch.setattr(replay_utils, "_replayed_layer_count", 3)
    replay_utils.patch_topk_router_fused_replay(enable_fused_kernel=enable_fused_kernel)
    logits, indices = _logits_and_indices(num_tokens=6, topk=3)
    fused_replay.replay.router_replay_action = fused_replay.actions.REPLAY_FORWARD
    fused_replay.replay.target_topk_idx = indices

    # 3 layers x 2 microbatches worth of router calls.
    for _ in range(6):
        fused_replay.route(logits, indices.shape[1])

    path_lines = [message for _, message in replay_logs if "MoE router replay active" in message]
    assert len(path_lines) == 1, path_lines
    assert str(expected_path) in path_lines[0]
    assert "3 layer(s)" in path_lines[0]
    assert "topk=3" in path_lines[0]
    assert "6 router tokens" in path_lines[0]


def test_non_replay_calls_log_no_path(fused_replay, replay_logs):
    """RECORD / replay-off routing is Megatron's business and must stay silent."""
    replay_utils.patch_topk_router_fused_replay(enable_fused_kernel=True)
    logits, indices = _logits_and_indices()
    fused_replay.replay.router_replay_action = fused_replay.actions.RECORD
    fused_replay.replay.target_topk_idx = indices

    fused_replay.route(logits, indices.shape[1])

    assert [message for _, message in replay_logs if "MoE router replay active" in message] == []


def test_missing_rollout_routes_warns_once(replay_logs, monkeypatch):
    """The silent-skip case: replay configured, no routes in the batch."""
    monkeypatch.setattr(replay_utils, "_warned_missing_rollout_routes", False)

    for _ in range(3):
        replay_utils.warn_if_training_without_replay(True, 2, 4)

    warnings = [message for level, message in replay_logs if level == "WARNING"]
    assert len(warnings) == 1, warnings
    assert "2/4" in warnings[0]
    assert "moe_enable_routing_replay=True" in warnings[0]
    assert "WITHOUT replay" in warnings[0]
    assert str(replay_utils.SideChannelPath.DISABLED) in warnings[0]


@pytest.mark.parametrize(
    ("replay_configured", "num_without_routes"),
    [(False, 4), (True, 0)],
)
def test_no_warning_when_replay_off_or_routes_present(replay_logs, monkeypatch, replay_configured, num_without_routes):
    monkeypatch.setattr(replay_utils, "_warned_missing_rollout_routes", False)

    replay_utils.warn_if_training_without_replay(replay_configured, num_without_routes, 4)

    assert [message for level, message in replay_logs if level == "WARNING"] == []
