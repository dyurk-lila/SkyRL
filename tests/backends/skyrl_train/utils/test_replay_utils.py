import inspect
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import (
    build_token_metadata_layout,
)
from skyrl.backends.skyrl_train.utils import replay_utils
from skyrl.backends.skyrl_train.utils.packed_tensor import PackedTensor
from skyrl.backends.skyrl_train.utils.replay_utils import make_replay_padding_indices


def _pack_routes(routes: torch.Tensor, attention_mask: torch.Tensor) -> PackedTensor:
    """Pack a ``[batch, seq_len, layers, topk]`` fixture to its real tokens."""
    return PackedTensor.from_segments([routes[row][attention_mask[row].bool()] for row in range(routes.shape[0])])


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


def test_replay_has_no_dispatcher_specific_patch():
    assert "TokenDispatcher" not in inspect.getsource(replay_utils)


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


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_replay_indices_are_dtype_independent(monkeypatch, parallel_state, packed, tp_size):
    """Compact host routes must produce the same int32 replay data as int32 routes."""
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


# Stands for a router the layer-number patch never touched, which has no such attribute.
_UNPATCHED = object()


def _install_router_replay(monkeypatch, layer_numbers):
    """Stub RouterReplay whose instances report the given 1-based Megatron layer numbers."""
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
    """Routes whose every value encodes the global layer index its slot came from."""
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


def test_replay_maps_interleaved_hybrid_routers_by_carried_layer(monkeypatch, parallel_state):
    """Nemotron-style alternating mamba/MoE stack: only the odd layers own a router.

    The capture spans all 8 layers, so slot selection must follow each router's own layer
    number. The tagged values catch a mapping that walks the routers positionally instead.
    """
    captured_layer_indices = tuple(range(8))
    moe_layers = (1, 3, 5, 7)
    router_replay = _install_router_replay(monkeypatch, [layer + 1 for layer in moe_layers])
    routes = _routes_tagged_by_layer(captured_layer_indices)

    _run_replay_setup(monkeypatch, routes, captured_layer_indices, stage_range=(0, 8))

    assert len(router_replay.replay_data) == len(moe_layers)
    for position, layer_index in enumerate(moe_layers):
        assert torch.all(router_replay.replay_data[position] == 300 + layer_index)


def test_replay_maps_an_all_moe_model_unchanged(monkeypatch, parallel_state):
    """Every layer owns a router, so the mapping is the identity."""
    captured_layer_indices = tuple(range(4))
    router_replay = _install_router_replay(monkeypatch, [1, 2, 3, 4])
    routes = _routes_tagged_by_layer(captured_layer_indices)

    _run_replay_setup(monkeypatch, routes, captured_layer_indices, stage_range=(0, 4))

    for slot, layer_index in enumerate(captured_layer_indices):
        assert torch.all(router_replay.replay_data[slot] == 300 + layer_index)


def test_replay_selects_a_pp_stage_whose_slots_are_not_zero_based(monkeypatch, parallel_state):
    """A later PP stage's slots start partway into the captured layer dimension.

    Routers for global layers 5 and 7 sit at slots 5 and 7, not at slots 0 and 1 nor at the
    slots an offset walk over this stage's own router list would produce.
    """
    captured_layer_indices = tuple(range(8))
    router_replay = _install_router_replay(monkeypatch, [6, 8])
    routes = _routes_tagged_by_layer(captured_layer_indices)

    _run_replay_setup(monkeypatch, routes, captured_layer_indices, stage_range=(4, 4))

    assert len(router_replay.replay_data) == 2
    assert torch.all(router_replay.replay_data[0] == 300 + 5)
    assert torch.all(router_replay.replay_data[1] == 300 + 7)


def test_replay_tolerates_a_captured_layer_that_owns_no_router(monkeypatch, parallel_state):
    """DeepSeek V3's layer 0 is dense: captured, but never replayed."""
    captured_layer_indices = tuple(range(4))
    router_replay = _install_router_replay(monkeypatch, [2, 3, 4])
    routes = _routes_tagged_by_layer(captured_layer_indices)

    _run_replay_setup(monkeypatch, routes, captured_layer_indices, stage_range=(0, 4))

    assert len(router_replay.replay_data) == 3
    assert torch.all(router_replay.replay_data[0] == 300 + 1)


def test_replay_raises_when_a_router_layer_was_not_captured(monkeypatch, parallel_state):
    """A router whose layer the rollout never captured must fail, not borrow a neighbour."""
    captured_layer_indices = tuple(range(4))
    _install_router_replay(monkeypatch, [2, 6])
    routes = _routes_tagged_by_layer(captured_layer_indices)

    with pytest.raises(ValueError, match="has no captured rollout routes"):
        _run_replay_setup(monkeypatch, routes, captured_layer_indices, stage_range=(0, 4))


def test_replay_raises_when_the_capture_does_not_cover_this_pp_stage(monkeypatch, parallel_state):
    """A stage running past the capture means the two stacks differ in depth.

    Every local router here IS captured, so only the stage-level check catches it -- and it has
    to, because the layer numbers those routers matched on came from a different model.
    """
    captured_layer_indices = tuple(range(4))
    _install_router_replay(monkeypatch, [3, 4])
    routes = _routes_tagged_by_layer(captured_layer_indices)

    with pytest.raises(ValueError, match=r"captured no routes for layers \[4, 5\]"):
        _run_replay_setup(monkeypatch, routes, captured_layer_indices, stage_range=(2, 4))


@pytest.mark.parametrize("layer_numbers", [[None, None], [_UNPATCHED, _UNPATCHED]])
def test_replay_raises_when_a_router_layer_number_is_unset(monkeypatch, parallel_state, layer_numbers):
    """Without patch_topk_router_layer_number the mapping is unknowable; guessing is unsafe.

    The patch injects ``layer_number``, so a router it never touched carries no such attribute
    at all; both that and an explicit ``None`` have to name the missing patch.
    """
    captured_layer_indices = tuple(range(4))
    _install_router_replay(monkeypatch, layer_numbers)
    routes = _routes_tagged_by_layer(captured_layer_indices)

    with pytest.raises(ValueError, match="no layer_number"):
        _run_replay_setup(monkeypatch, routes, captured_layer_indices, stage_range=(0, 4))


def test_replay_raises_on_layer_count_mismatch(monkeypatch, parallel_state):
    """The carried list must describe every slot of the route tensor's layer dimension."""
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
