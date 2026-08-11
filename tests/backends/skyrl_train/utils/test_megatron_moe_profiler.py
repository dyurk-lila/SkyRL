import sys
import types
from contextlib import contextmanager

from skyrl.backends.skyrl_train.utils import megatron_moe_profiler
from skyrl.backends.skyrl_train.utils.megatron_moe_profiler import MoEProfileRange


def _install_fake_megatron_modules(monkeypatch):
    for name in (
        "megatron",
        "megatron.core",
        "megatron.core.transformer",
        "megatron.core.transformer.moe",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    def method(self, *args, **kwargs):
        return args, kwargs

    class TopKRouter:
        gating = method
        routing = method

    class RouterReplay:
        get_replay_topk = method

    class MoEAlltoAllTokenDispatcher:
        preprocess = method
        dispatch_preprocess = method
        token_dispatch = method
        dispatch_postprocess = method
        combine_preprocess = method
        token_combine = method
        combine_postprocess = method

    class MoELayer:
        routed_experts_compute = method
        shared_experts_compute = method
        backward_dw = method

    modules = {
        "megatron.core.transformer.moe.router": ("TopKRouter", TopKRouter),
        "megatron.core.transformer.moe.router_replay": ("RouterReplay", RouterReplay),
        "megatron.core.transformer.moe.token_dispatcher": (
            "MoEAlltoAllTokenDispatcher",
            MoEAlltoAllTokenDispatcher,
        ),
        "megatron.core.transformer.moe.moe_layer": ("MoELayer", MoELayer),
    }
    for name, (attribute, value) in modules.items():
        module = types.ModuleType(name)
        setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)
    return TopKRouter, RouterReplay, MoEAlltoAllTokenDispatcher, MoELayer


def test_megatron_annotation_installer_is_gated_and_idempotent(monkeypatch):
    TopKRouter, RouterReplay, TokenDispatcher, MoELayer = _install_fake_megatron_modules(monkeypatch)
    monkeypatch.setattr(megatron_moe_profiler, "_annotations_enabled", False)
    entered = []

    @contextmanager
    def record_function(name):
        entered.append(name)
        yield

    monkeypatch.setattr(megatron_moe_profiler.torch.profiler, "record_function", record_function)

    megatron_moe_profiler.install_megatron_moe_profile_annotations()
    megatron_moe_profiler.install_megatron_moe_profile_annotations()

    TopKRouter().gating(1)
    TopKRouter().routing(1)
    RouterReplay().get_replay_topk(1)
    TokenDispatcher().preprocess(1)
    TokenDispatcher().dispatch_preprocess(1)
    TokenDispatcher().token_dispatch(1)
    TokenDispatcher().dispatch_postprocess(1)
    TokenDispatcher().combine_preprocess(1)
    TokenDispatcher().token_combine(1)
    TokenDispatcher().combine_postprocess(1)
    MoELayer().routed_experts_compute(1)
    MoELayer().shared_experts_compute(1)
    MoELayer().backward_dw(1)

    assert entered == [
        MoEProfileRange.ROUTER_GATE.value,
        MoEProfileRange.ROUTER_ROUTING_TOTAL.value,
        MoEProfileRange.ROUTER_REPLAY_SCORE.value,
        MoEProfileRange.DISPATCH_METADATA.value,
        MoEProfileRange.DISPATCH_PREPROCESS.value,
        MoEProfileRange.DISPATCH_A2A.value,
        MoEProfileRange.DISPATCH_POSTPROCESS.value,
        MoEProfileRange.COMBINE_PREPROCESS.value,
        MoEProfileRange.COMBINE_A2A.value,
        MoEProfileRange.COMBINE_POSTPROCESS.value,
        MoEProfileRange.ROUTED_EXPERT_COMPUTE.value,
        MoEProfileRange.SHARED_EXPERT_COMPUTE.value,
        MoEProfileRange.MOE_BACKWARD_DW.value,
    ]


def test_profile_range_is_noop_until_installed(monkeypatch):
    monkeypatch.setattr(megatron_moe_profiler, "_annotations_enabled", False)

    def unexpected_record_function(_name):
        raise AssertionError("disabled annotation attempted to enter torch.profiler")

    monkeypatch.setattr(
        megatron_moe_profiler.torch.profiler,
        "record_function",
        unexpected_record_function,
    )
    with megatron_moe_profiler.moe_profile_range(MoEProfileRange.REPLAY_INSTALL):
        pass
