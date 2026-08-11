"""Opt-in profiler ranges for Megatron router replay and MoE phases."""

from collections.abc import Callable
from contextlib import nullcontext
from enum import StrEnum
from functools import wraps

import torch


class MoEProfileRange(StrEnum):
    MICROBATCH_H2D = "r3/microbatch_h2d"
    REPLAY_ROUTES_H2D = "r3/replay_routes_h2d"
    REPLAY_LAYER_SELECT = "r3/replay_layer_select"
    REPLAY_METADATA_ALIGN = "r3/replay_metadata_align"
    REPLAY_TP_SLICE = "r3/replay_tp_slice"
    REPLAY_INSTALL = "r3/replay_install"
    REPLAY_PADDING_MASK_SCATTER = "r3/replay_padding_mask_scatter"
    REPLAY_BACKWARD_ACTION = "r3/replay_backward_action"
    ROUTER_GATE = "r3/router_gate"
    ROUTER_ROUTING_TOTAL = "r3/router_routing_total"
    ROUTER_REPLAY_SCORE = "r3/router_replay_score"
    DISPATCH_METADATA = "r3/dispatch_metadata"
    DISPATCH_PREPROCESS = "r3/dispatch_preprocess"
    DISPATCH_A2A = "r3/dispatch_a2a"
    DISPATCH_POSTPROCESS = "r3/dispatch_postprocess"
    ROUTED_EXPERT_COMPUTE = "r3/routed_expert_compute"
    SHARED_EXPERT_COMPUTE = "r3/shared_expert_compute"
    COMBINE_PREPROCESS = "r3/combine_preprocess"
    COMBINE_A2A = "r3/combine_a2a"
    COMBINE_POSTPROCESS = "r3/combine_postprocess"
    MOE_BACKWARD_DW = "r3/moe_backward_dw"


_annotations_enabled = False


def moe_profile_range(name: MoEProfileRange):
    """Return a record-function range only on ranks selected for profiling."""
    if not _annotations_enabled:
        return nullcontext()
    return torch.profiler.record_function(name.value)


def _profiled_method(name: MoEProfileRange, method: Callable) -> Callable:
    @wraps(method)
    def wrapped(*args, **kwargs):
        with torch.profiler.record_function(name.value):
            return method(*args, **kwargs)

    return wrapped


def install_megatron_moe_profile_annotations() -> None:
    """Annotate the pinned Megatron implementation without maintaining a fork."""
    global _annotations_enabled
    if _annotations_enabled:
        return

    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.moe.router import TopKRouter
    from megatron.core.transformer.moe.router_replay import RouterReplay
    from megatron.core.transformer.moe.token_dispatcher import (
        MoEAlltoAllTokenDispatcher,
    )

    TopKRouter.gating = _profiled_method(MoEProfileRange.ROUTER_GATE, TopKRouter.gating)
    TopKRouter.routing = _profiled_method(MoEProfileRange.ROUTER_ROUTING_TOTAL, TopKRouter.routing)
    RouterReplay.get_replay_topk = _profiled_method(
        MoEProfileRange.ROUTER_REPLAY_SCORE,
        RouterReplay.get_replay_topk,
    )

    MoEAlltoAllTokenDispatcher.preprocess = _profiled_method(
        MoEProfileRange.DISPATCH_METADATA,
        MoEAlltoAllTokenDispatcher.preprocess,
    )
    MoEAlltoAllTokenDispatcher.dispatch_preprocess = _profiled_method(
        MoEProfileRange.DISPATCH_PREPROCESS,
        MoEAlltoAllTokenDispatcher.dispatch_preprocess,
    )
    MoEAlltoAllTokenDispatcher.token_dispatch = _profiled_method(
        MoEProfileRange.DISPATCH_A2A,
        MoEAlltoAllTokenDispatcher.token_dispatch,
    )
    MoEAlltoAllTokenDispatcher.dispatch_postprocess = _profiled_method(
        MoEProfileRange.DISPATCH_POSTPROCESS,
        MoEAlltoAllTokenDispatcher.dispatch_postprocess,
    )
    MoEAlltoAllTokenDispatcher.combine_preprocess = _profiled_method(
        MoEProfileRange.COMBINE_PREPROCESS,
        MoEAlltoAllTokenDispatcher.combine_preprocess,
    )
    MoEAlltoAllTokenDispatcher.token_combine = _profiled_method(
        MoEProfileRange.COMBINE_A2A,
        MoEAlltoAllTokenDispatcher.token_combine,
    )
    MoEAlltoAllTokenDispatcher.combine_postprocess = _profiled_method(
        MoEProfileRange.COMBINE_POSTPROCESS,
        MoEAlltoAllTokenDispatcher.combine_postprocess,
    )

    MoELayer.routed_experts_compute = _profiled_method(
        MoEProfileRange.ROUTED_EXPERT_COMPUTE,
        MoELayer.routed_experts_compute,
    )
    MoELayer.shared_experts_compute = _profiled_method(
        MoEProfileRange.SHARED_EXPERT_COMPUTE,
        MoELayer.shared_experts_compute,
    )
    MoELayer.backward_dw = _profiled_method(MoEProfileRange.MOE_BACKWARD_DW, MoELayer.backward_dw)

    _annotations_enabled = True
