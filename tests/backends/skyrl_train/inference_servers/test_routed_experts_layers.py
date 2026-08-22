"""
uv run --isolated --extra dev pytest tests/backends/skyrl_train/inference_servers/test_routed_experts_layers.py
"""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from skyrl.backends.skyrl_train.inference_servers.routed_experts_layers import (
    MoELayerIndexResolver,
    collect_moe_layer_indices,
)


class _Engine:
    def __init__(self, per_worker):
        self._per_worker = per_worker
        self.calls = 0

    async def collective_rpc(self, method):
        self.calls += 1
        return self._per_worker


def _resolver(layer_indices):
    resolver = MoELayerIndexResolver(engine=None)
    resolver._layer_indices = tuple(layer_indices)
    return resolver


def _capture(num_layers, written_layers, *, tokens=4, topk=2):
    buffer = np.zeros((tokens, num_layers, topk), dtype=np.int32)
    for layer in written_layers:
        buffer[:, layer, :] = np.arange(1, topk + 1)
    return buffer


@pytest.mark.asyncio
async def test_resolve_reports_the_agreed_layers_once():
    engine = _Engine([[1, 3, 5], [1, 3, 5]])
    resolver = MoELayerIndexResolver(engine)

    assert await resolver.get() == (1, 3, 5)
    assert await resolver.get() == (1, 3, 5)
    assert engine.calls == 1


@pytest.mark.asyncio
async def test_resolve_raises_when_workers_disagree():
    resolver = MoELayerIndexResolver(_Engine([[1, 3], [1, 3, 5]]))

    with pytest.raises(RuntimeError, match="disagree on which layers are MoE"):
        await resolver.get()


@pytest.mark.asyncio
@pytest.mark.parametrize("per_worker", [[], [[]]])
async def test_resolve_raises_without_routed_moe_layers(per_worker):
    resolver = MoELayerIndexResolver(_Engine(per_worker))

    with pytest.raises(RuntimeError):
        await resolver.get()


def test_crosscheck_passes_when_written_layers_match():
    moe_layers = (1, 3, 5)
    resolver = _resolver(moe_layers)

    resolver.crosscheck_against_capture(_capture(6, moe_layers))


def test_crosscheck_raises_when_a_dropped_layer_carries_routes():
    resolver = _resolver((1, 3))

    with pytest.raises(RuntimeError, match=r"captured routes for layers \[4\]"):
        resolver.crosscheck_against_capture(_capture(6, (1, 3, 4)))


def test_crosscheck_allows_an_all_zero_selected_layer():
    resolver = _resolver((1, 3))

    resolver.crosscheck_against_capture(_capture(6, (1,)))


def test_crosscheck_runs_once_and_then_stops_scanning():
    resolver = _resolver((1, 3))
    resolver.crosscheck_against_capture(_capture(6, (1, 3)))

    # Would raise if it re-scanned: layer 4 is written but not selected.
    resolver.crosscheck_against_capture(_capture(6, (1, 3, 4)))


def test_crosscheck_is_inert_before_layers_resolve():
    resolver = MoELayerIndexResolver(engine=None)

    resolver.crosscheck_against_capture(_capture(6, (1, 3)))

    assert resolver._layer_indices is None


def test_vllm_moe_registry_api_contract():
    """Pin the vLLM interfaces used by ``collect_moe_layer_indices``."""
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    assert inspect.isclass(MoERunner), "isinstance() against MoERunner requires a class"
    assert isinstance(MoERunner.layer_id, property)
    # ``.router`` is set in __init__, so the class cannot be probed for it directly.
    assert "router" in inspect.signature(MoERunner.__init__).parameters
    assert callable(BaseRouter.set_capture_fn)


def test_collect_moe_layer_indices_selects_only_capturable_moe_layers():
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
        FusedMoERouter,
    )
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    def moe_runner(layer_id, router_spec=BaseRouter):
        runner = MagicMock(spec=MoERunner)
        runner.layer_id = layer_id
        runner.router = MagicMock(spec=router_spec)
        return runner

    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context={
                    "layers.0.self_attn": MagicMock(),
                    "layers.3.mlp": moe_runner(3),
                    "layers.1.mlp": moe_runner(1),
                    "layers.3.mlp.shard": moe_runner(3),
                    "layers.5.mlp": moe_runner(5, router_spec=FusedMoERouter),
                }
            )
        )
    )

    assert collect_moe_layer_indices(worker) == [1, 3]
