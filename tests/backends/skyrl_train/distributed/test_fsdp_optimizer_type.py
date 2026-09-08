"""Tests that the FSDP backend maps the configured ``optimizer`` type to the
correct ``torch.optim`` class (and rejects the Megatron-only types).

``build_fsdp_optimizer`` builds a plain ``torch.optim`` optimizer over an iterable of
parameters, so these assertions run on CPU without FSDP wrapping, dist init, or CUDA.

Run with:
  uv run --isolated --extra dev -- pytest -s tests/backends/skyrl_train/distributed/test_fsdp_optimizer_type.py
"""

import pytest
import torch
from torch import optim

from skyrl.backends.skyrl_train.distributed.fsdp_strategy import build_fsdp_optimizer
from skyrl.train.config import OptimizerConfig


def _params():
    # A single CPU parameter is enough to construct any torch optimizer.
    return [torch.nn.Parameter(torch.zeros(2, 2))]


def test_adam_maps_to_adamw():
    opt = build_fsdp_optimizer(OptimizerConfig(optimizer="adam"), _params())
    assert isinstance(opt, optim.AdamW)
    # adam_betas/weight_decay are forwarded.
    assert opt.param_groups[0]["betas"] == tuple(OptimizerConfig().adam_betas)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(OptimizerConfig().weight_decay)


def test_sgd_maps_to_sgd():
    opt = build_fsdp_optimizer(OptimizerConfig(optimizer="sgd"), _params())
    assert isinstance(opt, optim.SGD)
    # AdamW is a subclass-free distinct type; ensure we did not fall through to AdamW.
    assert not isinstance(opt, optim.AdamW)


def test_fsdp_sgd_default_momentum_is_zero():
    # Default sgd_momentum=0.0 keeps FSDP SGD byte-identical to the historical behavior
    # (torch's momentum=0 default, i.e. vanilla SGD). Pin it so the default does not drift.
    opt = build_fsdp_optimizer(OptimizerConfig(optimizer="sgd"), _params())
    assert opt.param_groups[0]["momentum"] == 0.0


def test_fsdp_sgd_momentum_forwarded():
    # The new sgd_momentum field must reach torch.optim.SGD on the FSDP backend.
    opt = build_fsdp_optimizer(OptimizerConfig(optimizer="sgd", sgd_momentum=0.9), _params())
    assert opt.param_groups[0]["momentum"] == pytest.approx(0.9)


@pytest.mark.parametrize("optimizer", ["lion", "muon", "adaptive_muon", "soap", "arbitrary_emerging_name"])
def test_emerging_optimizers_raise_on_fsdp(optimizer):
    # FSDP cannot build any non-core optimizer; it must fail clearly rather than mis-dispatch.
    with pytest.raises(NotImplementedError, match="cross-backend core optimizers 'adam' or 'sgd'"):
        build_fsdp_optimizer(OptimizerConfig(optimizer=optimizer), _params())
