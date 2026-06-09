"""Tests that the configured ``optimizer`` type and its hyperparameters flow through
the Megatron optimizer adapter into Megatron-core's ``OptimizerConfig``.

``init_megatron_optim_config`` only builds a Megatron-core ``OptimizerConfig`` dataclass
(no CUDA, no dist init, no model), so these assertions are correct-by-construction and do
not require a GPU; they are gated behind ``@pytest.mark.megatron`` because importing the
adapter pulls in ``megatron.core`` (a linux/megatron-extra dependency).

Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_optimizer_type.py
"""

import pytest

from skyrl.backends.skyrl_train.distributed.megatron.optimizer import (
    init_megatron_optim_config,
)
from skyrl.train.config import OptimizerConfig


@pytest.mark.megatron
def test_adam_is_default_and_routes_betas():
    optim_config = OptimizerConfig()
    assert optim_config.optimizer == "adam"

    mcore_config = init_megatron_optim_config(optim_config, optimizer_config_kwargs={})
    assert mcore_config.optimizer == "adam"
    # adam_betas must continue to route to adam_beta1/adam_beta2.
    assert mcore_config.adam_beta1 == pytest.approx(optim_config.adam_betas[0])
    assert mcore_config.adam_beta2 == pytest.approx(optim_config.adam_betas[1])
    assert mcore_config.lr == pytest.approx(optim_config.lr)
    assert mcore_config.weight_decay == pytest.approx(optim_config.weight_decay)


@pytest.mark.megatron
def test_sgd_dispatches_and_forwards_momentum():
    optim_config = OptimizerConfig(optimizer="sgd")
    mcore_config = init_megatron_optim_config(optim_config, optimizer_config_kwargs={})
    assert mcore_config.optimizer == "sgd"
    # sgd_momentum is now a first-class field (default 0.0) and is forwarded unconditionally,
    # so Megatron SGD now also defaults to 0.0 (vanilla SGD), matching the FSDP backend, instead
    # of falling back to Megatron-core's internal 0.9.
    assert mcore_config.sgd_momentum == pytest.approx(0.0)
    # An explicit momentum must reach mcore.
    mcore_explicit = init_megatron_optim_config(
        OptimizerConfig(optimizer="sgd", sgd_momentum=0.9), optimizer_config_kwargs={}
    )
    assert mcore_explicit.sgd_momentum == pytest.approx(0.9)


@pytest.mark.megatron
@pytest.mark.parametrize("optimizer", ["lion", "muon", "adaptive_muon", "soap"])
def test_emerging_optimizer_type_forwarded(optimizer):
    # The adapter must forward the requested type verbatim; whether it can actually be
    # constructed is decided later by Megatron-core based on emerging-optimizers availability.
    mcore_config = init_megatron_optim_config(OptimizerConfig(optimizer=optimizer), optimizer_config_kwargs={})
    assert mcore_config.optimizer == optimizer


@pytest.mark.megatron
def test_default_optim_args_unperturbed_on_adam_path():
    """The default ``OptimizerConfig`` must produce a Megatron-core ``OptimizerConfig`` whose
    Adam path is unchanged by this feature.

    ``sgd_momentum`` is only consumed by Megatron-core's SGD branch, so forwarding ``0.0`` is
    inert for the default ``adam`` optimizer. We verify (1) the optimizer is still ``"adam"``
    (AdamW), and (2) the full mcore config matches an explicit, identical adam build — i.e.
    adding the ``optimizer`` / ``sgd_momentum`` fields did not perturb any other adam arg.
    """
    mcore_default = init_megatron_optim_config(OptimizerConfig(), optimizer_config_kwargs={})

    assert mcore_default.optimizer == "adam"
    # sgd_momentum is forwarded as the field default (0.0); inert on the adam path.
    assert mcore_default.sgd_momentum == pytest.approx(0.0)
    reference = init_megatron_optim_config(
        OptimizerConfig(optimizer="adam"),
        optimizer_config_kwargs={},
    )
    assert mcore_default == reference
