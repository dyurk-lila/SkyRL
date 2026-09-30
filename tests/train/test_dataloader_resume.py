"""CPU unit tests for restoring the train dataloader position on resume (RL and SFT trainers)."""

from types import SimpleNamespace

import pytest
import torch
from loguru import logger
from torchdata.stateful_dataloader import StatefulDataLoader

from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl.train.sft_trainer import SFTTrainer
from skyrl.train.trainer import RayPPOTrainer

LOADERS = [RayPPOTrainer._load_dataloader_state, SFTTrainer._load_dataloader_state]


def _stub():
    return SimpleNamespace(train_dataloader=StatefulDataLoader(list(range(10)), batch_size=2))


@pytest.fixture
def warnings():
    messages = []
    handler_id = logger.add(messages.append, level="WARNING")
    yield messages
    logger.remove(handler_id)


@pytest.mark.parametrize("load", LOADERS)
def test_valid_state_restores_the_position(load, tmp_path):
    source = _stub()
    it = iter(source.train_dataloader)
    next(it)
    next(it)
    path = tmp_path / "data.pt"
    torch.save(source.train_dataloader.state_dict(), path)

    resumed = _stub()
    load(resumed, str(path))
    assert next(iter(resumed.train_dataloader)).tolist() == [4, 5]


@pytest.mark.parametrize("load", LOADERS)
def test_corrupt_state_raises_with_note(load, tmp_path):
    path = tmp_path / "data.pt"
    path.write_bytes(b"not a torch checkpoint")
    with pytest.raises(Exception) as exc_info:
        load(_stub(), str(path))
    (note,) = exc_info.value.__notes__
    assert str(path) in note and "refusing to restart data" in note


@pytest.mark.parametrize("load", LOADERS)
def test_missing_state_warns_and_restarts(load, tmp_path, warnings):
    stub = _stub()
    load(stub, str(tmp_path / "data.pt"))
    assert next(iter(stub.train_dataloader)).tolist() == [0, 1]
    (message,) = warnings
    assert str(tmp_path / "data.pt") in message


def test_fully_async_trainer_ignores_the_dataloader_state(tmp_path):
    path = tmp_path / "data.pt"
    path.write_bytes(b"not a torch checkpoint")
    FullyAsyncRayPPOTrainer._load_dataloader_state(_stub(), str(path))
