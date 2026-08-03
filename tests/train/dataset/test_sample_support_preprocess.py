import numpy as np
import pytest
import torch

from skyrl.train.dataset.preprocess import build_sample_support_replay


def _support(*rows: list[int]) -> np.ndarray:
    return np.asarray(rows, dtype=np.int32)


def test_dense_sample_support_is_response_aligned_int32():
    dense, csr_ids, csr_offsets = build_sample_support_replay(
        [_support([10, 11, -1], [12, -1, -1]), _support([20, 21, 22])],
        [[10, 12], [20]],
        [[1, 1], [1]],
        sequence_length=5,
        top_k=3,
        eos_token_id=2,
        use_sparse=False,
    )

    assert dense is not None
    assert csr_ids is None
    assert csr_offsets is None
    assert dense.dtype == torch.int32
    assert dense[0].tolist() == [[-1, -1, -1]] * 3 + [[10, 11, -1], [12, -1, -1]]
    assert dense[1].tolist() == [[-1, -1, -1]] * 4 + [[20, 21, 22]]


def test_sparse_sample_support_builds_per_trajectory_csr():
    dense, csr_ids, csr_offsets = build_sample_support_replay(
        [_support([10, 11, -1], [12, -1, -1]), _support([20, 21, 22])],
        [[10, 12], [20]],
        [[1, 1], [1]],
        sequence_length=5,
        top_k=3,
        eos_token_id=2,
        use_sparse=True,
    )

    assert dense is None
    assert csr_ids is not None
    assert csr_offsets is not None
    assert [tensor.tolist() for tensor in csr_ids.tensors] == [
        [10, 11, 12],
        [20, 21, 22],
    ]
    assert [tensor.tolist() for tensor in csr_offsets.tensors] == [[0, 2, 3], [0, 3]]
    assert all(tensor.dtype == torch.int32 for tensor in csr_ids.tensors + csr_offsets.tensors)


@pytest.mark.parametrize("use_sparse", [False, True])
def test_empty_masked_rows_and_loss_bearing_synthetic_eos_are_preserved(use_sparse):
    dense, csr_ids, csr_offsets = build_sample_support_replay(
        [_support([7, 8], [-1, -1], [-1, -1])],
        [[7, 9, 2]],
        [[1, 0, 1]],
        sequence_length=4,
        top_k=2,
        eos_token_id=2,
        use_sparse=use_sparse,
    )

    if use_sparse:
        assert dense is None
        assert csr_ids[0].tolist() == [7, 8]
        assert csr_offsets[0].tolist() == [0, 2, 2, 2]
    else:
        assert csr_ids is None
        assert csr_offsets is None
        assert dense.tolist() == [[[-1, -1], [7, 8], [-1, -1], [-1, -1]]]


def test_empty_loss_bearing_non_eos_is_rejected():
    with pytest.raises(ValueError, match="loss-bearing non-EOS"):
        build_sample_support_replay(
            [_support([-1, -1])],
            [[3]],
            [[1]],
            sequence_length=2,
            top_k=2,
            eos_token_id=2,
            use_sparse=True,
        )


@pytest.mark.parametrize("use_sparse", [False, True])
def test_multiple_loss_bearing_unsupported_eos_are_rejected(use_sparse):
    with pytest.raises(ValueError, match="more than one loss-bearing unsupported token"):
        build_sample_support_replay(
            [_support([-1, -1], [-1, -1])],
            [[2, 2]],
            [[1, 1]],
            sequence_length=2,
            top_k=2,
            eos_token_id=2,
            use_sparse=use_sparse,
        )


def test_loss_bearing_sampled_token_must_be_in_support():
    with pytest.raises(ValueError, match="sampled token 3 is missing"):
        build_sample_support_replay(
            [_support([2, 4])],
            [[3]],
            [[1]],
            sequence_length=2,
            top_k=2,
            eos_token_id=2,
            use_sparse=True,
        )
