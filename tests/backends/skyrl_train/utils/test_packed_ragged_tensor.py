"""Tests for two-level ragged batch operations."""

import pytest
import torch

from skyrl.backends.skyrl_train.utils.packed_ragged_tensor import (
    PackedRaggedTensor,
    packed_ragged_padding_segments,
)
from skyrl.backends.skyrl_train.utils.packed_tensor import (
    CU_SEQLENS_DTYPE,
    PackedTensor,
    cu_seqlens_from_lengths,
)

PADDING = -1
MEMBERS: list[list[list[int]]] = [
    [[7, 8, 9], [4], [1, 2]],
    [[]],
    [[5, 6, 7, 8], [3, 4]],
]


def _build(members=MEMBERS) -> PackedRaggedTensor:
    flat = [member for entry in members for row in entry for member in row]
    row_lengths = [len(row) for entry in members for row in entry]
    return PackedRaggedTensor(
        torch.tensor(flat, dtype=torch.int32),
        cu_seqlens_from_lengths(row_lengths),
        cu_seqlens_from_lengths([len(entry) for entry in members]),
    )


def _read(packed: PackedRaggedTensor) -> list[list[list[int]]]:
    entries = []
    row = 0
    for length in packed.sequence_lengths.tolist():
        entries.append([packed.row(row + offset).tolist() for offset in range(length)])
        row += length
    return entries


def _padded(members=MEMBERS, *, width=4) -> PackedTensor:
    rows = [row + [PADDING] * (width - len(row)) for entry in members for row in entry]
    return PackedTensor(
        torch.tensor(rows, dtype=torch.int32),
        cu_seqlens_from_lengths([len(entry) for entry in members]),
    )


def test_compressing_fixed_width_rows_keeps_the_members_and_the_segmentation():
    ragged = PackedRaggedTensor.from_padded_rows(_padded(), padding_value=PADDING)

    assert _read(ragged) == MEMBERS
    assert len(ragged) == len(MEMBERS)
    assert ragged.sequence_lengths.tolist() == [3, 1, 2]
    assert ragged.row_lengths.tolist() == [3, 1, 2, 0, 4, 2]
    assert ragged.dtype == torch.int32


def test_an_all_padding_row_compresses_to_length_zero():
    padded = _padded([[[1, 2]], [[], []]], width=64)

    ragged = PackedRaggedTensor.from_padded_rows(padded, padding_value=PADDING)

    assert ragged.row_lengths.tolist() == [2, 0, 0]
    assert _read(ragged) == [[[1, 2]], [[], []]]
    assert padded.values.numel() == 3 * 64
    assert ragged.values.numel() == 2


def test_compressing_rejects_padding_that_is_not_trailing():
    padded = PackedTensor(torch.tensor([[1, PADDING, 3]], dtype=torch.int32), cu_seqlens_from_lengths([1]))

    with pytest.raises(ValueError, match="after every member"):
        PackedRaggedTensor.from_padded_rows(padded, padding_value=PADDING)


def test_compressing_rejects_rows_that_are_not_a_matrix():
    padded = PackedTensor(torch.zeros((2, 3, 4), dtype=torch.int32), cu_seqlens_from_lengths([2]))

    with pytest.raises(ValueError, match=r"\[rows, width\]"):
        PackedRaggedTensor.from_padded_rows(padded, padding_value=PADDING)


def test_a_row_count_of_zero_survives_the_round_trip():
    padded = PackedTensor(torch.empty((0, 4), dtype=torch.int32), cu_seqlens_from_lengths([0]))

    ragged = PackedRaggedTensor.from_padded_rows(padded, padding_value=PADDING)

    assert len(ragged) == 1
    assert ragged.num_rows == 0
    assert ragged.values.numel() == 0
    assert _read(ragged) == [[]]


def test_the_outer_offsets_mean_what_they_mean_on_packed_tensor():
    ragged = _build()
    padded = _padded()

    assert len(ragged) == len(padded)
    assert ragged.sequence_lengths.tolist() == padded.sequence_lengths.tolist()
    assert torch.equal(ragged.cu_seqlens, padded.cu_seqlens)
    assert ragged.num_rows == padded.values.shape[0]
    assert int(ragged.cu_seqlens[-1]) == int(padded.cu_seqlens[-1])


@pytest.mark.parametrize("bounds", [(0, 3), (1, 3), (0, 1), (2, 3)])
def test_contiguous_slice_selects_the_same_entries(bounds):
    ragged = _build()
    start, stop = bounds

    sliced = ragged[start:stop]

    assert _read(sliced) == MEMBERS[start:stop]
    assert int(sliced.row_offsets[0]) == 0
    assert int(sliced.cu_seqlens[0]) == 0


@pytest.mark.parametrize("indices", [[2, 0], [1, 1, 2], [0, 1, 2], [1]])
def test_gather_selects_entries_in_the_requested_order(indices):
    ragged = _build()

    for gathered in (ragged[torch.tensor(indices)], ragged[indices], ragged[tuple(indices)]):
        assert _read(gathered) == [MEMBERS[index] for index in indices]


def test_integer_and_negative_indices_return_that_entry():
    ragged = _build()

    assert _read(ragged[1]) == [MEMBERS[1]]
    assert _read(ragged.segment(-1)) == [MEMBERS[-1]]
    assert _read(ragged[torch.tensor(2)]) == [MEMBERS[2]]
    with pytest.raises(IndexError, match="out of range"):
        ragged.segment(len(ragged))


def test_row_addresses_the_whole_batch_and_rejects_a_missing_row():
    ragged = _build()

    assert ragged.row(0).tolist() == MEMBERS[0][0]
    assert ragged.row(3).tolist() == []
    assert ragged.row(-1).tolist() == MEMBERS[2][1]
    with pytest.raises(IndexError, match="out of range"):
        ragged.row(ragged.num_rows)


def test_strided_slice_falls_back_to_a_gather():
    ragged = _build()

    assert _read(ragged[::2]) == [MEMBERS[0], MEMBERS[2]]


def test_cat_joins_batches_end_to_end():
    left = _build(MEMBERS[:1])
    right = _build(MEMBERS[1:])

    joined = PackedRaggedTensor.cat([left, right])

    assert _read(joined) == MEMBERS
    with pytest.raises(ValueError, match="empty list of packed ragged batches"):
        PackedRaggedTensor.cat([])


def test_repeat_tiles_and_repeat_interleave_duplicates():
    ragged = _build(MEMBERS[:2])

    assert _read(ragged.repeat(2)) == MEMBERS[:2] + MEMBERS[:2]
    assert _read(ragged.repeat_interleave(2)) == [MEMBERS[0], MEMBERS[0], MEMBERS[1], MEMBERS[1]]


def test_to_contiguous_and_equality_preserve_both_levels():
    ragged = _build()

    widened = ragged.to(dtype=torch.int64)

    assert widened.dtype == torch.int64
    assert widened.row_offsets.dtype == CU_SEQLENS_DTYPE
    assert widened.cu_seqlens.dtype == CU_SEQLENS_DTYPE
    assert _read(widened) == MEMBERS
    assert ragged.contiguous() == ragged
    assert ragged == _build()
    assert ragged != _build(MEMBERS[:2])
    assert ragged != ragged.values


def test_equality_sees_a_regrouping_that_leaves_the_members_alone():
    regrouped = _build([[[7, 8, 9], [4]], [[1, 2], []], [[5, 6, 7, 8], [3, 4]]])

    assert torch.equal(regrouped.values, _build().values)
    assert regrouped != _build()


def test_repr_names_the_batch_the_rows_and_the_members():
    assert repr(_build()) == "PackedRaggedTensor(batch=3, rows=6, members=12, dtype=torch.int32)"


def test_rejects_values_that_are_not_a_flat_member_buffer():
    with pytest.raises(ValueError, match="flat member buffer"):
        PackedRaggedTensor(
            torch.zeros((2, 3), dtype=torch.int32),
            cu_seqlens_from_lengths([3, 3]),
            cu_seqlens_from_lengths([2]),
        )


def test_rejects_offsets_with_the_wrong_dtype_or_rank():
    values = torch.zeros(4, dtype=torch.int32)

    with pytest.raises(ValueError, match="row_offsets must be torch.int32"):
        PackedRaggedTensor(values, torch.tensor([0, 4], dtype=torch.int64), cu_seqlens_from_lengths([1]))
    with pytest.raises(ValueError, match="cu_seqlens must be torch.int32"):
        PackedRaggedTensor(values, cu_seqlens_from_lengths([4]), torch.tensor([0, 1], dtype=torch.int64))
    with pytest.raises(ValueError, match="row_offsets must be a non-empty 1-D"):
        PackedRaggedTensor(values, torch.zeros((2, 2), dtype=CU_SEQLENS_DTYPE), cu_seqlens_from_lengths([1]))
    with pytest.raises(ValueError, match="at least two offsets"):
        PackedRaggedTensor(values, cu_seqlens_from_lengths([4]), torch.zeros(1, dtype=CU_SEQLENS_DTYPE))


def test_rejects_offsets_that_do_not_span_their_level():
    values = torch.zeros(4, dtype=torch.int32)

    with pytest.raises(ValueError, match="must run from 0 to the 4 packed members"):
        PackedRaggedTensor(values, cu_seqlens_from_lengths([3]), cu_seqlens_from_lengths([1]))
    with pytest.raises(ValueError, match="must run from 0 to the 2 packed rows"):
        PackedRaggedTensor(values, cu_seqlens_from_lengths([2, 2]), cu_seqlens_from_lengths([1]))


def test_contiguous_slices_view_the_member_buffer():
    ragged = _build()

    # Batch entry 0 spends the first six members, and row 1 the fourth.
    assert ragged[1:3].values.data_ptr() == ragged.values[6:].data_ptr()
    assert ragged.row(1).data_ptr() == ragged.values[3:].data_ptr()


@pytest.mark.parametrize(
    "operation",
    [
        lambda packed: packed[torch.tensor([1, 0])],
        lambda packed: packed[::2],
        lambda packed: packed.repeat(2),
        lambda packed: packed.repeat_interleave(2),
        lambda packed: PackedRaggedTensor.cat([packed, packed]),
        lambda packed: packed.to(dtype=torch.int64),
    ],
    ids=["gather", "strided_slice", "repeat", "repeat_interleave", "cat", "to_dtype"],
)
def test_reordering_operations_allocate_rather_than_alias(operation):
    ragged = _build()
    original = ragged.values.clone()

    produced = operation(ragged)
    produced.values[:] = -9

    assert torch.equal(ragged.values, original)


def test_padding_segments_hold_the_requested_rows_with_no_members():
    padding = packed_ragged_padding_segments(_build(), segment_lengths=[2, 0, 1], members=())

    assert padding.sequence_lengths.tolist() == [2, 0, 1]
    assert padding.row_lengths.tolist() == [0, 0, 0]
    assert padding.values.numel() == 0
    assert padding.dtype == torch.int32


def test_padding_segments_can_carry_a_stated_member_row():
    padding = packed_ragged_padding_segments(_build(), segment_lengths=[2], members=(3, 4))

    assert padding.row_lengths.tolist() == [2, 2]
    assert padding.values.tolist() == [3, 4, 3, 4]
