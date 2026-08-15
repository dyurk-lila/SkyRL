"""Two-level packing for batch fields whose token rows are themselves ragged.

``PackedTensor`` describes exactly one ragged level: a batch of fixed-width token rows. Sampler
support needs a second one, because each generated token's recorded candidate set has its own
length, so this adds the inner level as a second flat offsets array.

It is a distinct class rather than an optional level on ``PackedTensor`` for two reasons. An
optional inner level would leave that class's slice fast path, ``segment``, ``_gather``, ``cat``,
``repeat`` and ``repeat_interleave`` addressing the outer level alone: each would return corrupt
data while every dtype, device and batch-size check still passed. And ``PackedTensor`` is
upstream code, while the inner level is not.
"""

from collections.abc import Sequence

import torch

from skyrl.backends.skyrl_train.utils.packed_tensor import (
    CU_SEQLENS_DTYPE,
    PackedTensor,
    cu_seqlens_from_lengths,
    lengths_from_offsets,
    row_index_from_offsets,
)


class PackedRaggedTensor:
    """A ragged batch of ragged token rows: one member buffer plus two offset arrays.

    ``values`` is the ``[sum(row_lengths)]`` flat member buffer in canonical batch order,
    ``row_offsets`` is the ``[rows + 1]`` exclusive prefix sum of the per-row member counts, and
    ``cu_seqlens`` is the ``[batch + 1]`` exclusive prefix sum of the per-batch-entry ROW counts.

    ``cu_seqlens`` means exactly what it means on ``PackedTensor`` -- rows per batch entry -- so
    both forms answer every batch operation identically and share one batch field. A consumer
    tells them apart by the inner level being present, never by which field they arrived in.

    ``values`` stays a plain ``torch.Tensor``, so ``to``, ``contiguous`` and ``pin_memory`` are
    ordinary tensor calls and the per-buffer zero-copy plasma encoding still applies to it.
    """

    def __init__(self, values: torch.Tensor, row_offsets: torch.Tensor, cu_seqlens: torch.Tensor):
        if values.ndim != 1:
            raise ValueError(f"packed ragged values must be a flat member buffer, got shape {tuple(values.shape)}")
        for name, offsets in (("row_offsets", row_offsets), ("cu_seqlens", cu_seqlens)):
            if offsets.ndim != 1 or offsets.numel() < 1:
                raise ValueError(f"{name} must be a non-empty 1-D offsets array, got shape {tuple(offsets.shape)}")
            if offsets.dtype != CU_SEQLENS_DTYPE:
                raise ValueError(f"{name} must be {CU_SEQLENS_DTYPE}, got {offsets.dtype}")
            if offsets.device != values.device:
                raise ValueError(
                    f"packed ragged values and {name} must share a device, got {values.device} and {offsets.device}"
                )
        if cu_seqlens.numel() < 2:
            raise ValueError(f"cu_seqlens must hold at least two offsets, got shape {tuple(cu_seqlens.shape)}")
        if int(row_offsets[0]) != 0 or int(row_offsets[-1]) != values.shape[0]:
            raise ValueError(
                f"row_offsets must run from 0 to the {values.shape[0]} packed members, "
                f"got {int(row_offsets[0])} to {int(row_offsets[-1])}"
            )
        num_rows = row_offsets.numel() - 1
        if int(cu_seqlens[0]) != 0 or int(cu_seqlens[-1]) != num_rows:
            raise ValueError(
                f"cu_seqlens must run from 0 to the {num_rows} packed rows, "
                f"got {int(cu_seqlens[0])} to {int(cu_seqlens[-1])}"
            )
        self.values = values
        self.row_offsets = row_offsets
        self.cu_seqlens = cu_seqlens

    @classmethod
    def from_padded_rows(cls, padded: PackedTensor, *, padding_value: int) -> "PackedRaggedTensor":
        """Compress a fixed-width ``[rows, width]`` packed field, keeping its outer segmentation.

        Rows are right-padded, so a row's members are its leading run and a boolean select in
        row-major order already concatenates them in packed row order. A row that is padding all
        the way across becomes a row of length zero: the offsets array is the only trace it
        leaves, which is what makes an observation token or a synthetic EOS free here.
        """
        if padded.values.ndim != 2:
            raise ValueError(f"padded rows must be [rows, width], got shape {tuple(padded.values.shape)}")
        members = padded.values != padding_value
        if bool((members[:, :-1] < members[:, 1:]).any()):
            raise ValueError(f"padded rows must place their {padding_value} padding after every member")
        return cls(
            padded.values[members],
            cu_seqlens_from_lengths(members.sum(dim=1), device=padded.device),
            padded.cu_seqlens,
        )

    @property
    def num_rows(self) -> int:
        return self.row_offsets.numel() - 1

    @property
    def row_lengths(self) -> torch.Tensor:
        """Return the ``[rows]`` member count of every row, across the whole batch."""
        return lengths_from_offsets(self.row_offsets)

    @property
    def sequence_lengths(self) -> torch.Tensor:
        """Return the ``[batch]`` ROW count of each batch entry, as ``PackedTensor`` does."""
        return lengths_from_offsets(self.cu_seqlens)

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def dtype(self) -> torch.dtype:
        return self.values.dtype

    def __len__(self) -> int:
        return self.cu_seqlens.numel() - 1

    def __getitem__(self, index) -> "PackedRaggedTensor":
        if isinstance(index, slice):
            if index.step in (None, 1):
                start, stop, _ = index.indices(len(self))
                stop = max(start, stop)
                row_bounds = self.cu_seqlens[start : stop + 1]
                member_offsets = self.row_offsets[int(row_bounds[0]) : int(row_bounds[-1]) + 1]
                return PackedRaggedTensor(
                    self.values[int(member_offsets[0]) : int(member_offsets[-1])],
                    member_offsets - member_offsets[0],
                    row_bounds - row_bounds[0],
                )
            return self._gather(range(*index.indices(len(self))))
        if isinstance(index, torch.Tensor):
            if index.ndim == 0:
                return self.segment(int(index))
            return self._gather(index.tolist())
        if isinstance(index, (list, tuple, range)):
            return self._gather(index)
        return self.segment(index)

    def segment(self, index: int) -> "PackedRaggedTensor":
        """Return one batch entry as a single-entry ragged batch.

        ``PackedTensor.segment`` can hand back a bare row block because its rows are uniform;
        here the row lengths are part of the answer, so the inner level comes along.
        """
        position = index + len(self) if index < 0 else index
        if not 0 <= position < len(self):
            raise IndexError(f"segment {index} is out of range for a packed batch of {len(self)}")
        return self[position : position + 1]

    def row(self, row_index: int) -> torch.Tensor:
        """Return one token row's members as a view, addressed across the whole batch."""
        position = row_index + self.num_rows if row_index < 0 else row_index
        if not 0 <= position < self.num_rows:
            raise IndexError(f"row {row_index} is out of range for {self.num_rows} packed rows")
        return self.values[int(self.row_offsets[position]) : int(self.row_offsets[position + 1])]

    def _gather(self, indices: Sequence[int]) -> "PackedRaggedTensor":
        """Select batch entries in the requested order into freshly allocated buffers.

        Both levels are rebuilt: the outer index picks whole row runs, and the inner index then
        picks each selected row's members. ``row_index_from_offsets`` does both walks, because
        "lay these (start, length) runs down back to back" is the same problem twice.
        """
        selected = torch.as_tensor(list(indices), dtype=torch.long, device=self.values.device)
        row_counts = self.sequence_lengths.to(torch.long)[selected]
        row_index = row_index_from_offsets(self.cu_seqlens[:-1].to(torch.long)[selected], row_counts)
        row_lengths = self.row_lengths.to(torch.long)[row_index]
        member_index = row_index_from_offsets(self.row_offsets[:-1][row_index], row_lengths)
        return PackedRaggedTensor(
            self.values.index_select(0, member_index),
            cu_seqlens_from_lengths(row_lengths, device=self.values.device),
            cu_seqlens_from_lengths(row_counts, device=self.values.device),
        )

    def to(self, device=None, dtype=None, non_blocking: bool = False) -> "PackedRaggedTensor":
        return PackedRaggedTensor(
            self.values.to(device=device, dtype=dtype, non_blocking=non_blocking),
            self.row_offsets.to(device=device, non_blocking=non_blocking),
            self.cu_seqlens.to(device=device, non_blocking=non_blocking),
        )

    def contiguous(self) -> "PackedRaggedTensor":
        return PackedRaggedTensor(self.values.contiguous(), self.row_offsets.contiguous(), self.cu_seqlens.contiguous())

    def pin_memory(self) -> "PackedRaggedTensor":
        return PackedRaggedTensor(self.values.pin_memory(), self.row_offsets.pin_memory(), self.cu_seqlens.pin_memory())

    def repeat(self, repeats: int) -> "PackedRaggedTensor":
        return self._gather(list(range(len(self))) * repeats)

    def repeat_interleave(self, repeats: int) -> "PackedRaggedTensor":
        return self._gather([index for index in range(len(self)) for _ in range(repeats)])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PackedRaggedTensor):
            return False
        return (
            torch.equal(self.values, other.values)
            and torch.equal(self.row_offsets, other.row_offsets)
            and torch.equal(self.cu_seqlens, other.cu_seqlens)
        )

    def __repr__(self) -> str:
        return (
            f"PackedRaggedTensor(batch={len(self)}, rows={self.num_rows}, "
            f"members={self.values.numel()}, dtype={self.values.dtype})"
        )

    @staticmethod
    def cat(batches: Sequence["PackedRaggedTensor"]) -> "PackedRaggedTensor":
        if not batches:
            raise ValueError("cannot cat an empty list of packed ragged batches")
        device = batches[0].device
        return PackedRaggedTensor(
            torch.cat([batch.values for batch in batches], dim=0),
            cu_seqlens_from_lengths(torch.cat([batch.row_lengths for batch in batches]), device=device),
            cu_seqlens_from_lengths(torch.cat([batch.sequence_lengths for batch in batches]), device=device),
        )


def packed_ragged_padding_segments(
    reference: PackedRaggedTensor,
    *,
    segment_lengths: Sequence[int],
    members: Sequence[int],
) -> PackedRaggedTensor:
    """Return padding segments of ``segment_lengths`` rows each, every row holding ``members``.

    An empty ``members`` is the common case: a field whose padding row means "nothing recorded"
    spends only its offset entry, not a row of sentinels.
    """
    row_count = sum(segment_lengths)
    return PackedRaggedTensor(
        torch.tensor(list(members) * row_count, dtype=reference.dtype, device=reference.device),
        cu_seqlens_from_lengths([len(members)] * row_count, device=reference.device),
        cu_seqlens_from_lengths(segment_lengths, device=reference.device),
    )
