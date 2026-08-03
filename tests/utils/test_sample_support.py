import numpy as np
import pytest

from skyrl.utils.sample_support import (
    append_empty_sample_support_rows,
    slice_sample_support_rows,
    validate_sample_support,
)


def test_sample_support_array_operations_preserve_dense_int32_layout():
    first = np.asarray([[7, 8], [9, -1]], dtype=np.int32)

    combined = append_empty_sample_support_rows(first, 2)
    sliced = slice_sample_support_rows(combined, 3)

    assert sliced.dtype == np.int32
    assert sliced.flags.c_contiguous
    np.testing.assert_array_equal(sliced, [[7, 8], [9, -1], [-1, -1]])


def test_sample_support_rejects_non_int32():
    with pytest.raises(ValueError, match="int32"):
        validate_sample_support(np.asarray([[7, 8]], dtype=np.int64))
