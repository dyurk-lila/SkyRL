import numpy as np


def validate_sample_support(values: np.ndarray) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        raise TypeError("sample support must be a NumPy array")
    if values.ndim != 2:
        raise ValueError(f"sample support must have shape [tokens, top_k], got {values.shape!r}")
    if values.dtype != np.int32:
        raise ValueError("sample support must use int32 vocab IDs")
    if not values.flags.c_contiguous:
        raise ValueError("sample support must be contiguous")
    return values


def append_empty_sample_support_rows(values: np.ndarray, count: int) -> np.ndarray:
    values = validate_sample_support(values)
    empty_rows = make_empty_sample_support_rows(values.shape[1], count)
    if count == 0:
        return values
    return np.concatenate((values, empty_rows), axis=0)


def make_empty_sample_support_rows(top_k: int, count: int) -> np.ndarray:
    if top_k < 1:
        raise ValueError("sample support top_k must be positive")
    if count < 0:
        raise ValueError("sample support row count must be non-negative")
    return np.full((count, top_k), -1, dtype=np.int32)


def slice_sample_support_rows(values: np.ndarray, stop: int) -> np.ndarray:
    values = validate_sample_support(values)
    if stop < 0 or stop > values.shape[0]:
        raise ValueError(f"sample support row slice {stop} exceeds {values.shape[0]} rows")
    return values[:stop]
