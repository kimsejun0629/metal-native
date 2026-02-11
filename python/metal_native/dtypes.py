"""Data type constants and conversion utilities.

This module provides dtype constants that map to the C++ MNDType enum
and utilities for converting between metal_native, NumPy, and PyTorch dtypes.
"""

from typing import Any, Optional
import numpy as np

__all__ = [
    'DType',
    'float32', 'float16', 'bfloat16',
    'int64', 'int32', 'int16', 'int8',
    'uint8', 'bool_',
    'dtype_to_numpy',
    'numpy_to_dtype',
]


class DType:
    """Metal Native data type descriptor.

    This class wraps the C++ MNDType enum and provides a Python-friendly
    interface for working with tensor data types.
    """

    def __init__(self, name: str, itemsize: int, is_floating: bool, is_signed: bool):
        self.name = name
        self.itemsize = itemsize
        self.is_floating_point = is_floating
        self.is_signed = is_signed

    def __repr__(self) -> str:
        return f"metal_native.{self.name}"

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, DType):
            return self.name == other.name
        return False

    def __hash__(self) -> int:
        return hash(self.name)


# Define all supported dtypes
float32 = DType('float32', 4, True, True)
float16 = DType('float16', 2, True, True)
bfloat16 = DType('bfloat16', 2, True, True)
int64 = DType('int64', 8, False, True)
int32 = DType('int32', 4, False, True)
int16 = DType('int16', 2, False, True)
int8 = DType('int8', 1, False, True)
uint8 = DType('uint8', 1, False, False)
bool_ = DType('bool', 1, False, False)

# Mapping from dtype name to DType object
_DTYPE_MAP = {
    'float32': float32,
    'float16': float16,
    'bfloat16': bfloat16,
    'int64': int64,
    'int32': int32,
    'int16': int16,
    'int8': int8,
    'uint8': uint8,
    'bool': bool_,
}

# NumPy dtype mapping
_NUMPY_DTYPE_MAP = {
    float32: np.float32,
    float16: np.float16,
    # bfloat16 has no direct NumPy equivalent
    int64: np.int64,
    int32: np.int32,
    int16: np.int16,
    int8: np.int8,
    uint8: np.uint8,
    bool_: np.bool_,
}

_NUMPY_TO_DTYPE_MAP = {
    np.dtype('float32'): float32,
    np.dtype('float16'): float16,
    np.dtype('int64'): int64,
    np.dtype('int32'): int32,
    np.dtype('int16'): int16,
    np.dtype('int8'): int8,
    np.dtype('uint8'): uint8,
    np.dtype('bool'): bool_,
}


def dtype_to_numpy(dtype: DType) -> np.dtype:
    """Convert metal_native dtype to NumPy dtype.

    Args:
        dtype: Metal native data type

    Returns:
        Corresponding NumPy dtype

    Raises:
        ValueError: If the dtype has no NumPy equivalent (e.g., bfloat16)
    """
    if dtype not in _NUMPY_DTYPE_MAP:
        raise ValueError(f"DType {dtype} has no NumPy equivalent")
    return _NUMPY_DTYPE_MAP[dtype]


def numpy_to_dtype(np_dtype: np.dtype) -> DType:
    """Convert NumPy dtype to metal_native dtype.

    Args:
        np_dtype: NumPy data type

    Returns:
        Corresponding metal_native dtype

    Raises:
        ValueError: If the NumPy dtype is not supported
    """
    np_dtype = np.dtype(np_dtype)  # Normalize
    if np_dtype not in _NUMPY_TO_DTYPE_MAP:
        raise ValueError(f"NumPy dtype {np_dtype} is not supported by metal_native")
    return _NUMPY_TO_DTYPE_MAP[np_dtype]


def get_dtype(dtype: Any) -> DType:
    """Parse dtype from various input formats.

    Args:
        dtype: Can be a DType, string name, or NumPy dtype

    Returns:
        Corresponding DType object

    Raises:
        ValueError: If dtype cannot be parsed
    """
    if isinstance(dtype, DType):
        return dtype
    if isinstance(dtype, str):
        if dtype in _DTYPE_MAP:
            return _DTYPE_MAP[dtype]
        raise ValueError(f"Unknown dtype name: {dtype}")
    if hasattr(dtype, 'dtype'):  # NumPy array or dtype
        return numpy_to_dtype(np.dtype(dtype))
    try:
        return numpy_to_dtype(np.dtype(dtype))
    except (TypeError, ValueError):
        raise ValueError(f"Cannot convert {dtype} to metal_native dtype")
