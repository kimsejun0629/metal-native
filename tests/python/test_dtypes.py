"""Tests for dtypes module."""

import pytest
import numpy as np
from metal_native import dtypes


def test_dtype_objects_exist():
    """Test that all DType objects are defined."""
    assert hasattr(dtypes, 'float32')
    assert hasattr(dtypes, 'float16')
    assert hasattr(dtypes, 'bfloat16')
    assert hasattr(dtypes, 'int64')
    assert hasattr(dtypes, 'int32')
    assert hasattr(dtypes, 'int16')
    assert hasattr(dtypes, 'int8')
    assert hasattr(dtypes, 'uint8')
    assert hasattr(dtypes, 'bool_')


def test_dtype_instances():
    """Test that all dtypes are DType instances."""
    assert isinstance(dtypes.float32, dtypes.DType)
    assert isinstance(dtypes.float16, dtypes.DType)
    assert isinstance(dtypes.bfloat16, dtypes.DType)
    assert isinstance(dtypes.int64, dtypes.DType)
    assert isinstance(dtypes.int32, dtypes.DType)
    assert isinstance(dtypes.int16, dtypes.DType)
    assert isinstance(dtypes.int8, dtypes.DType)
    assert isinstance(dtypes.uint8, dtypes.DType)
    assert isinstance(dtypes.bool_, dtypes.DType)


def test_dtype_names():
    """Test DType name property."""
    assert dtypes.float32.name == 'float32'
    assert dtypes.float16.name == 'float16'
    assert dtypes.bfloat16.name == 'bfloat16'
    assert dtypes.int64.name == 'int64'
    assert dtypes.int32.name == 'int32'
    assert dtypes.int16.name == 'int16'
    assert dtypes.int8.name == 'int8'
    assert dtypes.uint8.name == 'uint8'
    assert dtypes.bool_.name == 'bool'


def test_dtype_itemsize():
    """Test DType itemsize property."""
    assert dtypes.float32.itemsize == 4
    assert dtypes.float16.itemsize == 2
    assert dtypes.bfloat16.itemsize == 2
    assert dtypes.int64.itemsize == 8
    assert dtypes.int32.itemsize == 4
    assert dtypes.int16.itemsize == 2
    assert dtypes.int8.itemsize == 1
    assert dtypes.uint8.itemsize == 1
    assert dtypes.bool_.itemsize == 1


def test_dtype_is_floating_point():
    """Test DType is_floating_point property."""
    assert dtypes.float32.is_floating_point is True
    assert dtypes.float16.is_floating_point is True
    assert dtypes.bfloat16.is_floating_point is True
    assert dtypes.int64.is_floating_point is False
    assert dtypes.int32.is_floating_point is False
    assert dtypes.int16.is_floating_point is False
    assert dtypes.int8.is_floating_point is False
    assert dtypes.uint8.is_floating_point is False
    assert dtypes.bool_.is_floating_point is False


def test_dtype_is_signed():
    """Test DType is_signed property."""
    assert dtypes.float32.is_signed is True
    assert dtypes.float16.is_signed is True
    assert dtypes.bfloat16.is_signed is True
    assert dtypes.int64.is_signed is True
    assert dtypes.int32.is_signed is True
    assert dtypes.int16.is_signed is True
    assert dtypes.int8.is_signed is True
    assert dtypes.uint8.is_signed is False
    assert dtypes.bool_.is_signed is False


def test_dtype_equality():
    """Test DType equality comparison."""
    assert dtypes.float32 == dtypes.float32
    assert dtypes.float32 != dtypes.float16
    assert dtypes.int32 == dtypes.int32
    assert dtypes.int32 != dtypes.int64
    assert dtypes.float32 != "float32"
    assert dtypes.float32 != 32


def test_dtype_hashing():
    """Test DType can be hashed and used in sets/dicts."""
    dtype_set = {dtypes.float32, dtypes.float16, dtypes.float32}
    assert len(dtype_set) == 2

    dtype_dict = {dtypes.float32: 'f32', dtypes.float16: 'f16'}
    assert dtype_dict[dtypes.float32] == 'f32'
    assert dtype_dict[dtypes.float16] == 'f16'


def test_dtype_repr():
    """Test DType repr format."""
    assert repr(dtypes.float32) == "metal_native.float32"
    assert repr(dtypes.int64) == "metal_native.int64"
    assert repr(dtypes.bool_) == "metal_native.bool"


def test_dtype_str():
    """Test DType str format."""
    assert str(dtypes.float32) == "float32"
    assert str(dtypes.int64) == "int64"
    assert str(dtypes.bool_) == "bool"


def test_dtype_to_numpy():
    """Test dtype_to_numpy conversion."""
    assert dtypes.dtype_to_numpy(dtypes.float32) == np.float32
    assert dtypes.dtype_to_numpy(dtypes.float16) == np.float16
    assert dtypes.dtype_to_numpy(dtypes.int64) == np.int64
    assert dtypes.dtype_to_numpy(dtypes.int32) == np.int32
    assert dtypes.dtype_to_numpy(dtypes.int16) == np.int16
    assert dtypes.dtype_to_numpy(dtypes.int8) == np.int8
    assert dtypes.dtype_to_numpy(dtypes.uint8) == np.uint8
    assert dtypes.dtype_to_numpy(dtypes.bool_) == np.bool_


def test_dtype_to_numpy_bfloat16_raises():
    """Test that bfloat16 has no numpy equivalent."""
    with pytest.raises(ValueError, match="has no NumPy equivalent"):
        dtypes.dtype_to_numpy(dtypes.bfloat16)


def test_numpy_to_dtype():
    """Test numpy_to_dtype conversion."""
    assert dtypes.numpy_to_dtype(np.float32) == dtypes.float32
    assert dtypes.numpy_to_dtype(np.float16) == dtypes.float16
    assert dtypes.numpy_to_dtype(np.int64) == dtypes.int64
    assert dtypes.numpy_to_dtype(np.int32) == dtypes.int32
    assert dtypes.numpy_to_dtype(np.int16) == dtypes.int16
    assert dtypes.numpy_to_dtype(np.int8) == dtypes.int8
    assert dtypes.numpy_to_dtype(np.uint8) == dtypes.uint8
    assert dtypes.numpy_to_dtype(np.bool_) == dtypes.bool_


def test_numpy_to_dtype_from_string():
    """Test numpy_to_dtype with string input."""
    assert dtypes.numpy_to_dtype('float32') == dtypes.float32
    assert dtypes.numpy_to_dtype('int64') == dtypes.int64


def test_numpy_to_dtype_unsupported():
    """Test numpy_to_dtype with unsupported dtype."""
    with pytest.raises(ValueError, match="not supported"):
        dtypes.numpy_to_dtype(np.float64)


def test_get_dtype_from_dtype():
    """Test get_dtype with DType input."""
    assert dtypes.get_dtype(dtypes.float32) == dtypes.float32
    assert dtypes.get_dtype(dtypes.int64) == dtypes.int64


def test_get_dtype_from_string():
    """Test get_dtype with string input."""
    assert dtypes.get_dtype('float32') == dtypes.float32
    assert dtypes.get_dtype('float16') == dtypes.float16
    assert dtypes.get_dtype('bfloat16') == dtypes.bfloat16
    assert dtypes.get_dtype('int64') == dtypes.int64
    assert dtypes.get_dtype('int32') == dtypes.int32
    assert dtypes.get_dtype('int16') == dtypes.int16
    assert dtypes.get_dtype('int8') == dtypes.int8
    assert dtypes.get_dtype('uint8') == dtypes.uint8
    assert dtypes.get_dtype('bool') == dtypes.bool_


def test_get_dtype_from_numpy():
    """Test get_dtype with numpy dtype input."""
    assert dtypes.get_dtype(np.float32) == dtypes.float32
    assert dtypes.get_dtype(np.dtype('int64')) == dtypes.int64


def test_get_dtype_invalid_string():
    """Test get_dtype with invalid string."""
    with pytest.raises(ValueError, match="Unknown dtype name"):
        dtypes.get_dtype('invalid_dtype')


def test_get_dtype_invalid_type():
    """Test get_dtype with invalid type."""
    with pytest.raises(ValueError, match="Cannot convert"):
        dtypes.get_dtype(123)


def test_module_all_exports():
    """Test that __all__ contains expected exports."""
    expected = [
        'DType',
        'float32', 'float16', 'bfloat16',
        'int64', 'int32', 'int16', 'int8',
        'uint8', 'bool_',
        'dtype_to_numpy',
        'numpy_to_dtype',
    ]
    assert set(dtypes.__all__) == set(expected)
