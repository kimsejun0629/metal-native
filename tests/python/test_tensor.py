"""Tests for tensor module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from metal_native import dtypes
from metal_native.tensor import Tensor


def _make_tensor(shape, dtype, requires_grad=False):
    """Create a Tensor bypassing __init__ (no C extension needed)."""
    t = Tensor.__new__(Tensor)
    t._shape = shape
    t._dtype = dtype
    t._requires_grad = requires_grad
    t._grad = None
    t._handle = None
    return t


def test_tensor_repr():
    """Test Tensor repr format."""
    tensor = _make_tensor((3, 3), dtypes.float32)
    assert repr(tensor) == "Tensor(shape=(3, 3), dtype=float32, device=metal)"


def test_tensor_repr_with_grad():
    """Test Tensor repr with requires_grad=True."""
    tensor = _make_tensor((2, 2), dtypes.float16, requires_grad=True)
    assert repr(tensor) == "Tensor(shape=(2, 2), dtype=float16, device=metal, requires_grad=True)"


def test_tensor_device():
    """Test Tensor device property always returns 'metal'."""
    tensor = _make_tensor((1,), dtypes.float32)
    assert tensor.device == 'metal'


def test_dlpack_device_tuple():
    """Test __dlpack_device__ returns correct tuple."""
    tensor = _make_tensor((1,), dtypes.float32)
    device_tuple = tensor.__dlpack_device__()
    assert device_tuple == (8, 0)  # 8 = kDLMetal, 0 = device index
    assert isinstance(device_tuple[0], int)
    assert isinstance(device_tuple[1], int)


def test_tensor_module_all():
    """Test tensor module __all__ exports."""
    from metal_native import tensor
    expected = [
        'Tensor',
        'from_numpy',
        'from_torch',
        'from_dlpack',
        'zeros',
        'ones',
        'empty',
        'tensor',
        'randn',
        'register_torch_override',
    ]
    assert set(tensor.__all__) == set(expected)


def test_factory_function_signatures():
    """Test that factory functions are defined with correct signatures."""
    from metal_native.tensor import zeros, ones, empty, tensor, randn, from_numpy, from_torch, from_dlpack
    import inspect

    # Check zeros signature
    sig = inspect.signature(zeros)
    assert 'shape' in sig.parameters
    assert 'dtype' in sig.parameters
    assert 'requires_grad' in sig.parameters

    # Check ones signature
    sig = inspect.signature(ones)
    assert 'shape' in sig.parameters
    assert 'dtype' in sig.parameters
    assert 'requires_grad' in sig.parameters

    # Check empty signature
    sig = inspect.signature(empty)
    assert 'shape' in sig.parameters
    assert 'dtype' in sig.parameters
    assert 'requires_grad' in sig.parameters

    # Check tensor signature
    sig = inspect.signature(tensor)
    assert 'data' in sig.parameters
    assert 'dtype' in sig.parameters
    assert 'requires_grad' in sig.parameters

    # Check randn signature
    sig = inspect.signature(randn)
    # randn uses *shape, so we check for VAR_POSITIONAL
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
    assert has_varargs
    assert 'dtype' in sig.parameters
    assert 'requires_grad' in sig.parameters

    # Check from_numpy signature
    sig = inspect.signature(from_numpy)
    assert 'array' in sig.parameters
    assert 'requires_grad' in sig.parameters

    # Check from_torch signature
    sig = inspect.signature(from_torch)
    assert 'tensor' in sig.parameters
    assert 'requires_grad' in sig.parameters

    # Check from_dlpack signature
    sig = inspect.signature(from_dlpack)
    assert 'capsule' in sig.parameters


def test_main_init_all():
    """Test metal_native.__init__.__all__ exports."""
    import metal_native

    expected_in_all = [
        '__version__',
        'Tensor',
        'zeros', 'ones', 'empty', 'tensor', 'randn',
        'from_numpy', 'from_torch', 'from_dlpack',
        'device_name', 'is_available', 'device_properties', 'supports_bfloat16',
        'synchronize', 'empty_cache',
        'memory_allocated', 'max_memory_allocated', 'reset_peak_stats',
        'set_seed', 'manual_seed',
        'float32', 'float16', 'bfloat16',
        'int64', 'int32', 'int16', 'int8',
        'uint8', 'bool_',
        'MetalNativeConfig', 'get_config', 'set_config',
        'interop', 'dlpack_bridge', 'accelerate_plugin',
    ]

    for name in expected_in_all:
        assert name in metal_native.__all__, f"{name} missing from __all__"


def test_tensor_properties():
    """Test Tensor lazy property access."""
    tensor = _make_tensor((2, 3, 4), dtypes.int32, requires_grad=True)

    assert tensor.shape == (2, 3, 4)
    assert tensor.dtype == dtypes.int32
    assert tensor.requires_grad is True
    assert tensor.grad is None
    assert tensor.ndim == 3


def test_register_torch_override():
    """Test register_torch_override decorator."""
    from metal_native.tensor import register_torch_override, _TORCH_FUNCTION_OVERRIDES

    # Create a mock torch function
    mock_torch_func = Mock()

    # Register an override
    @register_torch_override(mock_torch_func)
    def custom_impl(x):
        return x * 2

    # Check it was registered
    assert mock_torch_func in _TORCH_FUNCTION_OVERRIDES
    assert _TORCH_FUNCTION_OVERRIDES[mock_torch_func] is custom_impl
