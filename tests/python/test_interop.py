"""Tests for interop and dlpack modules."""

import pytest
import inspect
from unittest.mock import Mock, patch, MagicMock


def test_interop_module_exports():
    """Test interop module __all__ exports."""
    from metal_native import interop
    expected = ['from_torch', 'to_torch']
    assert set(interop.__all__) == set(expected)


def test_dlpack_bridge_module_exports():
    """Test dlpack_bridge module __all__ exports."""
    from metal_native import dlpack_bridge
    expected = ['to_dlpack', 'from_dlpack']
    assert set(dlpack_bridge.__all__) == set(expected)


def test_interop_from_torch_exists():
    """Test from_torch function is defined."""
    from metal_native.interop import from_torch
    assert from_torch is not None
    assert callable(from_torch)


def test_interop_to_torch_exists():
    """Test to_torch function is defined."""
    from metal_native.interop import to_torch
    assert to_torch is not None
    assert callable(to_torch)


def test_dlpack_to_dlpack_exists():
    """Test to_dlpack function is defined."""
    from metal_native.dlpack_bridge import to_dlpack
    assert to_dlpack is not None
    assert callable(to_dlpack)


def test_dlpack_from_dlpack_exists():
    """Test from_dlpack function is defined."""
    from metal_native.dlpack_bridge import from_dlpack
    assert from_dlpack is not None
    assert callable(from_dlpack)


def test_dlpack_get_dlpack_device_exists():
    """Test get_dlpack_device function is defined."""
    from metal_native.dlpack_bridge import get_dlpack_device
    assert get_dlpack_device is not None
    assert callable(get_dlpack_device)


def test_from_torch_signature():
    """Test from_torch function signature."""
    from metal_native.interop import from_torch
    sig = inspect.signature(from_torch)
    params = sig.parameters

    assert 'tensor' in params
    assert 'requires_grad' in params
    assert params['requires_grad'].default is None


def test_to_torch_signature():
    """Test to_torch function signature."""
    from metal_native.interop import to_torch
    sig = inspect.signature(to_torch)
    params = sig.parameters

    assert 'mn_tensor' in params


def test_dlpack_to_dlpack_signature():
    """Test to_dlpack function signature."""
    from metal_native.dlpack_bridge import to_dlpack
    sig = inspect.signature(to_dlpack)
    params = sig.parameters

    assert 'mn_tensor' in params


def test_dlpack_from_dlpack_signature():
    """Test from_dlpack function signature."""
    from metal_native.dlpack_bridge import from_dlpack
    sig = inspect.signature(from_dlpack)
    params = sig.parameters

    assert 'capsule' in params


def test_dlpack_get_dlpack_device_signature():
    """Test get_dlpack_device function signature."""
    from metal_native.dlpack_bridge import get_dlpack_device
    sig = inspect.signature(get_dlpack_device)
    params = sig.parameters

    assert 'mn_tensor' in params


def test_from_torch_raises_importerror_no_torch():
    """Test from_torch raises ImportError when torch is not available."""
    from metal_native.interop import from_torch

    mock_tensor = Mock()

    with patch.dict('sys.modules', {'torch': None, 'torch.utils.dlpack': None}):
        with pytest.raises(ImportError, match="PyTorch is required"):
            from_torch(mock_tensor)


def test_from_torch_importerror_message():
    """Test from_torch ImportError has helpful message."""
    from metal_native.interop import from_torch
    import sys

    mock_tensor = Mock()

    # Mock torch not being importable
    with patch.dict(sys.modules, {'torch': None, 'torch.utils.dlpack': None}):
        with pytest.raises(ImportError) as exc_info:
            from_torch(mock_tensor)

        error_msg = str(exc_info.value)
        assert "PyTorch is required" in error_msg
        assert "pip install torch" in error_msg


def test_to_torch_raises_importerror_no_torch():
    """Test to_torch raises ImportError when torch is not available."""
    from metal_native.interop import to_torch
    import sys

    mock_tensor = Mock()

    # Mock torch not being importable
    with patch.dict(sys.modules, {'torch': None, 'torch.utils.dlpack': None}):
        with pytest.raises(ImportError, match="PyTorch is required"):
            to_torch(mock_tensor)


def test_to_torch_importerror_message():
    """Test to_torch ImportError has helpful message."""
    from metal_native.interop import to_torch
    import sys

    mock_tensor = Mock()

    # Mock torch not being importable
    with patch.dict(sys.modules, {'torch': None, 'torch.utils.dlpack': None}):
        with pytest.raises(ImportError) as exc_info:
            to_torch(mock_tensor)

        error_msg = str(exc_info.value)
        assert "PyTorch is required" in error_msg
        assert "pip install torch" in error_msg


def test_dlpack_to_dlpack_type_check():
    """Test to_dlpack raises TypeError for non-Tensor input."""
    from metal_native.dlpack_bridge import to_dlpack

    with pytest.raises(TypeError, match="Expected MetalNative Tensor"):
        to_dlpack("not a tensor")

    with pytest.raises(TypeError, match="Expected MetalNative Tensor"):
        to_dlpack(42)


def test_dlpack_get_dlpack_device_type_check():
    """Test get_dlpack_device raises TypeError for non-Tensor input."""
    from metal_native.dlpack_bridge import get_dlpack_device

    with pytest.raises(TypeError, match="Expected MetalNative Tensor"):
        get_dlpack_device("not a tensor")


def test_to_torch_type_check_docstring():
    """Test to_torch documents TypeError for invalid input."""
    from metal_native.interop import to_torch

    # Check docstring mentions TypeError
    assert to_torch.__doc__ is not None
    assert 'TypeError' in to_torch.__doc__ or 'metal_native' in to_torch.__doc__.lower()


def test_dlpack_device_constants():
    """Test DLPack device type constant is documented."""
    # Device type 8 is Metal according to DLPack spec
    from metal_native.dlpack_bridge import get_dlpack_device

    # Check docstring mentions device type 8 for Metal
    assert get_dlpack_device.__doc__ is not None
    assert 'device_type' in get_dlpack_device.__doc__
    assert '8' in get_dlpack_device.__doc__


def test_accelerate_plugin_module_exists():
    """Test accelerate_plugin module can be imported."""
    from metal_native import accelerate_plugin
    assert accelerate_plugin is not None


def test_accelerate_plugin_exports():
    """Test accelerate_plugin module __all__ exports."""
    from metal_native import accelerate_plugin
    expected = ['MetalNativeAccelerator', 'is_available']
    assert set(accelerate_plugin.__all__) == set(expected)


def test_accelerate_is_available_exists():
    """Test is_available function exists in accelerate_plugin."""
    from metal_native.accelerate_plugin import is_available
    assert is_available is not None
    assert callable(is_available)


def test_accelerate_is_available_returns_bool():
    """Test is_available returns bool."""
    from metal_native.accelerate_plugin import is_available
    result = is_available()
    assert isinstance(result, bool)


def test_metal_native_accelerator_exists():
    """Test MetalNativeAccelerator class exists."""
    from metal_native.accelerate_plugin import MetalNativeAccelerator
    assert MetalNativeAccelerator is not None
    assert callable(MetalNativeAccelerator)


def test_metal_native_accelerator_has_methods():
    """Test MetalNativeAccelerator has expected methods."""
    from metal_native.accelerate_plugin import MetalNativeAccelerator

    # Check class methods exist
    assert hasattr(MetalNativeAccelerator, 'is_available')
    assert hasattr(MetalNativeAccelerator, '__init__')

    # Check instance methods are defined in class
    method_names = ['prepare_model', 'prepare_optimizer', 'synchronize', 'empty_cache', 'get_memory_info']
    for method_name in method_names:
        assert hasattr(MetalNativeAccelerator, method_name)


def test_interop_docstrings():
    """Test interop functions have docstrings."""
    from metal_native.interop import from_torch, to_torch

    assert from_torch.__doc__ is not None
    assert to_torch.__doc__ is not None
    assert 'PyTorch' in from_torch.__doc__
    assert 'PyTorch' in to_torch.__doc__


def test_dlpack_bridge_docstrings():
    """Test dlpack_bridge functions have docstrings."""
    from metal_native.dlpack_bridge import to_dlpack, from_dlpack

    assert to_dlpack.__doc__ is not None
    assert from_dlpack.__doc__ is not None
    assert 'DLPack' in to_dlpack.__doc__
    assert 'DLPack' in from_dlpack.__doc__


def test_interop_zero_copy_mentioned():
    """Test interop docstrings mention zero-copy."""
    from metal_native.interop import from_torch, to_torch

    # Check documentation mentions zero-copy capability
    assert 'zero-copy' in from_torch.__doc__ or 'zero copy' in from_torch.__doc__.lower()
    assert 'zero-copy' in to_torch.__doc__ or 'zero copy' in to_torch.__doc__.lower()


def test_dlpack_protocol_mentioned():
    """Test dlpack_bridge mentions DLPack protocol."""
    from metal_native import dlpack_bridge

    # Check module docstring
    assert dlpack_bridge.__doc__ is not None
    assert 'DLPack' in dlpack_bridge.__doc__
    assert 'protocol' in dlpack_bridge.__doc__.lower()
