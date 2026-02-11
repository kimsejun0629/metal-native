"""Tests for device module."""

import sys
import pytest
from unittest.mock import Mock, patch, MagicMock


def test_device_module_all():
    """Test device module __all__ exports."""
    import metal_native.device as device
    expected = [
        'device_name',
        'is_available',
        'device_properties',
        'supports_bfloat16',
        'get_device_info',
    ]
    assert set(device.__all__) == set(expected)


def test_is_available_returns_bool_when_c_available():
    """Test is_available returns bool when C extension is available."""
    mock_c = Mock()
    mock_c.is_available.return_value = True

    import metal_native
    with patch.object(metal_native, '_C', mock_c):
        # Force reimport to pick up the mocked _C
        if 'metal_native.device' in sys.modules:
            del sys.modules['metal_native.device']
        from metal_native.device import is_available
        result = is_available()
        assert isinstance(result, bool)
        assert result is True


def test_is_available_returns_false_when_c_unavailable():
    """Test is_available returns False when C extension is not available."""
    import metal_native

    with patch.object(metal_native, '_C', None):
        # Force reimport
        if 'metal_native.device' in sys.modules:
            del sys.modules['metal_native.device']
        from metal_native.device import is_available
        result = is_available()
        assert isinstance(result, bool)
        assert result is False


def test_device_name_signature():
    """Test device_name function signature."""
    from metal_native.device import device_name
    import inspect

    sig = inspect.signature(device_name)
    assert len(sig.parameters) == 0  # No parameters
    # Returns str
    assert sig.return_annotation == str or 'str' in str(sig.return_annotation)


def test_device_name_with_mock():
    """Test device_name returns string from C extension."""
    mock_c = Mock()
    mock_c.device_name.return_value = "Apple M1 Max"

    import metal_native
    with patch.object(metal_native, '_C', mock_c):
        if 'metal_native.device' in sys.modules:
            del sys.modules['metal_native.device']
        from metal_native.device import device_name
        result = device_name()
        assert isinstance(result, str)
        assert result == "Apple M1 Max"


def test_device_name_raises_when_c_unavailable():
    """Test device_name raises RuntimeError when C extension unavailable."""
    import metal_native

    with patch.object(metal_native, '_C', None):
        if 'metal_native.device' in sys.modules:
            del sys.modules['metal_native.device']
        from metal_native.device import device_name
        with pytest.raises(RuntimeError, match="Metal device not available"):
            device_name()


def test_device_properties_signature():
    """Test device_properties function signature."""
    from metal_native.device import device_properties
    import inspect

    sig = inspect.signature(device_properties)
    assert len(sig.parameters) == 0  # No parameters


def test_device_properties_with_mock():
    """Test device_properties returns dict from C extension."""
    mock_props = {
        'name': 'Apple M1 Max',
        'cores': 32,
        'memory': 68719476736,
        'bandwidth': 400.0,
        'max_buffer_length': 1073741824,
        'supports_bfloat16': True,
        'unified_memory': True,
        'recommended_working_set': 34359738368,
    }
    mock_c = Mock()
    mock_c.device_properties.return_value = mock_props

    import metal_native
    with patch.object(metal_native, '_C', mock_c):
        if 'metal_native.device' in sys.modules:
            del sys.modules['metal_native.device']
        from metal_native.device import device_properties
        result = device_properties()
        assert isinstance(result, dict)
        assert result == mock_props


def test_device_properties_raises_when_c_unavailable():
    """Test device_properties raises RuntimeError when C extension unavailable."""
    import metal_native

    with patch.object(metal_native, '_C', None):
        if 'metal_native.device' in sys.modules:
            del sys.modules['metal_native.device']
        from metal_native.device import device_properties
        with pytest.raises(RuntimeError, match="Metal device not available"):
            device_properties()


def test_supports_bfloat16_signature():
    """Test supports_bfloat16 function signature."""
    from metal_native.device import supports_bfloat16
    import inspect

    sig = inspect.signature(supports_bfloat16)
    assert len(sig.parameters) == 0  # No parameters
    # Returns bool
    assert sig.return_annotation == bool or 'bool' in str(sig.return_annotation)


def test_supports_bfloat16_with_mock():
    """Test supports_bfloat16 returns bool from C extension."""
    mock_c = Mock()
    mock_c.supports_bfloat16.return_value = True

    import metal_native
    with patch.object(metal_native, '_C', mock_c):
        if 'metal_native.device' in sys.modules:
            del sys.modules['metal_native.device']
        from metal_native.device import supports_bfloat16
        result = supports_bfloat16()
        assert isinstance(result, bool)
        assert result is True


def test_supports_bfloat16_raises_when_c_unavailable():
    """Test supports_bfloat16 raises RuntimeError when C extension unavailable."""
    import metal_native

    with patch.object(metal_native, '_C', None):
        if 'metal_native.device' in sys.modules:
            del sys.modules['metal_native.device']
        from metal_native.device import supports_bfloat16
        with pytest.raises(RuntimeError, match="Metal device not available"):
            supports_bfloat16()


def test_get_device_info_when_available():
    """Test get_device_info returns formatted string when Metal is available."""
    mock_props = {
        'name': 'Apple M1 Max',
        'cores': 32,
        'memory': 68719476736,  # 64 GB
        'bandwidth': 400.0,
        'unified_memory': True,
        'supports_bfloat16': True,
    }
    mock_c = Mock()
    mock_c.is_available.return_value = True
    mock_c.device_properties.return_value = mock_props

    import metal_native
    with patch.object(metal_native, '_C', mock_c):
        if 'metal_native.device' in sys.modules:
            del sys.modules['metal_native.device']
        from metal_native.device import get_device_info
        result = get_device_info()
        assert isinstance(result, str)
        assert 'Apple M1 Max' in result
        assert '32' in result  # cores
        assert '64.00 GB' in result  # memory formatted
        assert '400.00 GB/s' in result  # bandwidth


def test_get_device_info_when_unavailable():
    """Test get_device_info returns unavailable message when Metal is not available."""
    import metal_native

    with patch.object(metal_native, '_C', None):
        if 'metal_native.device' in sys.modules:
            del sys.modules['metal_native.device']
        from metal_native.device import get_device_info
        result = get_device_info()
        assert isinstance(result, str)
        assert result == "Metal: Not available"


def test_device_class_exists():
    """Test Device class exists and is importable."""
    from metal_native.device import Device
    assert Device is not None


def test_device_class_init_default():
    """Test Device class initialization with default parameter."""
    from metal_native.device import Device
    dev = Device()
    assert dev.device == 'metal'


def test_device_class_init_metal():
    """Test Device class initialization with 'metal' parameter."""
    from metal_native.device import Device
    dev = Device('metal')
    assert dev.device == 'metal'


def test_device_class_init_none():
    """Test Device class initialization with None parameter."""
    from metal_native.device import Device
    dev = Device(None)
    assert dev.device == 'metal'


def test_device_class_init_invalid_raises():
    """Test Device class raises ValueError for invalid device."""
    from metal_native.device import Device
    with pytest.raises(ValueError, match="Invalid device"):
        Device('cuda')


def test_device_class_context_manager():
    """Test Device class works as context manager."""
    from metal_native.device import Device
    dev = Device()
    with dev as d:
        assert d is dev
        assert d.device == 'metal'


def test_device_class_repr():
    """Test Device class __repr__ method."""
    from metal_native.device import Device
    dev = Device()
    assert repr(dev) == "Device('metal')"
