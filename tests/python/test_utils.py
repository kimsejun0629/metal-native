"""Tests for utils module."""

import sys
import pytest
from unittest.mock import Mock, patch, MagicMock


def test_utils_module_all():
    """Test utils module __all__ exports."""
    import metal_native.utils as utils
    expected = [
        'synchronize',
        'empty_cache',
        'memory_allocated',
        'max_memory_allocated',
        'reset_peak_stats',
        'set_seed',
        'manual_seed',
    ]
    assert set(utils.__all__) == set(expected)


def test_synchronize_signature():
    """Test synchronize function signature."""
    from metal_native.utils import synchronize
    import inspect

    sig = inspect.signature(synchronize)
    assert len(sig.parameters) == 0  # No parameters
    # Returns None
    assert sig.return_annotation is None or sig.return_annotation == type(None) or 'None' in str(sig.return_annotation)


def test_synchronize_with_mock():
    """Test synchronize calls C extension."""
    mock_c = Mock()
    mock_c.synchronize.return_value = None
    with patch('metal_native._C', mock_c):
        from metal_native.utils import synchronize
        synchronize()
        mock_c.synchronize.assert_called_once()


def test_synchronize_raises_when_c_unavailable():
    """Test synchronize raises RuntimeError when C extension unavailable."""
    import metal_native
    with patch.object(metal_native, '_C', None):
        if 'metal_native.utils' in sys.modules:
            del sys.modules['metal_native.utils']
        from metal_native.utils import synchronize
        with pytest.raises(RuntimeError, match="Cannot synchronize"):
            synchronize()


def test_empty_cache_signature():
    """Test empty_cache function signature."""
    from metal_native.utils import empty_cache
    import inspect

    sig = inspect.signature(empty_cache)
    assert len(sig.parameters) == 0  # No parameters


def test_empty_cache_with_mock():
    """Test empty_cache calls C extension."""
    mock_c = Mock()
    mock_c.empty_cache.return_value = None
    with patch('metal_native._C', mock_c):
        from metal_native.utils import empty_cache
        empty_cache()
        mock_c.empty_cache.assert_called_once()


def test_empty_cache_silent_when_c_unavailable():
    """Test empty_cache fails silently when C extension unavailable."""
    import metal_native
    with patch.object(metal_native, '_C', None):
        if 'metal_native.utils' in sys.modules:
            del sys.modules['metal_native.utils']
        from metal_native.utils import empty_cache
        # Should not raise
        empty_cache()


def test_memory_allocated_signature():
    """Test memory_allocated function signature."""
    from metal_native.utils import memory_allocated
    import inspect

    sig = inspect.signature(memory_allocated)
    assert len(sig.parameters) == 0  # No parameters
    # Returns int
    assert sig.return_annotation == int or 'int' in str(sig.return_annotation)


def test_memory_allocated_with_mock():
    """Test memory_allocated returns int from C extension."""
    mock_c = Mock()
    mock_c.memory_allocated.return_value = 1048576  # 1 MB
    with patch('metal_native._C', mock_c):
        from metal_native.utils import memory_allocated
        result = memory_allocated()
        assert isinstance(result, int)
        assert result == 1048576


def test_memory_allocated_raises_when_c_unavailable():
    """Test memory_allocated raises RuntimeError when C extension unavailable."""
    import metal_native
    with patch.object(metal_native, '_C', None):
        if 'metal_native.utils' in sys.modules:
            del sys.modules['metal_native.utils']
        from metal_native.utils import memory_allocated
        with pytest.raises(RuntimeError, match="Cannot query memory"):
            memory_allocated()


def test_max_memory_allocated_signature():
    """Test max_memory_allocated function signature."""
    from metal_native.utils import max_memory_allocated
    import inspect

    sig = inspect.signature(max_memory_allocated)
    assert len(sig.parameters) == 0  # No parameters
    # Returns int
    assert sig.return_annotation == int or 'int' in str(sig.return_annotation)


def test_max_memory_allocated_with_mock():
    """Test max_memory_allocated returns int from C extension."""
    mock_c = Mock()
    mock_c.max_memory_allocated.return_value = 2097152  # 2 MB
    with patch('metal_native._C', mock_c):
        from metal_native.utils import max_memory_allocated
        result = max_memory_allocated()
        assert isinstance(result, int)
        assert result == 2097152


def test_max_memory_allocated_raises_when_c_unavailable():
    """Test max_memory_allocated raises RuntimeError when C extension unavailable."""
    import metal_native
    with patch.object(metal_native, '_C', None):
        if 'metal_native.utils' in sys.modules:
            del sys.modules['metal_native.utils']
        from metal_native.utils import max_memory_allocated
        with pytest.raises(RuntimeError, match="Cannot query memory"):
            max_memory_allocated()


def test_reset_peak_stats_signature():
    """Test reset_peak_stats function signature."""
    from metal_native.utils import reset_peak_stats
    import inspect

    sig = inspect.signature(reset_peak_stats)
    assert len(sig.parameters) == 0  # No parameters


def test_reset_peak_stats_with_mock():
    """Test reset_peak_stats calls C extension."""
    mock_c = Mock()
    mock_c.reset_peak_stats.return_value = None
    with patch('metal_native._C', mock_c):
        from metal_native.utils import reset_peak_stats
        reset_peak_stats()
        mock_c.reset_peak_stats.assert_called_once()


def test_reset_peak_stats_silent_when_c_unavailable():
    """Test reset_peak_stats fails silently when C extension unavailable."""
    import metal_native
    with patch.object(metal_native, '_C', None):
        if 'metal_native.utils' in sys.modules:
            del sys.modules['metal_native.utils']
        from metal_native.utils import reset_peak_stats
        # Should not raise
        reset_peak_stats()


def test_set_seed_signature():
    """Test set_seed function signature."""
    from metal_native.utils import set_seed
    import inspect

    sig = inspect.signature(set_seed)
    assert 'seed' in sig.parameters
    assert sig.parameters['seed'].annotation == int or 'int' in str(sig.parameters['seed'].annotation)


def test_set_seed_with_mock():
    """Test set_seed calls C extension with seed value."""
    mock_c = Mock()
    mock_c.set_seed.return_value = None
    with patch('metal_native._C', mock_c):
        from metal_native.utils import set_seed
        set_seed(42)
        mock_c.set_seed.assert_called_once_with(42)


def test_set_seed_negative_raises():
    """Test set_seed raises ValueError for negative seed."""
    import metal_native.utils
    with pytest.raises(ValueError, match="Seed must be non-negative"):
        metal_native.utils.set_seed(-1)


def test_set_seed_raises_when_c_unavailable():
    """Test set_seed raises RuntimeError when C extension unavailable."""
    import metal_native
    with patch.object(metal_native, '_C', None):
        if 'metal_native.utils' in sys.modules:
            del sys.modules['metal_native.utils']
        from metal_native.utils import set_seed
        with pytest.raises(RuntimeError, match="Cannot set seed"):
            set_seed(42)


def test_manual_seed_signature():
    """Test manual_seed function signature."""
    from metal_native.utils import manual_seed
    import inspect

    sig = inspect.signature(manual_seed)
    assert 'seed' in sig.parameters
    assert sig.parameters['seed'].annotation == int or 'int' in str(sig.parameters['seed'].annotation)


def test_manual_seed_aliases_set_seed():
    """Test manual_seed is an alias for set_seed."""
    mock_c = Mock()
    mock_c.set_seed.return_value = None
    with patch('metal_native._C', mock_c):
        from metal_native.utils import manual_seed
        manual_seed(123)
        mock_c.set_seed.assert_called_once_with(123)


def test_manual_seed_negative_raises():
    """Test manual_seed raises ValueError for negative seed (via set_seed)."""
    import metal_native.utils
    with pytest.raises(ValueError, match="Seed must be non-negative"):
        metal_native.utils.manual_seed(-5)


def test_get_memory_info_with_mock():
    """Test get_memory_info returns formatted string."""
    mock_c = Mock()
    mock_c.memory_allocated.return_value = 1048576  # 1 MB
    mock_c.max_memory_allocated.return_value = 2097152  # 2 MB
    with patch('metal_native._C', mock_c):
        from metal_native.utils import get_memory_info
        result = get_memory_info()
        assert isinstance(result, str)
        assert 'GPU Memory:' in result
        assert '1.00 MB' in result  # current
        assert '2.00 MB' in result  # peak


def test_get_memory_info_when_c_unavailable():
    """Test get_memory_info returns unavailable message when C extension unavailable."""
    import metal_native
    with patch.object(metal_native, '_C', None):
        if 'metal_native.utils' in sys.modules:
            del sys.modules['metal_native.utils']
        from metal_native.utils import get_memory_info
        result = get_memory_info()
        assert isinstance(result, str)
        assert result == "GPU Memory: Not available"


def test_synchronize_context_class_exists():
    """Test SynchronizeContext class exists and is importable."""
    from metal_native.utils import SynchronizeContext
    assert SynchronizeContext is not None


def test_synchronize_context_as_context_manager():
    """Test SynchronizeContext works as context manager and calls synchronize on exit."""
    mock_c = Mock()
    mock_c.synchronize.return_value = None
    with patch('metal_native._C', mock_c):
        from metal_native.utils import SynchronizeContext

        with SynchronizeContext() as ctx:
            assert ctx is not None

        # synchronize should be called on exit
        mock_c.synchronize.assert_called_once()


def test_synchronize_context_enter_returns_self():
    """Test SynchronizeContext __enter__ returns self."""
    from metal_native.utils import SynchronizeContext
    ctx = SynchronizeContext()
    result = ctx.__enter__()
    assert result is ctx
