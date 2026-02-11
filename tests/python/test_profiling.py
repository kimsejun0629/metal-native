"""Tests for profiling module."""

import pytest
import inspect
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager


def test_profiling_module_exports():
    """Test profiling module __all__ exports."""
    import metal_native.profiling as profiling
    expected = [
        'begin_trace',
        'end_trace',
        'gpu_counters',
        'memory_snapshot',
        'memory_report',
        'TraceContext',
    ]
    assert set(profiling.__all__) == set(expected)


def test_begin_trace_exists():
    """Test begin_trace function is defined."""
    from metal_native.profiling import begin_trace
    assert begin_trace is not None
    assert callable(begin_trace)


def test_end_trace_exists():
    """Test end_trace function is defined."""
    from metal_native.profiling import end_trace
    assert end_trace is not None
    assert callable(end_trace)


def test_gpu_counters_exists():
    """Test gpu_counters function is defined."""
    from metal_native.profiling import gpu_counters
    assert gpu_counters is not None
    assert callable(gpu_counters)


def test_memory_snapshot_exists():
    """Test memory_snapshot function is defined."""
    from metal_native.profiling import memory_snapshot
    assert memory_snapshot is not None
    assert callable(memory_snapshot)


def test_memory_report_exists():
    """Test memory_report function is defined."""
    from metal_native.profiling import memory_report
    assert memory_report is not None
    assert callable(memory_report)


def test_trace_context_exists():
    """Test TraceContext is defined."""
    from metal_native.profiling import TraceContext
    assert TraceContext is not None


def test_begin_trace_signature():
    """Test begin_trace function signature."""
    from metal_native.profiling import begin_trace
    sig = inspect.signature(begin_trace)
    params = sig.parameters

    assert 'path' in params
    assert params['path'].default is None


def test_end_trace_signature():
    """Test end_trace function signature."""
    from metal_native.profiling import end_trace
    sig = inspect.signature(end_trace)
    params = sig.parameters

    # end_trace takes no parameters
    assert len(params) == 0


def test_gpu_counters_signature():
    """Test gpu_counters function signature."""
    from metal_native.profiling import gpu_counters
    sig = inspect.signature(gpu_counters)
    params = sig.parameters

    # gpu_counters takes no parameters
    assert len(params) == 0


def test_memory_snapshot_signature():
    """Test memory_snapshot function signature."""
    from metal_native.profiling import memory_snapshot
    sig = inspect.signature(memory_snapshot)
    params = sig.parameters

    # memory_snapshot takes no parameters
    assert len(params) == 0


def test_memory_report_signature():
    """Test memory_report function signature."""
    from metal_native.profiling import memory_report
    sig = inspect.signature(memory_report)
    params = sig.parameters

    assert 'output_path' in params
    assert params['output_path'].default is None


def test_trace_context_signature():
    """Test TraceContext signature."""
    from metal_native.profiling import TraceContext

    # TraceContext is a context manager function
    assert callable(TraceContext)

    sig = inspect.signature(TraceContext)
    params = sig.parameters
    assert 'path' in params
    assert params['path'].default is None


def test_trace_context_is_context_manager():
    """Test TraceContext is a context manager."""
    from metal_native.profiling import TraceContext
    from contextlib import AbstractContextManager

    # TraceContext should be callable and return a context manager
    assert callable(TraceContext)

    # Create a mock to avoid needing the C extension
    with patch('metal_native._C') as mock_c:
        mock_c.profiling_begin_capture.return_value = None
        mock_c.profiling_end_capture.return_value = None

        # Should return a context manager when called
        ctx = TraceContext()
        assert hasattr(ctx, '__enter__')
        assert hasattr(ctx, '__exit__')


def test_begin_trace_docstring():
    """Test begin_trace has descriptive docstring."""
    from metal_native.profiling import begin_trace

    assert begin_trace.__doc__ is not None
    assert 'GPU' in begin_trace.__doc__ or 'trace' in begin_trace.__doc__.lower()
    assert 'capture' in begin_trace.__doc__.lower()


def test_end_trace_docstring():
    """Test end_trace has descriptive docstring."""
    from metal_native.profiling import end_trace

    assert end_trace.__doc__ is not None
    assert 'trace' in end_trace.__doc__.lower() or 'capture' in end_trace.__doc__.lower()


def test_gpu_counters_docstring():
    """Test gpu_counters has descriptive docstring."""
    from metal_native.profiling import gpu_counters

    assert gpu_counters.__doc__ is not None
    assert 'GPU' in gpu_counters.__doc__ or 'performance' in gpu_counters.__doc__.lower()
    assert 'counter' in gpu_counters.__doc__.lower()


def test_gpu_counters_return_type_documented():
    """Test gpu_counters documents return type."""
    from metal_native.profiling import gpu_counters

    # Check docstring mentions dictionary and expected fields
    assert 'dict' in gpu_counters.__doc__.lower() or 'Dictionary' in gpu_counters.__doc__
    assert 'gpu_utilization' in gpu_counters.__doc__


def test_memory_snapshot_docstring():
    """Test memory_snapshot has descriptive docstring."""
    from metal_native.profiling import memory_snapshot

    assert memory_snapshot.__doc__ is not None
    assert 'memory' in memory_snapshot.__doc__.lower()
    assert 'snapshot' in memory_snapshot.__doc__.lower()


def test_memory_snapshot_return_type_documented():
    """Test memory_snapshot documents return type."""
    from metal_native.profiling import memory_snapshot

    # Check docstring mentions dictionary and expected fields
    assert 'dict' in memory_snapshot.__doc__.lower() or 'Dictionary' in memory_snapshot.__doc__
    assert 'total_allocated' in memory_snapshot.__doc__


def test_memory_report_docstring():
    """Test memory_report has descriptive docstring."""
    from metal_native.profiling import memory_report

    assert memory_report.__doc__ is not None
    assert 'memory' in memory_report.__doc__.lower()
    assert 'report' in memory_report.__doc__.lower()


def test_trace_context_docstring():
    """Test TraceContext has descriptive docstring."""
    from metal_native.profiling import TraceContext

    assert TraceContext.__doc__ is not None
    assert 'context manager' in TraceContext.__doc__.lower()
    assert 'trace' in TraceContext.__doc__.lower()


def test_trace_context_usage_example():
    """Test TraceContext docstring includes usage example."""
    from metal_native.profiling import TraceContext

    # Check for "with" statement in docstring
    assert 'with' in TraceContext.__doc__.lower()


def test_gpu_counters_no_c_extension():
    """Test gpu_counters handles missing C extension gracefully."""
    with patch('metal_native._C', None):
        from metal_native.profiling import gpu_counters
        result = gpu_counters()

        # Should return a dict with is_valid: False
        assert isinstance(result, dict)
        assert 'is_valid' in result
        assert result['is_valid'] is False


def test_begin_trace_raises_runtime_error():
    """Test begin_trace raises RuntimeError when C extension unavailable."""
    with patch('metal_native._C', None):
        from metal_native.profiling import begin_trace
        with pytest.raises(RuntimeError, match="Metal profiling not available"):
            begin_trace()


def test_end_trace_raises_runtime_error():
    """Test end_trace raises RuntimeError when C extension unavailable."""
    with patch('metal_native._C', None):
        from metal_native.profiling import end_trace
        with pytest.raises(RuntimeError, match="Metal profiling not available"):
            end_trace()


def test_memory_snapshot_raises_runtime_error():
    """Test memory_snapshot raises RuntimeError when C extension unavailable."""
    with patch('metal_native._C', None):
        from metal_native.profiling import memory_snapshot
        with pytest.raises(RuntimeError, match="Metal memory profiling not available"):
            memory_snapshot()


def test_memory_report_calls_memory_snapshot():
    """Test memory_report calls memory_snapshot internally."""
    from metal_native.profiling import memory_report

    mock_snapshot = {
        'total_allocated': 1024 * 1024,
        'total_cached': 512 * 1024,
        'peak_allocated': 2048 * 1024,
        'available_system': 8 * 1024 * 1024 * 1024,
        'heaps': [],
        'allocations': []
    }

    with patch('metal_native.profiling.memory_snapshot', return_value=mock_snapshot):
        report = memory_report()

        assert isinstance(report, str)
        assert 'Memory Report' in report
        assert 'MB' in report


def test_memory_report_formats_output():
    """Test memory_report produces formatted output."""
    from metal_native.profiling import memory_report

    mock_snapshot = {
        'total_allocated': 100 * 1024 * 1024,  # 100 MB
        'total_cached': 50 * 1024 * 1024,
        'peak_allocated': 150 * 1024 * 1024,
        'available_system': 16 * 1024 * 1024 * 1024,
        'heaps': [],
        'allocations': []
    }

    with patch('metal_native.profiling.memory_snapshot', return_value=mock_snapshot):
        report = memory_report()

        # Check report contains expected sections
        assert 'Overall Statistics' in report
        assert 'Total Allocated' in report


def test_memory_report_writes_file():
    """Test memory_report can write to file."""
    from metal_native.profiling import memory_report
    import tempfile
    import os

    mock_snapshot = {
        'total_allocated': 1024,
        'total_cached': 512,
        'peak_allocated': 2048,
        'available_system': 8 * 1024 * 1024 * 1024,
        'heaps': [],
        'allocations': []
    }

    with patch('metal_native.profiling.memory_snapshot', return_value=mock_snapshot):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            temp_path = f.name

        try:
            report = memory_report(output_path=temp_path)

            # Check file was written
            assert os.path.exists(temp_path)
            with open(temp_path, 'r') as f:
                content = f.read()
                assert content == report
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def test_trace_context_alias():
    """Test trace_context is an alias for TraceContext."""
    from metal_native.profiling import TraceContext, trace_context

    assert trace_context is TraceContext


def test_profiling_module_docstring():
    """Test profiling module has descriptive docstring."""
    from metal_native import profiling

    assert profiling.__doc__ is not None
    assert 'profiling' in profiling.__doc__.lower() or 'performance' in profiling.__doc__.lower()
    assert 'GPU' in profiling.__doc__ or 'Metal' in profiling.__doc__
