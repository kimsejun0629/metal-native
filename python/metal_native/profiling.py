"""Profiling and performance analysis API.

This module provides GPU trace capture, performance counter reading,
and memory profiling capabilities for Metal operations.
"""

from typing import Dict, Any, Optional
from contextlib import contextmanager

__all__ = [
    'begin_trace',
    'end_trace',
    'gpu_counters',
    'memory_snapshot',
    'memory_report',
    'TraceContext',
]


def begin_trace(path: Optional[str] = None) -> None:
    """Begin GPU trace capture.

    Starts recording all GPU commands for later analysis in Xcode Instruments
    or Metal Debugger.

    Args:
        path: Optional output path for .gputrace file. If None, uses default
              location or sends to Xcode developer tools.

    Raises:
        RuntimeError: If Metal is not available or capture fails to start
    """
    try:
        from metal_native import _C

        if path is not None:
            _C.profiling_set_capture_destination(path)

        _C.profiling_begin_capture()
    except (ImportError, AttributeError):
        raise RuntimeError("Metal profiling not available. Is metal_native._C loaded?")
    except Exception as e:
        raise RuntimeError(f"Failed to begin GPU trace capture: {e}")


def end_trace() -> None:
    """End GPU trace capture and write output file.

    Stops the current capture session and writes the .gputrace file if a
    path was specified in begin_trace().

    Raises:
        RuntimeError: If Metal is not available or no capture is in progress
    """
    try:
        from metal_native import _C
        _C.profiling_end_capture()
    except (ImportError, AttributeError):
        raise RuntimeError("Metal profiling not available")
    except Exception as e:
        raise RuntimeError(f"Failed to end GPU trace capture: {e}")


def gpu_counters() -> Dict[str, Any]:
    """Read GPU performance counters.

    Samples current GPU hardware performance counters including utilization,
    memory bandwidth, and occupancy.

    Returns:
        Dictionary with performance counter data:
        - gpu_utilization: GPU busy percentage (0.0 - 100.0)
        - alu_utilization: Shader core utilization (0.0 - 100.0)
        - memory_bandwidth: Memory bandwidth in GB/s
        - occupancy: Thread occupancy percentage (0.0 - 100.0)
        - timestamp_ns: Sample timestamp in nanoseconds
        - is_valid: Whether counter data is valid (false on older GPUs)

    Note:
        Performance counters may not be available on all GPU models.
        Check the 'is_valid' field in the returned dictionary.
        Returns invalid data if Metal is not available.
    """
    try:
        from metal_native import _C
        return _C.profiling_sample_counters()
    except (ImportError, AttributeError) as e:
        # Return invalid snapshot when C extension not available
        return {
            'gpu_utilization': 0.0,
            'alu_utilization': 0.0,
            'memory_bandwidth': 0.0,
            'occupancy': 0.0,
            'timestamp_ns': 0,
            'is_valid': False,
            'error': str(e)
        }
    except Exception as e:
        # Return invalid snapshot on error
        return {
            'gpu_utilization': 0.0,
            'alu_utilization': 0.0,
            'memory_bandwidth': 0.0,
            'occupancy': 0.0,
            'timestamp_ns': 0,
            'is_valid': False,
            'error': str(e)
        }


def memory_snapshot() -> Dict[str, Any]:
    """Take a snapshot of current GPU memory state.

    Captures detailed memory allocation information including heap statistics,
    individual allocations, and fragmentation metrics.

    Returns:
        Dictionary with memory snapshot data:
        - total_allocated: Total bytes currently allocated
        - total_cached: Total bytes in cache/free pool
        - peak_allocated: Peak allocation during session
        - available_system: Available system memory
        - heaps: List of per-heap statistics
        - allocations: List of individual allocation records

    Raises:
        RuntimeError: If Metal is not available
    """
    try:
        from metal_native import _C
        return _C.memory_snapshot()
    except (ImportError, AttributeError):
        raise RuntimeError("Metal memory profiling not available")
    except Exception as e:
        raise RuntimeError(f"Failed to take memory snapshot: {e}")


def memory_report(output_path: Optional[str] = None) -> str:
    """Generate a formatted memory report.

    Creates a human-readable report of GPU memory usage, heap statistics,
    and allocation details. Optionally writes the report to a file.

    Args:
        output_path: Optional path to write report. If None, returns string only.

    Returns:
        Formatted memory report as string

    Raises:
        RuntimeError: If Metal is not available or report generation fails
    """
    try:
        snapshot = memory_snapshot()

        # Build report
        lines = []
        lines.append("=" * 70)
        lines.append("Metal Native - Memory Report")
        lines.append("=" * 70)
        lines.append("")

        # Overall statistics
        lines.append("Overall Statistics:")
        lines.append(f"  Total Allocated:     {snapshot['total_allocated'] / (1024**2):.2f} MB")
        lines.append(f"  Total Cached:        {snapshot['total_cached'] / (1024**2):.2f} MB")
        lines.append(f"  Peak Allocated:      {snapshot['peak_allocated'] / (1024**2):.2f} MB")
        lines.append(f"  Available System:    {snapshot['available_system'] / (1024**3):.2f} GB")
        lines.append("")

        # Heap statistics
        if snapshot.get('heaps'):
            lines.append("Heap Statistics:")
            for i, heap in enumerate(snapshot['heaps']):
                lines.append(f"  Heap {i}:")
                lines.append(f"    Size:              {heap['heap_size'] / (1024**2):.2f} MB")
                lines.append(f"    Used:              {heap['used_bytes'] / (1024**2):.2f} MB")
                lines.append(f"    Free:              {heap['free_bytes'] / (1024**2):.2f} MB")
                lines.append(f"    Allocations:       {heap['num_allocations']}")
                lines.append(f"    Fragmentation:     {heap['fragmentation_ratio']:.2%}")
            lines.append("")

        # Allocation summary
        if snapshot.get('allocations'):
            lines.append(f"Active Allocations: {len(snapshot['allocations'])}")
            lines.append("")

            # Show top 10 largest allocations
            sorted_allocs = sorted(
                snapshot['allocations'],
                key=lambda x: x['size'],
                reverse=True
            )[:10]

            if sorted_allocs:
                lines.append("Top 10 Largest Allocations:")
                for alloc in sorted_allocs:
                    size_mb = alloc['size'] / (1024**2)
                    tag = alloc.get('tag', 'unknown')
                    lines.append(f"  {size_mb:>10.2f} MB  -  {tag}")
                lines.append("")

        lines.append("=" * 70)

        report = "\n".join(lines)

        # Write to file if requested
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)

        return report

    except Exception as e:
        raise RuntimeError(f"Failed to generate memory report: {e}")


@contextmanager
def TraceContext(path: Optional[str] = None):
    """Context manager for GPU trace capture.

    Automatically begins trace capture on entry and ends on exit.

    Args:
        path: Optional output path for .gputrace file

    Usage:
        >>> with TraceContext("my_trace.gputrace"):
        ...     # GPU operations here
        ...     model(input)

    Raises:
        RuntimeError: If Metal is not available or capture fails
    """
    begin_trace(path)
    try:
        yield
    finally:
        try:
            end_trace()
        except Exception:
            # Silently ignore errors on cleanup
            pass


# Convenience aliases
trace_context = TraceContext
