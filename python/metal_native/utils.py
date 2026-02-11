"""Utility functions for device management and debugging.

This module provides utility functions for synchronization, memory management,
and random number generation.
"""

from typing import Optional

__all__ = [
    'synchronize',
    'empty_cache',
    'memory_allocated',
    'max_memory_allocated',
    'reset_peak_stats',
    'set_seed',
    'manual_seed',
]


def synchronize() -> None:
    """Block until all GPU operations complete.

    This function waits for all enqueued Metal commands to finish execution.
    Useful for accurate timing measurements and ensuring side effects are visible.

    Raises:
        RuntimeError: If synchronization fails
    """
    try:
        from metal_native import _C
        _C.synchronize()
    except (ImportError, AttributeError):
        raise RuntimeError("Cannot synchronize: metal_native._C not loaded")


def empty_cache() -> None:
    """Release all cached memory back to the system.

    This function frees memory held by the allocator's cache but not currently
    in use by tensors. Does not affect live tensor allocations.

    Note:
        This is primarily useful for reducing memory pressure when running
        multiple frameworks or processes that compete for GPU memory.
    """
    try:
        from metal_native import _C
        _C.empty_cache()
    except (ImportError, AttributeError):
        pass  # Fail silently if C extension not loaded


def memory_allocated() -> int:
    """Get current GPU memory allocated by metal_native.

    Returns:
        Number of bytes currently allocated on the GPU

    Raises:
        RuntimeError: If memory stats are not available
    """
    try:
        from metal_native import _C
        return _C.memory_allocated()
    except (ImportError, AttributeError):
        raise RuntimeError("Cannot query memory: metal_native._C not loaded")


def max_memory_allocated() -> int:
    """Get peak GPU memory allocated since program start or last reset.

    Returns:
        Peak number of bytes allocated on the GPU

    Raises:
        RuntimeError: If memory stats are not available
    """
    try:
        from metal_native import _C
        return _C.max_memory_allocated()
    except (ImportError, AttributeError):
        raise RuntimeError("Cannot query memory: metal_native._C not loaded")


def reset_peak_stats() -> None:
    """Reset peak memory statistics.

    After calling this, max_memory_allocated() will return the peak
    since this call, not since program start.
    """
    try:
        from metal_native import _C
        _C.reset_peak_stats()
    except (ImportError, AttributeError):
        pass  # Fail silently


def set_seed(seed: int) -> None:
    """Set the random number generator seed.

    Args:
        seed: Random seed value

    Raises:
        ValueError: If seed is negative
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    try:
        from metal_native import _C
        _C.set_seed(seed)
    except (ImportError, AttributeError):
        raise RuntimeError("Cannot set seed: metal_native._C not loaded")


def manual_seed(seed: int) -> None:
    """Alias for set_seed() for PyTorch compatibility.

    Args:
        seed: Random seed value
    """
    set_seed(seed)


def get_memory_info() -> str:
    """Get a human-readable summary of memory usage.

    Returns:
        Formatted string with memory statistics
    """
    try:
        current = memory_allocated()
        peak = max_memory_allocated()

        return (
            f"GPU Memory:\n"
            f"  Current: {current / (1024**2):.2f} MB\n"
            f"  Peak:    {peak / (1024**2):.2f} MB"
        )
    except RuntimeError:
        return "GPU Memory: Not available"


class SynchronizeContext:
    """Context manager for automatic synchronization.

    Example:
        with SynchronizeContext():
            # GPU operations
            result = tensor_a + tensor_b
        # GPU is synchronized here
        print(result.numpy())
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        synchronize()
        return False
