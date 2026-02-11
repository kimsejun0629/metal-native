"""Device query and management API.

This module provides functions for querying Metal device capabilities
and managing device state.
"""

from typing import Dict, Any, Optional

__all__ = [
    'device_name',
    'is_available',
    'device_properties',
    'supports_bfloat16',
    'get_device_info',
]


def device_name() -> str:
    """Get the name of the Metal device.

    Returns:
        Device name string (e.g., "Apple M1 Max")

    Raises:
        RuntimeError: If Metal is not available
    """
    try:
        from metal_native import _C
        return _C.device_name()
    except (ImportError, AttributeError):
        raise RuntimeError("Metal device not available. Is metal_native._C loaded?")


def is_available() -> bool:
    """Check if Metal is available on this system.

    Returns:
        True if Metal is available and functional
    """
    try:
        from metal_native import _C
        return _C.is_available()
    except (ImportError, AttributeError):
        return False


def device_properties() -> Dict[str, Any]:
    """Get detailed Metal device properties.

    Returns:
        Dictionary with device properties:
        - name: Device name
        - cores: Number of GPU cores
        - memory: Total device memory in bytes
        - bandwidth: Memory bandwidth in GB/s
        - max_buffer_length: Maximum buffer size
        - supports_bfloat16: Whether bfloat16 is supported
        - unified_memory: Whether using unified memory architecture
        - recommended_working_set: Recommended working set size in bytes

    Raises:
        RuntimeError: If Metal is not available
    """
    try:
        from metal_native import _C
        return _C.device_properties()
    except (ImportError, AttributeError):
        raise RuntimeError("Metal device not available")


def supports_bfloat16() -> bool:
    """Check if the Metal device supports bfloat16 operations.

    Returns:
        True if bfloat16 is supported

    Raises:
        RuntimeError: If Metal is not available
    """
    try:
        from metal_native import _C
        return _C.supports_bfloat16()
    except (ImportError, AttributeError):
        raise RuntimeError("Metal device not available")


def get_device_info() -> str:
    """Get a human-readable summary of device information.

    Returns:
        Formatted string with device information
    """
    if not is_available():
        return "Metal: Not available"

    props = device_properties()

    info_lines = [
        f"Metal Device: {props.get('name', 'Unknown')}",
        f"  GPU Cores: {props.get('cores', 'Unknown')}",
        f"  Memory: {props.get('memory', 0) / (1024**3):.2f} GB",
        f"  Bandwidth: {props.get('bandwidth', 0):.2f} GB/s",
        f"  Unified Memory: {props.get('unified_memory', False)}",
        f"  BFloat16 Support: {props.get('supports_bfloat16', False)}",
    ]

    return "\n".join(info_lines)


class Device:
    """Device context manager for explicit device selection.

    Note: Metal has a single unified device, so this is mainly for
    API compatibility with multi-GPU frameworks.
    """

    def __init__(self, device: Optional[str] = None):
        """Initialize device context.

        Args:
            device: Device identifier (only 'metal' or None supported)

        Raises:
            ValueError: If device is not 'metal' or None
        """
        if device is not None and device != 'metal':
            raise ValueError(f"Invalid device: {device}. Only 'metal' is supported.")
        self.device = 'metal'

    def __enter__(self):
        """Enter device context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit device context."""
        pass

    def __repr__(self) -> str:
        return f"Device('{self.device}')"
