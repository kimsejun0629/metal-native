"""MetalNative - High-performance deep learning on Apple Silicon.

MetalNative provides a PyTorch-compatible tensor library optimized for
Apple Silicon GPUs using Metal and MPSGraph. It features:

- Zero-copy interop with NumPy and PyTorch via DLPack
- Automatic mixed precision (AMP) with bfloat16 support
- Unified memory architecture (UMA) optimization
- MPSGraph-backed operations for maximum performance
- PyTorch-compatible API and autograd support

Basic usage:
    >>> import metal_native as mn
    >>> x = mn.zeros((3, 3))
    >>> y = mn.ones((3, 3))
    >>> z = x + y
    >>> z.numpy()  # Zero-copy on UMA
"""

from typing import Optional, Tuple, Union, List
import sys

__all__ = [
    # Version
    '__version__',
    # Tensor class and factory functions
    'Tensor',
    'zeros', 'ones', 'empty', 'tensor', 'randn',
    'from_numpy', 'from_torch', 'from_dlpack',
    # Device management
    'device_name', 'is_available', 'device_properties',
    'supports_bfloat16',
    # Memory and sync utilities
    'synchronize', 'empty_cache',
    'memory_allocated', 'max_memory_allocated', 'reset_peak_stats',
    # Random number generation
    'set_seed', 'manual_seed',
    # Data types
    'float32', 'float16', 'bfloat16',
    'int64', 'int32', 'int16', 'int8',
    'uint8', 'bool_',
    # Configuration
    'MetalNativeConfig', 'get_config', 'set_config',
    # Interoperability modules
    'interop', 'dlpack_bridge', 'accelerate_plugin',
]

# Import version
from ._version import __version__, __version_info__

# Try to import C extension
_C = None
_import_error: Optional[Exception] = None

try:
    from . import _C
except ImportError as e:
    _import_error = e
    # C extension not available - will raise errors on usage
    # This is OK during build/install time


def _ensure_initialized():
    """Ensure the C extension is loaded.

    Raises:
        RuntimeError: If C extension is not available
    """
    if _C is None:
        msg = ("MetalNative requires macOS 14+ on Apple Silicon (M1/M2/M3/M4). "
               "Ensure you have Xcode Command Line Tools installed and run: pip install -e .")
        if _import_error is not None:
            msg += f"\nImport error: {_import_error}"
        raise RuntimeError(msg)


# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    """Lazy attribute loading for submodules."""
    # Handle submodule imports
    if name in ('nn', 'optim', 'interop', 'dlpack_bridge', 'accelerate_plugin',
                'dtypes', 'config', '_version'):
        import importlib
        module = importlib.import_module(f'.{name}', __name__)
        globals()[name] = module
        return module

    # Handle direct imports that need the C extension
    _ensure_initialized()

    # Try importing from submodules
    try:
        if name in ('Tensor', 'zeros', 'ones', 'empty', 'tensor', 'randn', 'from_numpy', 'from_torch', 'from_dlpack'):
            from .tensor import Tensor, zeros, ones, empty, tensor, randn, from_numpy, from_torch, from_dlpack
            globals().update({
                'Tensor': Tensor, 'zeros': zeros, 'ones': ones, 'empty': empty,
                'tensor': tensor, 'randn': randn,
                'from_numpy': from_numpy, 'from_torch': from_torch, 'from_dlpack': from_dlpack,
            })
            return globals()[name]

        if name in ('device_name', 'is_available', 'device_properties', 'supports_bfloat16'):
            from .device import device_name, is_available, device_properties, supports_bfloat16
            globals().update({
                'device_name': device_name, 'is_available': is_available,
                'device_properties': device_properties, 'supports_bfloat16': supports_bfloat16,
            })
            return globals()[name]

        if name in ('synchronize', 'empty_cache', 'memory_allocated', 'max_memory_allocated',
                    'reset_peak_stats', 'set_seed', 'manual_seed'):
            from .utils import (synchronize, empty_cache, memory_allocated,
                               max_memory_allocated, reset_peak_stats, set_seed, manual_seed)
            globals().update({
                'synchronize': synchronize, 'empty_cache': empty_cache,
                'memory_allocated': memory_allocated, 'max_memory_allocated': max_memory_allocated,
                'reset_peak_stats': reset_peak_stats, 'set_seed': set_seed, 'manual_seed': manual_seed,
            })
            return globals()[name]

        if name in ('float32', 'float16', 'bfloat16', 'int64', 'int32', 'int16', 'int8', 'uint8', 'bool_'):
            from . import dtypes
            dtype_obj = getattr(dtypes, name)
            globals()[name] = dtype_obj
            return dtype_obj

        if name in ('MetalNativeConfig', 'get_config', 'set_config'):
            from .config import MetalNativeConfig, get_config, set_config
            globals().update({
                'MetalNativeConfig': MetalNativeConfig,
                'get_config': get_config,
                'set_config': set_config,
            })
            return globals()[name]

    except ImportError as e:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from e

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def _check_installation():
    """Check if metal_native is properly installed and Metal is available.

    Returns:
        Tuple of (is_ok, message)
    """
    if _C is None:
        return False, f"C extension not loaded: {_import_error}"

    try:
        from .device import is_available, device_name
        if not is_available():
            return False, "Metal is not available on this system"

        name = device_name()
        return True, f"metal_native {__version__} ready on {name}"
    except Exception as e:
        return False, f"Error checking Metal availability: {e}"


def print_device_info():
    """Print detailed device information to stdout."""
    ok, msg = _check_installation()
    if not ok:
        print(f"⚠️  {msg}", file=sys.stderr)
        return

    print(msg)

    try:
        from .device import get_device_info
        print(get_device_info())
    except Exception as e:
        print(f"Error getting device info: {e}", file=sys.stderr)


# Module-level initialization
def _initialize():
    """Initialize metal_native runtime (called on first import)."""
    if _C is None:
        # C extension not available, skip initialization
        return

    try:
        # Initialize device and allocator
        _C.initialize()
    except Exception as e:
        print(f"Warning: metal_native initialization failed: {e}", file=sys.stderr)


# Perform initialization on import
_initialize()
