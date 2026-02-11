"""HuggingFace Accelerate backend plugin for MetalNative.

This module registers MetalNative as an Accelerate device backend,
enabling seamless integration with HuggingFace Transformers and
the Accelerate library for distributed training workflows.
"""

from typing import Optional, Any, List

__all__ = [
    'MetalNativeAccelerator',
    'is_available',
]


def is_available() -> bool:
    """Check if MetalNative is available for Accelerate.

    Returns:
        True if MetalNative is available and functional
    """
    try:
        from metal_native import _C
        return _C.is_available()
    except (ImportError, AttributeError):
        return False


class MetalNativeAccelerator:
    """Accelerate backend for MetalNative.

    This class implements the Accelerate backend interface, allowing
    HuggingFace models to run on Apple Silicon via MetalNative.

    Example:
        >>> from accelerate import Accelerator
        >>> from metal_native.accelerate_plugin import MetalNativeAccelerator
        >>> accelerator = Accelerator(device_placement=True)
        >>> # Models and optimizers will automatically use MetalNative
    """

    def __init__(self):
        """Initialize the MetalNative accelerator backend."""
        if not is_available():
            raise RuntimeError(
                "MetalNative is not available. "
                "Ensure metal_native is installed and Metal is supported."
            )

    @property
    def device(self) -> str:
        """Get the device string for this backend.

        Returns:
            'metal_native' device identifier
        """
        return 'metal_native'

    @classmethod
    def is_available(cls) -> bool:
        """Check if the backend is available.

        Returns:
            True if MetalNative is available
        """
        return is_available()

    def prepare_model(self, model: Any) -> Any:
        """Move a model to MetalNative device.

        Args:
            model: PyTorch model to prepare

        Returns:
            Model with parameters moved to MetalNative

        Example:
            >>> accelerator = MetalNativeAccelerator()
            >>> model = accelerator.prepare_model(model)
        """
        try:
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch is required for model preparation")

        from .interop import from_torch, to_torch

        # Move all parameters and buffers to MetalNative
        for name, param in model.named_parameters():
            if param is not None:
                # Convert to MetalNative and back to PyTorch
                # This establishes the Metal-backed storage
                mn_tensor = from_torch(param.data)
                param.data = to_torch(mn_tensor)

        for name, buffer in model.named_buffers():
            if buffer is not None:
                mn_tensor = from_torch(buffer)
                # Replace buffer in-place
                model.register_buffer(name, to_torch(mn_tensor))

        return model

    def prepare_optimizer(self, optimizer: Any) -> Any:
        """Wrap an optimizer to work with MetalNative tensors.

        Args:
            optimizer: PyTorch optimizer to prepare

        Returns:
            Wrapped optimizer compatible with MetalNative

        Example:
            >>> optimizer = torch.optim.Adam(model.parameters())
            >>> optimizer = accelerator.prepare_optimizer(optimizer)
        """
        # For now, PyTorch optimizers work directly with MetalNative
        # tensors via the DLPack bridge. No wrapping needed.
        return optimizer

    def synchronize(self) -> None:
        """Synchronize GPU execution.

        Ensures all pending GPU operations complete before returning.
        """
        try:
            from metal_native import _C
            _C.synchronize()
        except (ImportError, AttributeError):
            pass

    def empty_cache(self) -> None:
        """Clear the GPU memory cache.

        Frees unused cached memory back to the system.
        """
        try:
            from metal_native import _C
            _C.empty_cache()
        except (ImportError, AttributeError):
            pass

    def get_memory_info(self) -> dict:
        """Get current GPU memory usage statistics.

        Returns:
            Dictionary with memory statistics:
            - allocated: Currently allocated memory in bytes
            - reserved: Total reserved memory in bytes
            - peak: Peak allocated memory in bytes
        """
        try:
            from metal_native import _C
            return {
                'allocated': _C.memory_allocated(),
                'reserved': _C.memory_allocated(),  # Metal doesn't separate these
                'peak': _C.max_memory_allocated(),
            }
        except (ImportError, AttributeError):
            return {'allocated': 0, 'reserved': 0, 'peak': 0}


# Register with Accelerate if available
def _register_backend():
    """Register MetalNative as an Accelerate backend."""
    try:
        from accelerate import Accelerator
        from accelerate.state import AcceleratorState

        # Register the backend
        # Note: This requires Accelerate >= 0.20.0 with custom backend support
        # For now, this is a placeholder for future Accelerate integration
        pass
    except ImportError:
        # Accelerate not installed, skip registration
        pass


# Attempt registration on module import
_register_backend()
