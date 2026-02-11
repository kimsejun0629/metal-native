"""DLPack bridge for cross-framework tensor exchange.

This module provides DLPack-based interoperability with any framework that
supports the DLPack protocol (PyTorch, JAX, TensorFlow, CuPy, etc.).

DLPack is a zero-copy tensor exchange protocol that allows different
frameworks to share tensor data without copying.
"""

from typing import Any, Tuple

__all__ = [
    'to_dlpack',
    'from_dlpack',
]


def to_dlpack(mn_tensor: 'Tensor') -> Any:
    """Export a MetalNative tensor as a DLPack capsule.

    The DLPack capsule can be consumed by any framework that supports
    the DLPack protocol for zero-copy tensor import.

    Args:
        mn_tensor: MetalNative tensor to export

    Returns:
        PyCapsule object containing DLPack tensor descriptor

    Raises:
        TypeError: If mn_tensor is not a MetalNative Tensor
        RuntimeError: If DLPack export fails

    Example:
        >>> import metal_native as mn
        >>> import torch
        >>> mn_tensor = mn.ones((3, 3))
        >>> capsule = mn.dlpack_bridge.to_dlpack(mn_tensor)
        >>> pt_tensor = torch.utils.dlpack.from_dlpack(capsule)
    """
    from .tensor import Tensor

    if not isinstance(mn_tensor, Tensor):
        raise TypeError(f"Expected MetalNative Tensor, got {type(mn_tensor)}")

    try:
        from metal_native import _C
        return _C.tensor_to_dlpack(mn_tensor._handle)
    except Exception as e:
        raise RuntimeError(f"Failed to export tensor to DLPack: {e}")


def from_dlpack(capsule: Any) -> 'Tensor':
    """Import a tensor from a DLPack capsule.

    Creates a MetalNative tensor from any framework that exports DLPack
    capsules. The tensor may share memory with the source framework.

    Args:
        capsule: PyCapsule object containing DLPack tensor descriptor

    Returns:
        MetalNative Tensor

    Raises:
        RuntimeError: If DLPack import fails
        TypeError: If capsule is not a valid DLPack object

    Example:
        >>> import torch
        >>> import metal_native as mn
        >>> pt_tensor = torch.randn(3, 3)
        >>> capsule = torch.utils.dlpack.to_dlpack(pt_tensor)
        >>> mn_tensor = mn.dlpack_bridge.from_dlpack(capsule)
    """
    from .tensor import Tensor

    try:
        from metal_native import _C
        handle = _C.tensor_from_dlpack(capsule, False)
        return Tensor(_native_handle=handle, requires_grad=False)
    except Exception as e:
        raise RuntimeError(f"Failed to import tensor from DLPack: {e}")


def get_dlpack_device(mn_tensor: 'Tensor') -> Tuple[int, int]:
    """Get the DLPack device type and ID for a MetalNative tensor.

    Args:
        mn_tensor: MetalNative tensor

    Returns:
        Tuple of (device_type, device_id) where device_type=8 is Metal

    Example:
        >>> import metal_native as mn
        >>> tensor = mn.zeros((3, 3))
        >>> device_type, device_id = mn.dlpack_bridge.get_dlpack_device(tensor)
        >>> print(f"Device: type={device_type}, id={device_id}")
        Device: type=8, id=0
    """
    from .tensor import Tensor

    if not isinstance(mn_tensor, Tensor):
        raise TypeError(f"Expected MetalNative Tensor, got {type(mn_tensor)}")

    return mn_tensor.__dlpack_device__()
