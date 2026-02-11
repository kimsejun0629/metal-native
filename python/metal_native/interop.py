"""PyTorch interoperability for MetalNative tensors.

This module provides zero-copy conversion between PyTorch and MetalNative
tensors via shared memory on Apple Silicon's unified memory architecture.
"""

from typing import Optional
import numpy as np

__all__ = [
    'from_torch',
    'to_torch',
]


def from_torch(tensor: 'torch.Tensor', requires_grad: Optional[bool] = None) -> 'Tensor':
    """Convert a PyTorch tensor to MetalNative tensor.

    On Apple Silicon with unified memory, this uses zero-copy when the tensor
    is contiguous and on CPU. Otherwise, data is copied.

    Args:
        tensor: PyTorch tensor to convert
        requires_grad: Whether to track gradients (default: inherit from PyTorch)

    Returns:
        MetalNative Tensor with the same data

    Raises:
        ImportError: If PyTorch is not installed

    Example:
        >>> import torch
        >>> import metal_native as mn
        >>> pt_tensor = torch.randn(3, 3)
        >>> mn_tensor = mn.interop.from_torch(pt_tensor)
    """
    try:
        import torch
        import torch.utils.dlpack
    except ImportError:
        raise ImportError(
            "PyTorch is required for from_torch(). "
            "Install it with: pip install torch"
        )

    from .tensor import Tensor
    from metal_native import _C

    if requires_grad is None:
        requires_grad = tensor.requires_grad

    # Zero-copy path for MPS tensors (Apple Silicon unified memory)
    if tensor.device.type == 'mps':
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Map torch dtype to string
        dtype_map = {
            torch.float32: 'float32',
            torch.float16: 'float16',
            torch.bfloat16: 'bfloat16',
            torch.int64: 'int64',
            torch.int32: 'int32',
            torch.int16: 'int16',
            torch.int8: 'int8',
            torch.uint8: 'uint8',
            torch.bool: 'bool',
        }
        dtype_str = dtype_map.get(tensor.dtype)
        if dtype_str is None:
            # Fall back to string conversion for unknown types
            dtype_str = str(tensor.dtype).replace('torch.', '')

        handle = _C.tensor_from_mps_ptr(
            tensor.data_ptr(),
            tensor.storage().nbytes(),
            list(tensor.shape),
            list(tensor.stride()),
            dtype_str,
        )
        return Tensor(_native_handle=handle, requires_grad=requires_grad)

    # Move to CPU if needed for UMA shared memory (for CUDA/other GPUs)
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()

    # Use zero-copy conversion via DLPack when tensor is contiguous
    if tensor.is_contiguous():
        try:
            dlpack_capsule = torch.utils.dlpack.to_dlpack(tensor)
            handle = _C.tensor_from_dlpack(dlpack_capsule, requires_grad)
            return Tensor(_native_handle=handle, requires_grad=requires_grad)
        except Exception:
            # Fall back to copy if DLPack fails
            pass

    # Fallback: copy via NumPy
    np_array = tensor.detach().cpu().numpy()
    return Tensor(data=np_array, requires_grad=requires_grad)


def to_torch(mn_tensor: 'Tensor') -> 'torch.Tensor':
    """Convert a MetalNative tensor to PyTorch tensor.

    Uses zero-copy via shared memory when possible on Apple Silicon.

    Args:
        mn_tensor: MetalNative tensor to convert

    Returns:
        PyTorch tensor sharing the same underlying memory (when possible)

    Raises:
        ImportError: If PyTorch is not installed

    Example:
        >>> import metal_native as mn
        >>> mn_tensor = mn.zeros((3, 3))
        >>> pt_tensor = mn.interop.to_torch(mn_tensor)
    """
    try:
        import torch
        import torch.utils.dlpack
    except ImportError:
        raise ImportError(
            "PyTorch is required for to_torch(). "
            "Install it with: pip install torch"
        )

    from .tensor import Tensor

    if not isinstance(mn_tensor, Tensor):
        raise TypeError(f"Expected MetalNative Tensor, got {type(mn_tensor)}")

    # Use DLPack for zero-copy conversion
    try:
        dlpack_capsule = mn_tensor.__dlpack__()
        pt_tensor = torch.utils.dlpack.from_dlpack(dlpack_capsule)

        # Preserve requires_grad flag
        if mn_tensor.requires_grad:
            pt_tensor.requires_grad_(True)

        return pt_tensor
    except Exception:
        # Fallback: copy via NumPy
        np_array = mn_tensor.numpy()
        pt_tensor = torch.from_numpy(np_array)

        if mn_tensor.requires_grad:
            pt_tensor.requires_grad_(True)

        return pt_tensor
