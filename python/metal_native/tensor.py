"""MetalNative Tensor implementation with PyTorch and NumPy interoperability.

This module provides the Tensor class, which is the primary data structure for
metal_native computations. It supports zero-copy interop with NumPy and PyTorch
via the DLPack protocol.
"""

from typing import Union, Tuple, List, Optional, Any, Callable
import numpy as np
from . import dtypes

__all__ = [
    'Tensor',
    'from_numpy',
    'from_torch',
    'from_dlpack',
    'zeros',
    'ones',
    'empty',
    'tensor',
    'randn',
    'register_torch_override',
]


class Tensor:
    """MetalNative Tensor - the primary data structure for GPU computations.

    A Tensor represents a multi-dimensional array stored on the Metal GPU.
    It supports arithmetic operations, interoperability with NumPy/PyTorch,
    and automatic gradient tracking.

    Attributes:
        shape: Tuple of tensor dimensions
        dtype: Data type (metal_native.dtypes.DType)
        device: Always 'metal' for MetalNative tensors
        requires_grad: Whether to track gradients for autograd
        grad: Gradient tensor (if requires_grad=True and backward called)
    """

    def __init__(
        self,
        data: Optional[Union[List, np.ndarray, 'Tensor']] = None,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[dtypes.DType] = None,
        requires_grad: bool = False,
        _native_handle: Optional[Any] = None,
    ):
        """Initialize a Tensor.

        Args:
            data: Initial data (list, NumPy array, or another Tensor)
            shape: Explicit shape (required if data is None)
            dtype: Data type (inferred from data if not provided)
            requires_grad: Whether to track gradients
            _native_handle: Internal C++ tensor handle (for internal use)

        Raises:
            ValueError: If shape and data are incompatible
            ImportError: If C extension is not loaded
        """
        if _native_handle is not None:
            # Constructed from C++ side
            self._handle = _native_handle
            self._shape = None  # Lazy-loaded from C++
            self._dtype = None  # Lazy-loaded from C++
            self._requires_grad = requires_grad
            self._grad = None
            return

        # Python-side construction
        try:
            from metal_native import _C
        except ImportError:
            raise ImportError("metal_native._C extension not loaded")

        # Convert input data to NumPy for easier handling
        if data is not None:
            if isinstance(data, Tensor):
                # Copy from another tensor
                np_data = data.numpy()
            elif isinstance(data, np.ndarray):
                np_data = data
            else:
                np_data = np.array(data)

            if dtype is None:
                # Infer dtype from data
                try:
                    dtype = dtypes.numpy_to_dtype(np_data.dtype)
                except ValueError:
                    # Default to float32 for unsupported types
                    dtype = dtypes.float32
                    np_data = np_data.astype(np.float32)
            else:
                # Convert to requested dtype
                if dtype != dtypes.bfloat16:
                    target_np_dtype = dtypes.dtype_to_numpy(dtype)
                    np_data = np_data.astype(target_np_dtype)

            if shape is not None and np_data.shape != shape:
                np_data = np_data.reshape(shape)

            self._handle = _C.tensor_from_numpy(np_data, dtype.name, requires_grad)
        else:
            if shape is None:
                raise ValueError("Either data or shape must be provided")
            if dtype is None:
                dtype = dtypes.float32

            self._handle = _C.tensor_empty(shape, dtype.name, requires_grad)

        self._shape = shape
        self._dtype = dtype
        self._requires_grad = requires_grad
        self._grad = None

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape."""
        if self._shape is None:
            from metal_native import _C
            self._shape = tuple(_C.tensor_shape(self._handle))
        return self._shape

    @property
    def dtype(self) -> dtypes.DType:
        """Get tensor data type."""
        if self._dtype is None:
            from metal_native import _C
            dtype_name = _C.tensor_dtype(self._handle)
            self._dtype = dtypes._DTYPE_MAP[dtype_name]
        return self._dtype

    @property
    def device(self) -> str:
        """Get device string (always 'metal')."""
        return 'metal'

    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Get total number of elements."""
        import math
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        """Get total size in bytes."""
        return self.size * self.dtype.itemsize

    @property
    def requires_grad(self) -> bool:
        """Check if gradient tracking is enabled."""
        return self._requires_grad

    @property
    def grad(self) -> Optional['Tensor']:
        """Get gradient tensor (None if no gradient computed)."""
        return self._grad

    def item(self) -> Union[float, int, bool]:
        """Extract scalar value (for single-element tensors).

        Returns:
            Python scalar value

        Raises:
            ValueError: If tensor has more than one element
        """
        if self.size != 1:
            raise ValueError(f"item() only works for single-element tensors, got shape {self.shape}")

        from metal_native import _C
        return _C.tensor_item(self._handle)

    def numpy(self) -> np.ndarray:
        """Convert to NumPy array.

        On Apple Silicon (UMA), this is zero-copy when possible.
        Otherwise, data is copied from GPU to CPU memory.

        Returns:
            NumPy array with the same data
        """
        from metal_native import _C
        return _C.tensor_to_numpy(self._handle)

    def torch(self) -> 'torch.Tensor':
        """Convert to PyTorch tensor via DLPack (zero-copy).

        Returns:
            PyTorch tensor sharing the same underlying memory

        Raises:
            ImportError: If PyTorch is not installed
        """
        try:
            import torch.utils.dlpack
        except ImportError:
            raise ImportError("PyTorch is required for torch() conversion")

        return torch.utils.dlpack.from_dlpack(self)

    def __dlpack__(self, stream=None) -> Any:
        """Export tensor via DLPack protocol.

        Args:
            stream: Stream for synchronization (ignored for Metal)

        Returns:
            DLPack capsule
        """
        from metal_native import _C
        return _C.tensor_to_dlpack(self._handle)

    def __dlpack_device__(self) -> Tuple[int, int]:
        """Return DLPack device tuple.

        Returns:
            (device_type, device_id) where device_type=8 is Metal
        """
        return (8, 0)  # 8 = kDLMetal, 0 = device index

    def __repr__(self) -> str:
        """String representation of the tensor."""
        grad_str = ", requires_grad=True" if self._requires_grad else ""
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, device={self.device}{grad_str})"

    def __str__(self) -> str:
        """Detailed string representation with data preview."""
        # For small tensors, show actual data
        if self.size <= 10:
            try:
                data_str = str(self.numpy())
            except Exception:
                data_str = "<data unavailable>"
        else:
            data_str = f"<{self.size} elements>"

        return f"Tensor({data_str}, dtype={self.dtype}, device={self.device})"

    # Arithmetic operations
    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Element-wise addition."""
        from metal_native import _C
        if isinstance(other, Tensor):
            return Tensor(_native_handle=_C.tensor_add(self._handle, other._handle))
        else:
            return Tensor(_native_handle=_C.tensor_add_scalar(self._handle, float(other)))

    def __radd__(self, other: Union[float, int]) -> 'Tensor':
        """Reverse addition."""
        return self.__add__(other)

    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Element-wise subtraction."""
        from metal_native import _C
        if isinstance(other, Tensor):
            return Tensor(_native_handle=_C.tensor_sub(self._handle, other._handle))
        else:
            return Tensor(_native_handle=_C.tensor_sub_scalar(self._handle, float(other)))

    def __rsub__(self, other: Union[float, int]) -> 'Tensor':
        """Reverse subtraction."""
        from metal_native import _C
        return Tensor(_native_handle=_C.tensor_rsub_scalar(self._handle, float(other)))

    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Element-wise multiplication."""
        from metal_native import _C
        if isinstance(other, Tensor):
            return Tensor(_native_handle=_C.tensor_mul(self._handle, other._handle))
        else:
            return Tensor(_native_handle=_C.tensor_mul_scalar(self._handle, float(other)))

    def __rmul__(self, other: Union[float, int]) -> 'Tensor':
        """Reverse multiplication."""
        return self.__mul__(other)

    def __truediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Element-wise division."""
        from metal_native import _C
        if isinstance(other, Tensor):
            return Tensor(_native_handle=_C.tensor_div(self._handle, other._handle))
        else:
            return Tensor(_native_handle=_C.tensor_div_scalar(self._handle, float(other)))

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication."""
        from metal_native import _C
        if not isinstance(other, Tensor):
            raise TypeError(f"matmul requires Tensor, got {type(other)}")
        return Tensor(_native_handle=_C.tensor_matmul(self._handle, other._handle))

    def __neg__(self) -> 'Tensor':
        """Unary negation."""
        return self.__mul__(-1)

    # PyTorch function protocol (basic structure)
    def __torch_function__(self, func: Callable, types: Tuple, args: Tuple, kwargs: Optional[dict] = None) -> Any:
        """Handle PyTorch function calls on MetalNative tensors.

        This enables using PyTorch functions directly on metal_native tensors.
        Falls back to converting to PyTorch, applying the function, and converting back.

        Args:
            func: PyTorch function being called
            types: Types of arguments
            args: Function arguments
            kwargs: Keyword arguments

        Returns:
            Result of the function (may be Tensor or other type)
        """
        if kwargs is None:
            kwargs = {}

        # Check if we have a registered override
        if func in _TORCH_FUNCTION_OVERRIDES:
            return _TORCH_FUNCTION_OVERRIDES[func](*args, **kwargs)

        # Fallback: convert to PyTorch, apply function, convert back
        try:
            import torch
            torch_args = tuple(
                arg.torch() if isinstance(arg, Tensor) else arg
                for arg in args
            )
            result = func(*torch_args, **kwargs)

            # Convert result back to metal_native if it's a tensor
            if isinstance(result, torch.Tensor):
                return from_torch(result)
            return result
        except Exception as e:
            raise NotImplementedError(
                f"PyTorch function {func.__name__} not supported for metal_native tensors: {e}"
            )


# Factory functions
def zeros(shape: Tuple[int, ...], dtype: Optional[dtypes.DType] = None, requires_grad: bool = False) -> Tensor:
    """Create a tensor filled with zeros.

    Args:
        shape: Shape of the tensor
        dtype: Data type (default: float32)
        requires_grad: Whether to track gradients

    Returns:
        Tensor filled with zeros
    """
    if dtype is None:
        dtype = dtypes.float32

    from metal_native import _C
    handle = _C.tensor_zeros(shape, dtype.name, requires_grad)
    return Tensor(_native_handle=handle, requires_grad=requires_grad)


def ones(shape: Tuple[int, ...], dtype: Optional[dtypes.DType] = None, requires_grad: bool = False) -> Tensor:
    """Create a tensor filled with ones.

    Args:
        shape: Shape of the tensor
        dtype: Data type (default: float32)
        requires_grad: Whether to track gradients

    Returns:
        Tensor filled with ones
    """
    if dtype is None:
        dtype = dtypes.float32

    from metal_native import _C
    handle = _C.tensor_ones(shape, dtype.name, requires_grad)
    return Tensor(_native_handle=handle, requires_grad=requires_grad)


def empty(shape: Tuple[int, ...], dtype: Optional[dtypes.DType] = None, requires_grad: bool = False) -> Tensor:
    """Create an uninitialized tensor.

    Args:
        shape: Shape of the tensor
        dtype: Data type (default: float32)
        requires_grad: Whether to track gradients

    Returns:
        Uninitialized tensor
    """
    return Tensor(shape=shape, dtype=dtype, requires_grad=requires_grad)


def tensor(data: Union[List, np.ndarray, 'Tensor'], dtype: Optional[dtypes.DType] = None, requires_grad: bool = False) -> Tensor:
    """Create a tensor from a Python list or NumPy array.

    Args:
        data: Input data (list, NumPy array, or another Tensor)
        dtype: Data type (default: inferred from data)
        requires_grad: Whether to track gradients

    Returns:
        Tensor with data copied from the input

    Examples:
        >>> import metal_native as mn
        >>> t = mn.tensor([1, 2, 3])
        >>> t = mn.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=mn.float32)
        >>> t = mn.tensor(np.array([1, 2, 3]))
    """
    return Tensor(data=data, dtype=dtype, requires_grad=requires_grad)


def randn(*shape: int, dtype: Optional[dtypes.DType] = None, requires_grad: bool = False) -> Tensor:
    """Create a tensor filled with random values from a standard normal distribution.

    Args:
        *shape: Shape dimensions (variable number of integers)
        dtype: Data type (default: float32)
        requires_grad: Whether to track gradients

    Returns:
        Tensor filled with random normal values (mean=0, std=1)

    Examples:
        >>> import metal_native as mn
        >>> t = mn.randn(3, 3)
        >>> t = mn.randn(2, 3, 4, dtype=mn.float16)
    """
    if dtype is None:
        dtype = dtypes.float32

    # Create random data using NumPy
    shape_tuple = tuple(shape)
    np_data = np.random.randn(*shape_tuple).astype(dtypes.dtype_to_numpy(dtype))

    return Tensor(data=np_data, dtype=dtype, requires_grad=requires_grad)


def from_numpy(array: np.ndarray, requires_grad: bool = False) -> Tensor:
    """Create a tensor from a NumPy array.

    Args:
        array: NumPy array
        requires_grad: Whether to track gradients

    Returns:
        Tensor with data copied from the array
    """
    return Tensor(data=array, requires_grad=requires_grad)


def from_torch(tensor: 'torch.Tensor', requires_grad: Optional[bool] = None) -> Tensor:
    """Create a tensor from a PyTorch tensor via DLPack (zero-copy when possible).

    Args:
        tensor: PyTorch tensor
        requires_grad: Whether to track gradients (default: inherit from PyTorch tensor)

    Returns:
        MetalNative tensor

    Raises:
        ImportError: If PyTorch is not installed
    """
    try:
        import torch.utils.dlpack
    except ImportError:
        raise ImportError("PyTorch is required for from_torch()")

    if requires_grad is None:
        requires_grad = tensor.requires_grad

    # Move to CPU if needed (DLPack Metal support may be limited)
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()

    from metal_native import _C
    handle = _C.tensor_from_dlpack(torch.utils.dlpack.to_dlpack(tensor), requires_grad)
    return Tensor(_native_handle=handle, requires_grad=requires_grad)


def from_dlpack(capsule: Any) -> Tensor:
    """Create a tensor from a DLPack capsule.

    Args:
        capsule: DLPack capsule object

    Returns:
        MetalNative tensor
    """
    from metal_native import _C
    handle = _C.tensor_from_dlpack(capsule, False)
    return Tensor(_native_handle=handle)


# PyTorch function override registry
_TORCH_FUNCTION_OVERRIDES: dict = {}


def register_torch_override(torch_func: Callable) -> Callable:
    """Decorator to register a PyTorch function override.

    Example:
        @register_torch_override(torch.nn.functional.relu)
        def metal_relu(input):
            return input.relu()

    Args:
        torch_func: PyTorch function to override

    Returns:
        Decorator function
    """
    def decorator(impl: Callable) -> Callable:
        _TORCH_FUNCTION_OVERRIDES[torch_func] = impl
        return impl
    return decorator
