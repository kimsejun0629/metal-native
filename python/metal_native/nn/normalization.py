"""Normalization layers for MetalNative."""

from typing import Optional
import numpy as np
from ..tensor import Tensor, from_numpy
from .. import dtypes


class RMSNorm:
    """Root Mean Square Layer Normalization.

    Uses fused Metal kernel via fast.rms_norm for optimal performance.

    Args:
        normalized_shape: Input shape from an expected input (last dimension).
        eps: Numerical stability constant. Default: 1e-6.
        dtype: Data type for parameters. Default: float32.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6,
                 dtype: Optional[dtypes.DType] = None):
        self.normalized_shape = normalized_shape
        self.eps = eps

        if dtype is None:
            dtype = dtypes.float32

        self.weight = from_numpy(np.ones(normalized_shape,
                                         dtype=dtypes.dtype_to_numpy(dtype)))

    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass using fused Metal RMSNorm kernel."""
        from metal_native import _C
        return Tensor(_native_handle=_C.fast.rms_norm(
            x._handle, self.weight._handle, self.eps))

    def __repr__(self) -> str:
        return f"RMSNorm({self.normalized_shape}, eps={self.eps})"


class LayerNorm:
    """Layer Normalization.

    Uses fused Metal kernel via fast.layer_norm for optimal performance.

    Args:
        normalized_shape: Input shape from an expected input (last dimension).
        eps: Numerical stability constant. Default: 1e-5.
        dtype: Data type for parameters. Default: float32.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5,
                 dtype: Optional[dtypes.DType] = None):
        self.normalized_shape = normalized_shape
        self.eps = eps

        if dtype is None:
            dtype = dtypes.float32

        np_dtype = dtypes.dtype_to_numpy(dtype)
        self.weight = from_numpy(np.ones(normalized_shape, dtype=np_dtype))
        self.bias = from_numpy(np.zeros(normalized_shape, dtype=np_dtype))

    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass using fused Metal LayerNorm kernel."""
        from metal_native import _C
        return Tensor(_native_handle=_C.fast.layer_norm(
            x._handle, self.weight._handle, self.bias._handle, self.eps))

    def __repr__(self) -> str:
        return f"LayerNorm({self.normalized_shape}, eps={self.eps})"
