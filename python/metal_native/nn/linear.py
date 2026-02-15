"""Linear (fully-connected) layer for MetalNative."""

from typing import Optional
import numpy as np
from ..tensor import Tensor, from_numpy
from .. import dtypes


class Linear:
    """Applies a linear transformation: y = x @ W^T + b.

    PyTorch-compatible API.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If True, adds a learnable bias. Default: True.
        dtype: Data type for parameters. Default: float32.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, dtype: Optional[dtypes.DType] = None):
        self.in_features = in_features
        self.out_features = out_features

        if dtype is None:
            dtype = dtypes.float32

        # Xavier uniform initialization
        limit = np.sqrt(6.0 / (in_features + out_features))
        w_np = np.random.uniform(-limit, limit,
                                 (out_features, in_features)).astype(
                                     dtypes.dtype_to_numpy(dtype))
        self.weight = from_numpy(w_np)

        if bias:
            self.bias = from_numpy(np.zeros(out_features,
                                            dtype=dtypes.dtype_to_numpy(dtype)))
        else:
            self.bias = None

    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass: y = x @ W^T + b."""
        from metal_native import _C

        # x: [..., in_features], weight: [out_features, in_features]
        # result = x @ weight^T = matmul(x, weight, transpose_b=True)
        output = Tensor(_native_handle=_C.matmul(
            x._handle, self.weight._handle,
            False, True))

        if self.bias is not None:
            output = output + self.bias

        return output

    def __repr__(self) -> str:
        return (f"Linear(in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None})")
