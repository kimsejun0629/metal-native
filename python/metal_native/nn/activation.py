"""Activation function modules for MetalNative."""

from ..tensor import Tensor


class ReLU:
    """Applies ReLU activation: max(0, x).

    Uses Metal clamp kernel for optimal performance.
    """

    def __call__(self, x: Tensor) -> Tensor:
        return x.relu()

    def __repr__(self) -> str:
        return "ReLU()"


class GELU:
    """Applies Gaussian Error Linear Unit activation.

    Uses fused Metal kernel when available, falls back to
    x * 0.5 * (1 + erf(x / sqrt(2))) approximation.
    """

    def __call__(self, x: Tensor) -> Tensor:
        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # Use the simpler sigmoid approximation: x * sigmoid(1.702 * x)
        # For now, use NumPy fallback for correctness
        import numpy as np
        from ..tensor import Tensor as T, from_numpy
        x_np = x.numpy()
        result = x_np * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x_np + 0.044715 * x_np ** 3)))
        return from_numpy(result.astype(x_np.dtype))

    def __repr__(self) -> str:
        return "GELU()"


class SiLU:
    """Applies SiLU (Swish) activation: x * sigmoid(x).

    For paired gate/up projections, prefer fast.swiglu() which is fused.
    """

    def __call__(self, x: Tensor) -> Tensor:
        import numpy as np
        from ..tensor import from_numpy
        x_np = x.numpy()
        result = x_np / (1.0 + np.exp(-x_np))
        return from_numpy(result.astype(x_np.dtype))

    def __repr__(self) -> str:
        return "SiLU()"


class Softmax:
    """Applies softmax along a dimension.

    Uses optimized Metal softmax kernel with SIMD-group reductions.

    Args:
        dim: Dimension along which to apply softmax. Default: -1.
    """

    def __init__(self, dim: int = -1):
        self.dim = dim

    def __call__(self, x: Tensor) -> Tensor:
        return x.softmax(dim=self.dim)

    def __repr__(self) -> str:
        return f"Softmax(dim={self.dim})"
