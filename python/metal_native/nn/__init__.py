"""Neural network modules for MetalNative.

Provides PyTorch-compatible neural network layers optimized for Apple Silicon.
"""

from .linear import Linear
from .normalization import RMSNorm, LayerNorm
from .activation import ReLU, GELU, SiLU, Softmax

__all__ = [
    'Linear',
    'RMSNorm',
    'LayerNorm',
    'ReLU',
    'GELU',
    'SiLU',
    'Softmax',
]
