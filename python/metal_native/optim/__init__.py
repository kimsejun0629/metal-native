"""Optimizers for MetalNative.

This package provides PyTorch-compatible optimizers for training
neural networks with MetalNative tensors.

Available optimizers:
    - MetalAdam: Adam optimizer with optional AMSGrad
    - MetalAdamW: AdamW optimizer with decoupled weight decay
    - MetalSGD: SGD with momentum and Nesterov acceleration
"""

from .adam import MetalAdam, MetalAdamW
from .sgd import MetalSGD

__all__ = [
    'MetalAdam',
    'MetalAdamW',
    'MetalSGD',
]
