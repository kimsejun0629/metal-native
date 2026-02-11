"""SGD optimizer for MetalNative.

This module implements stochastic gradient descent (SGD) with momentum
and weight decay, compatible with the PyTorch optimizer API.
"""

from typing import List, Optional, Callable, Union
import numpy as np

__all__ = ['MetalSGD']


class MetalSGD:
    """Stochastic Gradient Descent optimizer for MetalNative tensors.

    Implements SGD with optional momentum, dampening, weight decay, and Nesterov momentum.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate
        momentum: Momentum factor (default: 0)
        dampening: Dampening for momentum (default: 0)
        weight_decay: L2 penalty (default: 0)
        nesterov: Whether to use Nesterov momentum (default: False)

    Example:
        >>> import metal_native as mn
        >>> model = create_model()
        >>> optimizer = mn.optim.MetalSGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> for inputs, targets in dataloader:
        >>>     loss = model(inputs, targets)
        >>>     optimizer.zero_grad()
        >>>     loss.backward()
        >>>     optimizer.step()

    Note:
        When using momentum, the optimizer maintains a velocity buffer for each parameter.
    """

    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        """Initialize SGD optimizer."""
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov

        # Convert params to list
        self.param_groups = []
        if isinstance(params, (list, tuple)):
            self.param_groups = [{'params': list(params)}]
        else:
            self.param_groups = [{'params': list(params)}]

        # State for momentum buffers
        self.state = {}

    def zero_grad(self) -> None:
        """Clear gradients of all optimized parameters."""
        for group in self.param_groups:
            for param in group['params']:
                if hasattr(param, '_grad') and param._grad is not None:
                    param._grad = None

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)

        Returns:
            Loss value if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                # Skip parameters without gradients
                if not hasattr(param, 'grad') or param.grad is None:
                    continue

                grad = param.grad

                # Get gradient as numpy array
                grad_np = grad.numpy() if hasattr(grad, 'numpy') else grad

                # Apply weight decay (L2 regularization)
                if self.weight_decay != 0:
                    param_np = param.numpy() if hasattr(param, 'numpy') else param
                    grad_np = grad_np + self.weight_decay * param_np

                # Apply momentum
                if self.momentum != 0:
                    # Initialize momentum buffer if needed
                    if param not in self.state:
                        self.state[param] = {
                            'momentum_buffer': np.zeros_like(grad_np)
                        }

                    buf = self.state[param]['momentum_buffer']

                    # Update momentum buffer
                    buf = self.momentum * buf + (1 - self.dampening) * grad_np
                    self.state[param]['momentum_buffer'] = buf

                    # Apply Nesterov momentum if enabled
                    if self.nesterov:
                        grad_np = grad_np + self.momentum * buf
                    else:
                        grad_np = buf

                # Apply update to parameter
                param_np = param.numpy() if hasattr(param, 'numpy') else param
                param_np = param_np - self.lr * grad_np

                # Update parameter data
                if hasattr(param, 'data'):
                    from ..tensor import Tensor
                    param.data = Tensor(data=param_np, dtype=param.dtype)
                else:
                    # Assume it's a numpy array
                    param[:] = param_np

        return loss

    def state_dict(self) -> dict:
        """Get optimizer state as a dictionary.

        Returns:
            Dictionary containing optimizer state
        """
        return {
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load optimizer state from a dictionary.

        Args:
            state_dict: Dictionary containing optimizer state
        """
        self.state = state_dict['state']
        self.param_groups = state_dict['param_groups']
