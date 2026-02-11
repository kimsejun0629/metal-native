"""Adam and AdamW optimizers for MetalNative.

This module implements the Adam and AdamW optimizers with the same API
as PyTorch's torch.optim.Adam and torch.optim.AdamW.

Reference:
    Adam: https://arxiv.org/abs/1412.6980
    AdamW: https://arxiv.org/abs/1711.05101
"""

from typing import List, Optional, Callable, Tuple, Union
import math
import numpy as np

__all__ = ['MetalAdam', 'MetalAdamW']


class MetalAdam:
    """Adam optimizer for MetalNative tensors.

    Implements the Adam algorithm with optional AMSGrad.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: L2 penalty (default: 0)
        amsgrad: Whether to use AMSGrad variant (default: False)

    Example:
        >>> import metal_native as mn
        >>> model = create_model()
        >>> optimizer = mn.optim.MetalAdam(model.parameters(), lr=1e-3)
        >>> for inputs, targets in dataloader:
        >>>     loss = model(inputs, targets)
        >>>     optimizer.zero_grad()
        >>>     loss.backward()
        >>>     optimizer.step()
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        """Initialize Adam optimizer."""
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

        # Convert params to list if it's an iterator
        self.param_groups = []
        if isinstance(params, (list, tuple)):
            self.param_groups = [{'params': list(params)}]
        else:
            # Assume it's an iterator
            self.param_groups = [{'params': list(params)}]

        # Initialize state for each parameter
        self.state = {}
        self._step_count = 0

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

        self._step_count += 1

        for group in self.param_groups:
            beta1, beta2 = self.betas

            for param in group['params']:
                # Skip parameters without gradients
                if not hasattr(param, 'grad') or param.grad is None:
                    continue

                grad = param.grad

                # Initialize state for this parameter if needed
                if param not in self.state:
                    self.state[param] = {
                        'm': np.zeros_like(grad.numpy() if hasattr(grad, 'numpy') else grad),
                        'v': np.zeros_like(grad.numpy() if hasattr(grad, 'numpy') else grad),
                    }
                    if self.amsgrad:
                        self.state[param]['v_max'] = np.zeros_like(
                            grad.numpy() if hasattr(grad, 'numpy') else grad
                        )

                state = self.state[param]

                # Get gradient as numpy array
                grad_np = grad.numpy() if hasattr(grad, 'numpy') else grad

                # Apply weight decay (L2 regularization)
                if self.weight_decay != 0:
                    param_np = param.numpy() if hasattr(param, 'numpy') else param
                    grad_np = grad_np + self.weight_decay * param_np

                # Update biased first moment estimate
                state['m'] = beta1 * state['m'] + (1 - beta1) * grad_np

                # Update biased second moment estimate
                state['v'] = beta2 * state['v'] + (1 - beta2) * (grad_np ** 2)

                # Compute bias-corrected first moment
                m_hat = state['m'] / (1 - beta1 ** self._step_count)

                # Compute bias-corrected second moment
                if self.amsgrad:
                    # Use max of all v_t for more stable updates
                    state['v_max'] = np.maximum(state['v_max'], state['v'])
                    v_hat = state['v_max'] / (1 - beta2 ** self._step_count)
                else:
                    v_hat = state['v'] / (1 - beta2 ** self._step_count)

                # Compute update
                update = -self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

                # Apply update to parameter
                param_np = param.numpy() if hasattr(param, 'numpy') else param
                param_np = param_np + update

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
            'step_count': self._step_count,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load optimizer state from a dictionary.

        Args:
            state_dict: Dictionary containing optimizer state
        """
        self.state = state_dict['state']
        self.param_groups = state_dict['param_groups']
        self._step_count = state_dict['step_count']


class MetalAdamW(MetalAdam):
    """AdamW optimizer for MetalNative tensors.

    AdamW decouples weight decay from the gradient-based update,
    which improves generalization compared to L2 regularization.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
        amsgrad: Whether to use AMSGrad variant (default: False)

    Example:
        >>> import metal_native as mn
        >>> optimizer = mn.optim.MetalAdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    """

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform a single optimization step with decoupled weight decay.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            Loss value if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            beta1, beta2 = self.betas

            for param in group['params']:
                if not hasattr(param, 'grad') or param.grad is None:
                    continue

                grad = param.grad

                # Initialize state
                if param not in self.state:
                    self.state[param] = {
                        'm': np.zeros_like(grad.numpy() if hasattr(grad, 'numpy') else grad),
                        'v': np.zeros_like(grad.numpy() if hasattr(grad, 'numpy') else grad),
                    }
                    if self.amsgrad:
                        self.state[param]['v_max'] = np.zeros_like(
                            grad.numpy() if hasattr(grad, 'numpy') else grad
                        )

                state = self.state[param]
                grad_np = grad.numpy() if hasattr(grad, 'numpy') else grad

                # Update first moment
                state['m'] = beta1 * state['m'] + (1 - beta1) * grad_np

                # Update second moment
                state['v'] = beta2 * state['v'] + (1 - beta2) * (grad_np ** 2)

                # Bias correction
                m_hat = state['m'] / (1 - beta1 ** self._step_count)

                if self.amsgrad:
                    state['v_max'] = np.maximum(state['v_max'], state['v'])
                    v_hat = state['v_max'] / (1 - beta2 ** self._step_count)
                else:
                    v_hat = state['v'] / (1 - beta2 ** self._step_count)

                # Adam update (without weight decay)
                param_np = param.numpy() if hasattr(param, 'numpy') else param
                adam_update = -self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

                # Decoupled weight decay (AdamW)
                if self.weight_decay != 0:
                    param_np = param_np * (1 - self.lr * self.weight_decay)

                # Apply Adam update
                param_np = param_np + adam_update

                # Update parameter
                if hasattr(param, 'data'):
                    from ..tensor import Tensor
                    param.data = Tensor(data=param_np, dtype=param.dtype)
                else:
                    param[:] = param_np

        return loss
