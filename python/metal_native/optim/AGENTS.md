<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# optim

## Purpose
PyTorch-compatible optimizers for training neural networks with MetalNative tensors. Provides CPU-side parameter updates using NumPy with Adam/AdamW and SGD implementations.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Package exports: MetalAdam, MetalAdamW, MetalSGD |
| `adam.py` | Adam optimizer with AMSGrad support and AdamW with decoupled weight decay |
| `sgd.py` | SGD with momentum, dampening, Nesterov acceleration, and weight decay |

## For AI Agents

### Working In This Directory
- **NumPy-Based**: All optimizers use NumPy for parameter updates (CPU-side) via `tensor.numpy()`
- **State Management**: Each optimizer maintains per-parameter state (momentum buffers, running averages)
- **PyTorch API**: Follows `torch.optim` conventions (zero_grad, step, state_dict, load_state_dict)
- **Bias Correction**: Adam/AdamW apply bias correction to moments using `_step_count`

### Testing Requirements
- Test convergence on simple problems (quadratic loss, linear regression)
- Verify state persistence via `state_dict()` / `load_state_dict()`
- Test edge cases: zero gradients, negative learning rates, invalid beta values
- Compare numerical outputs against PyTorch optimizers on same problem

### Common Patterns
- **Initialization**: Validate hyperparameters in `__init__` (lr >= 0, 0 <= betas < 1, etc.)
- **Gradient Check**: Skip parameters with `None` gradients in `step()`
- **State Init**: Lazy-create momentum/running average buffers on first `step()` call
- **Parameter Update**: Convert to NumPy, compute update, wrap back in Tensor via `param.data = Tensor(...)`
- **Weight Decay**: Adam uses L2 penalty (add to gradient), AdamW uses decoupled decay (multiply param)

## Dependencies

### Internal
- `../tensor.py`: Tensor class for parameter access and gradient retrieval
- `../dtypes.py`: For preserving dtype during parameter updates

### External
- **Required**: NumPy (for all numerical operations)
- **Optional**: PyTorch (for API compatibility testing only)

<!-- MANUAL: -->
