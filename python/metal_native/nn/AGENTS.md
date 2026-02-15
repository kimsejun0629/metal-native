<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# nn

## Purpose
Placeholder for PyTorch-compatible neural network layers, loss functions, and building blocks. Implementation planned for Phase 4.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Empty module stub with docstring and empty `__all__` |

## For AI Agents

### Working In This Directory
- **Status**: Not yet implemented - module exists for API structure only
- **Future**: Will contain Metal-optimized layers (Linear, Conv2d, BatchNorm, etc.)
- **Imports**: Currently safe to import but provides no functionality

### Testing Requirements
- No tests needed until implementation begins
- Future tests should verify layer outputs match PyTorch reference implementations

### Common Patterns
- Follow PyTorch's `torch.nn` API conventions when implementing
- Use `metal_native.Tensor` as base type for all operations
- Implement both forward pass and backward pass (autograd) support

## Dependencies

### Internal
- `../tensor.py`: Will use Tensor class for all operations
- `../dtypes.py`: For layer parameter initialization with correct dtypes

### External
- **Planned**: PyTorch (for API compatibility verification)

<!-- MANUAL: -->
