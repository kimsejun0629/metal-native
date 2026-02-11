"""Tests for optimizer modules."""

import pytest
import inspect
from unittest.mock import Mock, patch
from metal_native.optim import MetalAdam, MetalAdamW, MetalSGD


def test_optim_module_exports():
    """Test optim module __all__ exports."""
    from metal_native import optim
    expected = ['MetalAdam', 'MetalAdamW', 'MetalSGD']
    assert set(optim.__all__) == set(expected)


def test_adam_class_exists():
    """Test MetalAdam class is defined and importable."""
    assert MetalAdam is not None
    assert callable(MetalAdam)


def test_adamw_class_exists():
    """Test MetalAdamW class is defined and importable."""
    assert MetalAdamW is not None
    assert callable(MetalAdamW)


def test_sgd_class_exists():
    """Test MetalSGD class is defined and importable."""
    assert MetalSGD is not None
    assert callable(MetalSGD)


def test_adam_signature():
    """Test MetalAdam constructor signature."""
    sig = inspect.signature(MetalAdam.__init__)
    params = sig.parameters

    assert 'params' in params
    assert 'lr' in params
    assert 'betas' in params
    assert 'eps' in params
    assert 'weight_decay' in params
    assert 'amsgrad' in params

    # Check defaults
    assert params['lr'].default == 1e-3
    assert params['betas'].default == (0.9, 0.999)
    assert params['eps'].default == 1e-8
    assert params['weight_decay'].default == 0.0
    assert params['amsgrad'].default is False


def test_adamw_signature():
    """Test MetalAdamW inherits from MetalAdam."""
    assert issubclass(MetalAdamW, MetalAdam)

    # MetalAdamW uses parent __init__
    sig = inspect.signature(MetalAdamW.__init__)
    params = sig.parameters

    assert 'params' in params
    assert 'lr' in params
    assert 'betas' in params
    assert 'eps' in params
    assert 'weight_decay' in params


def test_sgd_signature():
    """Test MetalSGD constructor signature."""
    sig = inspect.signature(MetalSGD.__init__)
    params = sig.parameters

    assert 'params' in params
    assert 'lr' in params
    assert 'momentum' in params
    assert 'dampening' in params
    assert 'weight_decay' in params
    assert 'nesterov' in params

    # Check defaults (lr has no default)
    assert params['momentum'].default == 0.0
    assert params['dampening'].default == 0.0
    assert params['weight_decay'].default == 0.0
    assert params['nesterov'].default is False


def test_adam_init_basic():
    """Test MetalAdam can be instantiated with mock params."""
    mock_params = [Mock(), Mock()]
    optimizer = MetalAdam(mock_params)

    assert optimizer.lr == 1e-3
    assert optimizer.betas == (0.9, 0.999)
    assert optimizer.eps == 1e-8
    assert optimizer.weight_decay == 0.0
    assert optimizer.amsgrad is False
    assert len(optimizer.param_groups) == 1


def test_adam_init_custom_params():
    """Test MetalAdam with custom hyperparameters."""
    mock_params = [Mock()]
    optimizer = MetalAdam(
        mock_params,
        lr=0.01,
        betas=(0.95, 0.999),
        eps=1e-7,
        weight_decay=0.01,
        amsgrad=True
    )

    assert optimizer.lr == 0.01
    assert optimizer.betas == (0.95, 0.999)
    assert optimizer.eps == 1e-7
    assert optimizer.weight_decay == 0.01
    assert optimizer.amsgrad is True


def test_adam_invalid_lr():
    """Test MetalAdam raises ValueError for negative lr."""
    with pytest.raises(ValueError, match="Invalid learning rate"):
        MetalAdam([Mock()], lr=-0.1)


def test_adam_invalid_eps():
    """Test MetalAdam raises ValueError for negative eps."""
    with pytest.raises(ValueError, match="Invalid epsilon value"):
        MetalAdam([Mock()], eps=-1e-8)


def test_adam_invalid_beta():
    """Test MetalAdam raises ValueError for invalid beta values."""
    with pytest.raises(ValueError, match="Invalid beta parameter"):
        MetalAdam([Mock()], betas=(1.1, 0.999))

    with pytest.raises(ValueError, match="Invalid beta parameter"):
        MetalAdam([Mock()], betas=(0.9, -0.1))


def test_adam_invalid_weight_decay():
    """Test MetalAdam raises ValueError for negative weight_decay."""
    with pytest.raises(ValueError, match="Invalid weight_decay value"):
        MetalAdam([Mock()], weight_decay=-0.1)


def test_adamw_init():
    """Test MetalAdamW can be instantiated."""
    mock_params = [Mock()]
    optimizer = MetalAdamW(mock_params, lr=1e-3, weight_decay=0.01)

    assert optimizer.lr == 1e-3
    assert optimizer.weight_decay == 0.01


def test_sgd_init_basic():
    """Test MetalSGD can be instantiated."""
    mock_params = [Mock()]
    optimizer = MetalSGD(mock_params, lr=0.1)

    assert optimizer.lr == 0.1
    assert optimizer.momentum == 0.0
    assert optimizer.dampening == 0.0
    assert optimizer.weight_decay == 0.0
    assert optimizer.nesterov is False


def test_sgd_init_with_momentum():
    """Test MetalSGD with momentum."""
    mock_params = [Mock()]
    optimizer = MetalSGD(mock_params, lr=0.1, momentum=0.9)

    assert optimizer.momentum == 0.9


def test_sgd_invalid_lr():
    """Test MetalSGD raises ValueError for negative lr."""
    with pytest.raises(ValueError, match="Invalid learning rate"):
        MetalSGD([Mock()], lr=-0.1)


def test_sgd_invalid_momentum():
    """Test MetalSGD raises ValueError for negative momentum."""
    with pytest.raises(ValueError, match="Invalid momentum value"):
        MetalSGD([Mock()], lr=0.1, momentum=-0.1)


def test_sgd_invalid_weight_decay():
    """Test MetalSGD raises ValueError for negative weight_decay."""
    with pytest.raises(ValueError, match="Invalid weight_decay value"):
        MetalSGD([Mock()], lr=0.1, weight_decay=-0.1)


def test_sgd_nesterov_requires_momentum():
    """Test MetalSGD raises ValueError for Nesterov without momentum."""
    with pytest.raises(ValueError, match="Nesterov momentum requires"):
        MetalSGD([Mock()], lr=0.1, nesterov=True)


def test_sgd_nesterov_requires_zero_dampening():
    """Test MetalSGD raises ValueError for Nesterov with dampening."""
    with pytest.raises(ValueError, match="Nesterov momentum requires"):
        MetalSGD([Mock()], lr=0.1, momentum=0.9, dampening=0.1, nesterov=True)


def test_adam_has_methods():
    """Test MetalAdam has required methods."""
    mock_params = [Mock()]
    optimizer = MetalAdam(mock_params)

    assert hasattr(optimizer, 'zero_grad')
    assert hasattr(optimizer, 'step')
    assert hasattr(optimizer, 'state_dict')
    assert hasattr(optimizer, 'load_state_dict')
    assert callable(optimizer.zero_grad)
    assert callable(optimizer.step)
    assert callable(optimizer.state_dict)
    assert callable(optimizer.load_state_dict)


def test_sgd_has_methods():
    """Test MetalSGD has required methods."""
    mock_params = [Mock()]
    optimizer = MetalSGD(mock_params, lr=0.1)

    assert hasattr(optimizer, 'zero_grad')
    assert hasattr(optimizer, 'step')
    assert hasattr(optimizer, 'state_dict')
    assert hasattr(optimizer, 'load_state_dict')
    assert callable(optimizer.zero_grad)
    assert callable(optimizer.step)
    assert callable(optimizer.state_dict)
    assert callable(optimizer.load_state_dict)


def test_adam_state_dict():
    """Test MetalAdam state_dict returns dict."""
    mock_params = [Mock()]
    optimizer = MetalAdam(mock_params)
    state = optimizer.state_dict()

    assert isinstance(state, dict)
    assert 'state' in state
    assert 'param_groups' in state
    assert 'step_count' in state


def test_sgd_state_dict():
    """Test MetalSGD state_dict returns dict."""
    mock_params = [Mock()]
    optimizer = MetalSGD(mock_params, lr=0.1)
    state = optimizer.state_dict()

    assert isinstance(state, dict)
    assert 'state' in state
    assert 'param_groups' in state


def test_adam_docstring():
    """Test MetalAdam has docstring."""
    assert MetalAdam.__doc__ is not None
    assert 'Adam optimizer' in MetalAdam.__doc__


def test_adamw_docstring():
    """Test MetalAdamW has docstring."""
    assert MetalAdamW.__doc__ is not None
    assert 'AdamW optimizer' in MetalAdamW.__doc__


def test_sgd_docstring():
    """Test MetalSGD has docstring."""
    assert MetalSGD.__doc__ is not None
    assert 'SGD' in MetalSGD.__doc__ or 'Stochastic Gradient Descent' in MetalSGD.__doc__
