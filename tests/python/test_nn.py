"""Tests for nn module."""

import pytest


def test_nn_module_exists():
    """Test nn module is importable."""
    from metal_native import nn
    assert nn is not None


def test_nn_module_all_empty():
    """Test nn module __all__ is empty (Phase 4 planned)."""
    from metal_native import nn
    assert hasattr(nn, '__all__')
    assert nn.__all__ == []


def test_nn_module_docstring():
    """Test nn module has a docstring."""
    from metal_native import nn
    assert nn.__doc__ is not None
    assert 'neural network' in nn.__doc__.lower() or 'nn' in nn.__doc__.lower()


def test_nn_module_is_package():
    """Test nn is a package (has __path__)."""
    from metal_native import nn
    assert hasattr(nn, '__path__')


def test_nn_module_no_unexpected_exports():
    """Test nn module has no unexpected public exports (Phase 4 not implemented yet)."""
    from metal_native import nn
    # Get all public attributes (not starting with _)
    public_attrs = [name for name in dir(nn) if not name.startswith('_')]

    # Should be minimal since implementation is planned for Phase 4
    # Only __all__ should be present as a public attribute
    expected_public = []  # __all__ is in dir() but we filter it in this test

    # Filter out __all__ itself from the check
    public_attrs_no_all = [name for name in public_attrs if name != '__all__']

    # There should be very few or no public exports yet
    assert len(public_attrs_no_all) == 0, f"Unexpected public exports: {public_attrs_no_all}"
