"""Integration test configuration."""
import pytest


def _has_metal():
    """Check if Metal backend is available."""
    try:
        from metal_native import _C
        return _C is not None
    except ImportError:
        return False


requires_metal = pytest.mark.skipif(
    not _has_metal(),
    reason="Metal backend not available"
)
