"""Shared pytest fixtures and configuration for MetalNative tests."""

import pytest

# Check if C extension is available
_c_available = False
try:
    import metal_native._C
    _c_available = True
except ImportError:
    pass

# Mark to skip tests that require C extension
skip_no_c_ext = pytest.mark.skipif(
    not _c_available,
    reason="C extension not loaded"
)


@pytest.fixture
def c_extension_available():
    """Fixture that returns whether C extension is available."""
    return _c_available
