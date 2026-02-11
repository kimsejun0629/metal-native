"""Tests for version module."""

import metal_native._version as version


def test_version_is_string():
    """Test that __version__ is a string."""
    assert isinstance(version.__version__, str)


def test_version_value():
    """Test that __version__ matches expected value."""
    assert version.__version__ == "0.1.0"


def test_version_info_is_tuple():
    """Test that __version_info__ is a tuple."""
    assert isinstance(version.__version_info__, tuple)


def test_version_info_value():
    """Test that __version_info__ matches expected value."""
    assert version.__version_info__ == (0, 1, 0)


def test_version_info_length():
    """Test that __version_info__ has 3 components."""
    assert len(version.__version_info__) == 3


def test_version_info_all_integers():
    """Test that all components of __version_info__ are integers."""
    assert all(isinstance(x, int) for x in version.__version_info__)
