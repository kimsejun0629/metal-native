"""Comprehensive import tests for metal_native package.

This module tests that all public APIs can be imported and that
the package structure is correct.
"""

import pytest
import sys
from unittest.mock import Mock, patch


def test_main_import():
    """Test that metal_native can be imported."""
    import metal_native
    assert metal_native is not None


def test_version_accessible():
    """Test that __version__ is accessible."""
    import metal_native
    assert hasattr(metal_native, '__version__')
    assert isinstance(metal_native.__version__, str)
    assert len(metal_native.__version__) > 0


def test_main_module_has_all():
    """Test that __all__ is defined in main module."""
    import metal_native
    assert hasattr(metal_native, '__all__')
    assert isinstance(metal_native.__all__, list)
    assert len(metal_native.__all__) > 0


def test_core_submodules_import():
    """Test that core submodules can be imported."""
    # These should not raise ImportError
    import metal_native.nn
    import metal_native.optim
    import metal_native.interop
    import metal_native.dlpack_bridge
    import metal_native.profiling

    assert metal_native.nn is not None
    assert metal_native.optim is not None
    assert metal_native.interop is not None
    assert metal_native.dlpack_bridge is not None
    assert metal_native.profiling is not None


def test_utility_submodules_import():
    """Test that utility submodules can be imported."""
    import metal_native.config
    import metal_native.device
    import metal_native.dtypes
    import metal_native.utils
    import metal_native.tensor

    assert metal_native.config is not None
    assert metal_native.device is not None
    assert metal_native.dtypes is not None
    assert metal_native.utils is not None
    assert metal_native.tensor is not None


def test_nn_module_has_all():
    """Test that nn module has __all__ defined."""
    import metal_native.nn
    assert hasattr(metal_native.nn, '__all__')
    assert isinstance(metal_native.nn.__all__, list)


def test_optim_module_has_all():
    """Test that optim module has __all__ defined."""
    import metal_native.optim
    assert hasattr(metal_native.optim, '__all__')
    assert isinstance(metal_native.optim.__all__, list)


def test_tensor_factory_functions():
    """Test that tensor factory functions are importable."""
    import metal_native

    # Mock _C to bypass initialization check
    with patch.object(metal_native, '_C', Mock()):
        with patch.object(metal_native, '_import_error', None):
            from metal_native import zeros, ones, empty, tensor, randn

            assert callable(zeros)
            assert callable(ones)
            assert callable(empty)
            assert callable(tensor)
            assert callable(randn)


def test_interop_functions():
    """Test that interop functions are importable."""
    import metal_native

    with patch.object(metal_native, '_C', Mock()):
        with patch.object(metal_native, '_import_error', None):
            from metal_native import from_numpy, from_torch, from_dlpack

            assert callable(from_numpy)
            assert callable(from_torch)
            assert callable(from_dlpack)


def test_device_functions():
    """Test that device functions are importable."""
    import metal_native

    with patch.object(metal_native, '_C', Mock()):
        with patch.object(metal_native, '_import_error', None):
            from metal_native import device_name, is_available, device_properties, supports_bfloat16

            assert callable(device_name)
            assert callable(is_available)
            assert callable(device_properties)
            assert callable(supports_bfloat16)


def test_memory_functions():
    """Test that memory functions are importable."""
    import metal_native

    with patch.object(metal_native, '_C', Mock()):
        with patch.object(metal_native, '_import_error', None):
            from metal_native import synchronize, empty_cache, memory_allocated, max_memory_allocated, reset_peak_stats

            assert callable(synchronize)
            assert callable(empty_cache)
            assert callable(memory_allocated)
            assert callable(max_memory_allocated)
            assert callable(reset_peak_stats)


def test_random_functions():
    """Test that random functions are importable."""
    import metal_native

    with patch.object(metal_native, '_C', Mock()):
        with patch.object(metal_native, '_import_error', None):
            from metal_native import set_seed, manual_seed

            assert callable(set_seed)
            assert callable(manual_seed)


def test_dtype_imports():
    """Test that dtype constants are importable."""
    import metal_native

    with patch.object(metal_native, '_C', Mock()):
        with patch.object(metal_native, '_import_error', None):
            from metal_native import float32, float16, bfloat16, int64, int32, int16, int8, uint8, bool_

            # All dtypes should be objects (not None)
            assert float32 is not None
            assert float16 is not None
            assert bfloat16 is not None
            assert int64 is not None
            assert int32 is not None
            assert int16 is not None
            assert int8 is not None
            assert uint8 is not None
            assert bool_ is not None


def test_config_imports():
    """Test that config classes are importable."""
    import metal_native

    with patch.object(metal_native, '_C', Mock()):
        with patch.object(metal_native, '_import_error', None):
            from metal_native import MetalNativeConfig, get_config, set_config

            assert MetalNativeConfig is not None
            assert callable(get_config)
            assert callable(set_config)


def test_tensor_class_import():
    """Test that Tensor class is importable."""
    import metal_native

    with patch.object(metal_native, '_C', Mock()):
        with patch.object(metal_native, '_import_error', None):
            from metal_native import Tensor

            assert Tensor is not None
            assert type(Tensor).__name__ in ('type', 'ABCMeta', 'pybind11_type')


def test_all_exports_importable():
    """Test that all items in __all__ can actually be imported."""
    import metal_native

    with patch.object(metal_native, '_C', Mock()):
        with patch.object(metal_native, '_import_error', None):
            failed_imports = []

            for name in metal_native.__all__:
                try:
                    obj = getattr(metal_native, name)
                    assert obj is not None, f"{name} is None"
                except Exception as e:
                    failed_imports.append((name, str(e)))

            if failed_imports:
                msg = "\n".join([f"  - {name}: {err}" for name, err in failed_imports])
                pytest.fail(f"Failed to import items from __all__:\n{msg}")


def test_no_star_import_pollution():
    """Test that 'from metal_native import *' doesn't pollute namespace excessively."""
    import metal_native

    with patch.object(metal_native, '_C', Mock()):
        with patch.object(metal_native, '_import_error', None):
            # Save original modules
            original_modules = set(sys.modules.keys())

            # Create a clean namespace
            namespace = {}
            exec("from metal_native import *", namespace)

            # Remove builtins
            namespace = {k: v for k, v in namespace.items() if not k.startswith('__')}

            # Should have imported items but not internal modules
            assert len(namespace) > 0, "No items imported with star import"

            # Check that all imported items are in __all__
            for name in namespace:
                assert name in metal_native.__all__, f"{name} not in __all__ but was imported with *"


def test_submodule_independence():
    """Test that submodules can be imported independently."""
    # Clear any cached imports
    modules_to_clear = [m for m in sys.modules if m.startswith('metal_native')]
    for mod in modules_to_clear:
        if mod != 'metal_native':  # Keep main module
            del sys.modules[mod]

    # Import submodules independently
    import metal_native.nn
    import metal_native.optim

    # Should not raise errors
    assert metal_native.nn is not None
    assert metal_native.optim is not None


def test_c_extension_available():
    """Test that C extension status is consistent."""
    import metal_native

    if metal_native._C is not None:
        # C extension loaded successfully
        assert metal_native._import_error is None
    else:
        # C extension not available - this is OK during development
        # _import_error may or may not be set depending on how _C failed
        pass  # No assertion needed - both None states are valid


def test_lazy_import_mechanism():
    """Test that lazy imports work correctly."""
    import metal_native

    # Access a lazily-loaded attribute
    try:
        tensor_class = metal_native.Tensor
        assert tensor_class is not None
    except RuntimeError as e:
        # OK if C extension not available
        assert 'macOS' in str(e) or 'Apple Silicon' in str(e)


def test_module_docstring():
    """Test that main module has docstring."""
    import metal_native
    assert metal_native.__doc__ is not None
    assert len(metal_native.__doc__) > 0
    assert 'MetalNative' in metal_native.__doc__


def test_version_consistency():
    """Test that version is consistent across module and _version."""
    import metal_native
    from metal_native import _version

    assert metal_native.__version__ == _version.__version__


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
