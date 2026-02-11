"""Tests for config module."""

import json
import os
import tempfile
import pytest
from metal_native.config import (
    AllocatorConfig,
    BackpressureConfig,
    DispatchConfig,
    KernelCacheConfig,
    ProfilingConfig,
    MetalNativeConfig,
    get_config,
    set_config,
)


def test_allocator_config_defaults():
    """Test AllocatorConfig default values."""
    config = AllocatorConfig()
    assert config.initial_heap_size == 256 * 1024 * 1024  # 256 MB
    assert config.max_heap_size == 8 * 1024 * 1024 * 1024  # 8 GB
    assert config.growth_policy == 'exponential'
    assert config.growth_factor == 2.0
    assert config.growth_increment == 256 * 1024 * 1024  # 256 MB
    assert config.enable_pooling is True
    assert config.small_buffer_threshold == 1024 * 1024  # 1 MB


def test_backpressure_config_defaults():
    """Test BackpressureConfig default values."""
    config = BackpressureConfig()
    assert config.soft_limit == 64
    assert config.hard_limit == 128
    assert config.throttle_poll_us == 100
    assert config.enable_adaptive is True


def test_dispatch_config_defaults():
    """Test DispatchConfig default values."""
    config = DispatchConfig()
    assert config.use_commit_and_continue is True
    assert config.max_commands_per_buffer == 32
    assert config.release_gil is True
    assert config.enable_concurrent_dispatch is True
    assert config.command_buffer_pool_size == 8


def test_kernel_cache_config_defaults():
    """Test KernelCacheConfig default values."""
    config = KernelCacheConfig()
    assert config.cache_dir is None
    assert config.max_cache_size == 1024 * 1024 * 1024  # 1 GB
    assert config.enable_bucketing is True
    assert config.bucket_sizes == [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    assert config.enable_persistent_cache is True
    assert config.cache_version == "0.1.0"


def test_profiling_config_defaults():
    """Test ProfilingConfig default values."""
    config = ProfilingConfig()
    assert config.enable_signposts is False
    assert config.enable_capture is False
    assert config.track_callstacks is False
    assert config.log_kernel_launches is False
    assert config.detailed_memory_stats is False


def test_metal_native_config_defaults():
    """Test MetalNativeConfig creates with default subconfigs."""
    config = MetalNativeConfig()
    assert isinstance(config.allocator, AllocatorConfig)
    assert isinstance(config.backpressure, BackpressureConfig)
    assert isinstance(config.dispatch, DispatchConfig)
    assert isinstance(config.kernel_cache, KernelCacheConfig)
    assert isinstance(config.profiling, ProfilingConfig)


def test_config_to_dict():
    """Test MetalNativeConfig.to_dict() conversion."""
    config = MetalNativeConfig()
    data = config.to_dict()

    assert isinstance(data, dict)
    assert 'allocator' in data
    assert 'backpressure' in data
    assert 'dispatch' in data
    assert 'kernel_cache' in data
    assert 'profiling' in data

    assert data['allocator']['initial_heap_size'] == 256 * 1024 * 1024
    assert data['backpressure']['soft_limit'] == 64
    assert data['dispatch']['release_gil'] is True
    assert data['kernel_cache']['enable_bucketing'] is True
    assert data['profiling']['enable_signposts'] is False


def test_config_to_dict_round_trip():
    """Test to_dict() -> from_dict() round trip."""
    config1 = MetalNativeConfig()
    data = config1.to_dict()
    config2 = MetalNativeConfig.from_dict(data)

    assert config2.allocator.initial_heap_size == config1.allocator.initial_heap_size
    assert config2.backpressure.soft_limit == config1.backpressure.soft_limit
    assert config2.dispatch.release_gil == config1.dispatch.release_gil
    assert config2.kernel_cache.cache_version == config1.kernel_cache.cache_version
    assert config2.profiling.enable_signposts == config1.profiling.enable_signposts


def test_config_from_dict_partial():
    """Test from_dict() with partial data uses defaults."""
    data = {
        'allocator': {'max_heap_size': 1024 * 1024 * 1024},  # 1 GB
        'profiling': {'enable_signposts': True},
    }
    config = MetalNativeConfig.from_dict(data)

    # Overridden values
    assert config.allocator.max_heap_size == 1024 * 1024 * 1024
    assert config.profiling.enable_signposts is True

    # Default values for other fields
    assert config.allocator.initial_heap_size == 256 * 1024 * 1024
    assert config.backpressure.soft_limit == 64
    assert config.dispatch.release_gil is True


def test_config_from_dict_empty():
    """Test from_dict() with empty dict uses all defaults."""
    config = MetalNativeConfig.from_dict({})

    assert config.allocator.initial_heap_size == 256 * 1024 * 1024
    assert config.backpressure.soft_limit == 64
    assert config.dispatch.release_gil is True
    assert config.kernel_cache.enable_bucketing is True
    assert config.profiling.enable_signposts is False


def test_config_from_json():
    """Test from_json() loads from file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        data = {
            'allocator': {'max_heap_size': 2 * 1024 * 1024 * 1024},
            'profiling': {'enable_capture': True},
        }
        json.dump(data, f)
        json_path = f.name

    try:
        config = MetalNativeConfig.from_json(json_path)
        assert config.allocator.max_heap_size == 2 * 1024 * 1024 * 1024
        assert config.profiling.enable_capture is True
        assert config.backpressure.soft_limit == 64  # Default
    finally:
        os.unlink(json_path)


def test_config_from_env():
    """Test from_env() loads from environment variables."""
    env_vars = {
        'METAL_NATIVE_ALLOCATOR_MAX_HEAP_SIZE': '4294967296',  # 4 GB
        'METAL_NATIVE_BACKPRESSURE_SOFT_LIMIT': '32',
        'METAL_NATIVE_DISPATCH_RELEASE_GIL': 'false',
        'METAL_NATIVE_PROFILING_ENABLE_SIGNPOSTS': 'true',
        # Note: multi-word section names (kernel_cache) aren't supported by from_env
    }

    # Save original env
    original_env = {}
    for key in env_vars:
        original_env[key] = os.environ.get(key)

    try:
        # Set test env vars
        for key, value in env_vars.items():
            os.environ[key] = value

        config = MetalNativeConfig.from_env()

        assert config.allocator.max_heap_size == 4294967296
        assert config.backpressure.soft_limit == 32
        assert config.dispatch.release_gil is False
        assert config.profiling.enable_signposts is True

        # Unset vars should use defaults
        assert config.allocator.initial_heap_size == 256 * 1024 * 1024

    finally:
        # Restore original env
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_get_config_returns_singleton():
    """Test get_config() returns the same instance."""
    config1 = get_config()
    config2 = get_config()
    assert config1 is config2


def test_set_config_replaces_global():
    """Test set_config() replaces global configuration."""
    original = get_config()

    new_config = MetalNativeConfig()
    new_config.allocator.max_heap_size = 123456789

    set_config(new_config)

    retrieved = get_config()
    assert retrieved is new_config
    assert retrieved.allocator.max_heap_size == 123456789

    # Restore original
    set_config(original)


def test_config_custom_values():
    """Test creating configs with custom values."""
    allocator = AllocatorConfig(
        initial_heap_size=512 * 1024 * 1024,
        growth_policy='linear',
        enable_pooling=False,
    )

    assert allocator.initial_heap_size == 512 * 1024 * 1024
    assert allocator.growth_policy == 'linear'
    assert allocator.enable_pooling is False
    assert allocator.max_heap_size == 8 * 1024 * 1024 * 1024  # Default


def test_module_all_exports():
    """Test that __all__ contains expected exports."""
    from metal_native import config
    expected = [
        'AllocatorConfig',
        'BackpressureConfig',
        'DispatchConfig',
        'KernelCacheConfig',
        'ProfilingConfig',
        'MetalNativeConfig',
    ]
    assert set(config.__all__) == set(expected)
