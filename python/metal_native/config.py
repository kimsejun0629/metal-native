"""Configuration classes for MetalNative runtime behavior.

This module provides dataclasses for configuring various aspects of the
MetalNative runtime, including memory allocation, dispatch, caching, and profiling.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json
import os

__all__ = [
    'AllocatorConfig',
    'BackpressureConfig',
    'DispatchConfig',
    'KernelCacheConfig',
    'ProfilingConfig',
    'MetalNativeConfig',
]


@dataclass
class AllocatorConfig:
    """Memory allocator configuration.

    Attributes:
        initial_heap_size: Initial heap size in bytes (default: 256 MB)
        max_heap_size: Maximum heap size in bytes (default: 8 GB)
        growth_policy: How to grow the heap ('exponential' or 'linear')
        growth_factor: Multiplier for exponential growth (default: 2.0)
        growth_increment: Bytes to add for linear growth (default: 256 MB)
        enable_pooling: Whether to pool allocations for reuse
        small_buffer_threshold: Buffers <= this size use small pool (default: 1 MB)
    """
    initial_heap_size: int = 256 * 1024 * 1024  # 256 MB
    max_heap_size: int = 8 * 1024 * 1024 * 1024  # 8 GB
    growth_policy: str = 'exponential'
    growth_factor: float = 2.0
    growth_increment: int = 256 * 1024 * 1024  # 256 MB
    enable_pooling: bool = True
    small_buffer_threshold: int = 1024 * 1024  # 1 MB


@dataclass
class BackpressureConfig:
    """GPU backpressure management configuration.

    Attributes:
        soft_limit: Soft limit for in-flight commands (default: 64)
        hard_limit: Hard limit for in-flight commands (default: 128)
        throttle_poll_us: Microseconds to sleep when throttling (default: 100)
        enable_adaptive: Adapt limits based on workload characteristics
    """
    soft_limit: int = 64
    hard_limit: int = 128
    throttle_poll_us: int = 100
    enable_adaptive: bool = True


@dataclass
class DispatchConfig:
    """Command dispatch configuration.

    Attributes:
        use_commit_and_continue: Use commit-and-continue pattern for throughput
        max_commands_per_buffer: Max compute commands per command buffer
        release_gil: Release Python GIL during GPU dispatch (improves parallelism)
        enable_concurrent_dispatch: Allow multiple threads to dispatch simultaneously
        command_buffer_pool_size: Number of command buffers to pool
    """
    use_commit_and_continue: bool = True
    max_commands_per_buffer: int = 32
    release_gil: bool = True
    enable_concurrent_dispatch: bool = True
    command_buffer_pool_size: int = 8


@dataclass
class KernelCacheConfig:
    """Kernel compilation cache configuration.

    Attributes:
        cache_dir: Directory for cached compiled kernels (None = use temp dir)
        max_cache_size: Maximum cache size in bytes (default: 1 GB)
        enable_bucketing: Use bucketing for kernel specialization
        bucket_sizes: List of bucket sizes for shape specialization
        enable_persistent_cache: Persist cache across sessions
        cache_version: Version string for cache invalidation
    """
    cache_dir: Optional[str] = None
    max_cache_size: int = 1024 * 1024 * 1024  # 1 GB
    enable_bucketing: bool = True
    bucket_sizes: List[int] = field(default_factory=lambda: [
        64, 128, 256, 512, 1024, 2048, 4096, 8192
    ])
    enable_persistent_cache: bool = True
    cache_version: str = "0.1.0"


@dataclass
class ProfilingConfig:
    """Profiling and debugging configuration.

    Attributes:
        enable_signposts: Enable os_signpost markers for Instruments
        enable_capture: Allow GPU frame capture
        track_callstacks: Track allocation call stacks (high overhead)
        log_kernel_launches: Log all kernel dispatches
        detailed_memory_stats: Track detailed per-allocation statistics
    """
    enable_signposts: bool = False
    enable_capture: bool = False
    track_callstacks: bool = False
    log_kernel_launches: bool = False
    detailed_memory_stats: bool = False


@dataclass
class MetalNativeConfig:
    """Top-level configuration for MetalNative runtime.

    This class combines all configuration subsystems and provides methods
    for loading from files and environment variables.

    Attributes:
        allocator: Memory allocator configuration
        backpressure: GPU backpressure management configuration
        dispatch: Command dispatch configuration
        kernel_cache: Kernel cache configuration
        profiling: Profiling and debugging configuration
    """
    allocator: AllocatorConfig = field(default_factory=AllocatorConfig)
    backpressure: BackpressureConfig = field(default_factory=BackpressureConfig)
    dispatch: DispatchConfig = field(default_factory=DispatchConfig)
    kernel_cache: KernelCacheConfig = field(default_factory=KernelCacheConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            'allocator': asdict(self.allocator),
            'backpressure': asdict(self.backpressure),
            'dispatch': asdict(self.dispatch),
            'kernel_cache': asdict(self.kernel_cache),
            'profiling': asdict(self.profiling),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetalNativeConfig':
        """Create configuration from a dictionary.

        Args:
            data: Dictionary with configuration sections

        Returns:
            MetalNativeConfig instance
        """
        return cls(
            allocator=AllocatorConfig(**data.get('allocator', {})),
            backpressure=BackpressureConfig(**data.get('backpressure', {})),
            dispatch=DispatchConfig(**data.get('dispatch', {})),
            kernel_cache=KernelCacheConfig(**data.get('kernel_cache', {})),
            profiling=ProfilingConfig(**data.get('profiling', {})),
        )

    @classmethod
    def from_json(cls, path: str) -> 'MetalNativeConfig':
        """Load configuration from a JSON file.

        Args:
            path: Path to JSON configuration file

        Returns:
            MetalNativeConfig instance
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, path: str) -> 'MetalNativeConfig':
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            MetalNativeConfig instance

        Raises:
            ImportError: If PyYAML is not installed
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML configuration loading. "
                            "Install it with: pip install pyyaml")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_env(cls) -> 'MetalNativeConfig':
        """Load configuration from environment variables.

        Environment variables use the format:
        METAL_NATIVE_<SECTION>_<KEY> (e.g., METAL_NATIVE_ALLOCATOR_MAX_HEAP_SIZE)

        Returns:
            MetalNativeConfig instance with values from environment
        """
        config = cls()

        # Parse environment variables
        prefix = "METAL_NATIVE_"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue

            parts = key[len(prefix):].lower().split('_', 1)
            if len(parts) != 2:
                continue

            section, attr = parts

            # Find the corresponding config section
            if hasattr(config, section):
                section_obj = getattr(config, section)
                if hasattr(section_obj, attr):
                    # Type conversion based on current value
                    current = getattr(section_obj, attr)
                    if isinstance(current, bool):
                        setattr(section_obj, attr, value.lower() in ('true', '1', 'yes'))
                    elif isinstance(current, int):
                        setattr(section_obj, attr, int(value))
                    elif isinstance(current, float):
                        setattr(section_obj, attr, float(value))
                    else:
                        setattr(section_obj, attr, value)

        return config


# Global configuration instance
_global_config: Optional[MetalNativeConfig] = None


def get_config() -> MetalNativeConfig:
    """Get the global configuration instance.

    Returns:
        Global MetalNativeConfig instance
    """
    global _global_config
    if _global_config is None:
        _global_config = MetalNativeConfig()
    return _global_config


def set_config(config: MetalNativeConfig) -> None:
    """Set the global configuration instance.

    Args:
        config: New configuration to use
    """
    global _global_config
    _global_config = config
