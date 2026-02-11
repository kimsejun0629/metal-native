"""MetalNative benchmark suite.

This package contains benchmarks for comparing MetalNative performance
against NumPy and PyTorch MPS across various operations.

Available benchmarks:
- benchmark_matmul: Matrix multiplication
- benchmark_attention: Attention mechanisms (including FlashAttention)
- benchmark_conv: 2D convolution operations
- benchmark_transformer: Full transformer blocks
- run_all: Run all benchmarks and generate reports
"""

__all__ = [
    'benchmark_matmul',
    'benchmark_attention',
    'benchmark_conv',
    'benchmark_transformer',
    'run_all',
]
