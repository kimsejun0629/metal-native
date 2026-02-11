#!/usr/bin/env python3
"""Benchmark matrix multiplication performance across different backends.

This benchmark compares MetalNative's matrix multiplication against NumPy
and PyTorch MPS (if available) at various matrix sizes.
"""

import argparse
import time
from typing import List, Tuple, Dict
import numpy as np

try:
    import metal_native as mn
    HAS_METAL_NATIVE = True
except ImportError:
    HAS_METAL_NATIVE = False
    print("Warning: metal_native not available")

try:
    import torch
    HAS_TORCH = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
except ImportError:
    HAS_TORCH = False


def benchmark_numpy(size: int, iterations: int, warmup: int) -> float:
    """Benchmark NumPy matrix multiplication.

    Args:
        size: Matrix dimension (NxN)
        iterations: Number of iterations to run
        warmup: Number of warmup iterations

    Returns:
        Average time in milliseconds
    """
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        _ = np.matmul(a, b)

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        c = np.matmul(a, b)
    end = time.perf_counter()

    return (end - start) * 1000 / iterations


def benchmark_metal_native(size: int, iterations: int, warmup: int) -> float:
    """Benchmark MetalNative matrix multiplication.

    Args:
        size: Matrix dimension (NxN)
        iterations: Number of iterations to run
        warmup: Number of warmup iterations

    Returns:
        Average time in milliseconds
    """
    if not HAS_METAL_NATIVE:
        return float('nan')

    a = mn.from_numpy(np.random.randn(size, size).astype(np.float32))
    b = mn.from_numpy(np.random.randn(size, size).astype(np.float32))

    # Warmup
    for _ in range(warmup):
        _ = a @ b
        mn.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        c = a @ b
        mn.synchronize()
    end = time.perf_counter()

    return (end - start) * 1000 / iterations


def benchmark_torch_mps(size: int, iterations: int, warmup: int) -> float:
    """Benchmark PyTorch MPS matrix multiplication.

    Args:
        size: Matrix dimension (NxN)
        iterations: Number of iterations to run
        warmup: Number of warmup iterations

    Returns:
        Average time in milliseconds
    """
    if not HAS_TORCH:
        return float('nan')

    device = torch.device('mps')
    a = torch.randn(size, size, device=device, dtype=torch.float32)
    b = torch.randn(size, size, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(warmup):
        _ = torch.matmul(a, b)
        torch.mps.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        c = torch.matmul(a, b)
        torch.mps.synchronize()
    end = time.perf_counter()

    return (end - start) * 1000 / iterations


def compute_gflops(size: int, time_ms: float) -> float:
    """Compute GFLOPS from matrix size and execution time.

    Args:
        size: Matrix dimension (NxN)
        time_ms: Execution time in milliseconds

    Returns:
        GFLOPS (billion floating-point operations per second)
    """
    # Matrix multiplication: 2 * N^3 FLOPs
    flops = 2 * size ** 3
    return (flops / 1e9) / (time_ms / 1000)


def run_benchmark(sizes: List[int], iterations: int, warmup: int) -> Dict:
    """Run benchmarks across all backends and sizes.

    Args:
        sizes: List of matrix sizes to benchmark
        iterations: Number of iterations per benchmark
        warmup: Number of warmup iterations

    Returns:
        Dictionary with benchmark results
    """
    results = {
        'sizes': sizes,
        'numpy': [],
        'metal_native': [],
        'torch_mps': [],
    }

    print(f"\nRunning matmul benchmarks (iterations={iterations}, warmup={warmup})")
    print("=" * 80)

    for size in sizes:
        print(f"\nSize: {size}x{size}")

        # NumPy
        numpy_time = benchmark_numpy(size, iterations, warmup)
        numpy_gflops = compute_gflops(size, numpy_time)
        results['numpy'].append({'time_ms': numpy_time, 'gflops': numpy_gflops})
        print(f"  NumPy:        {numpy_time:8.3f} ms | {numpy_gflops:7.2f} GFLOPS")

        # MetalNative
        mn_time = benchmark_metal_native(size, iterations, warmup)
        if not np.isnan(mn_time):
            mn_gflops = compute_gflops(size, mn_time)
            speedup = numpy_time / mn_time
            results['metal_native'].append({'time_ms': mn_time, 'gflops': mn_gflops, 'speedup': speedup})
            print(f"  MetalNative:  {mn_time:8.3f} ms | {mn_gflops:7.2f} GFLOPS | {speedup:5.2f}x speedup")
        else:
            results['metal_native'].append({'time_ms': float('nan'), 'gflops': float('nan'), 'speedup': float('nan')})
            print(f"  MetalNative:  Not available")

        # PyTorch MPS
        torch_time = benchmark_torch_mps(size, iterations, warmup)
        if not np.isnan(torch_time):
            torch_gflops = compute_gflops(size, torch_time)
            speedup = numpy_time / torch_time
            results['torch_mps'].append({'time_ms': torch_time, 'gflops': torch_gflops, 'speedup': speedup})
            print(f"  PyTorch MPS:  {torch_time:8.3f} ms | {torch_gflops:7.2f} GFLOPS | {speedup:5.2f}x speedup")
        else:
            results['torch_mps'].append({'time_ms': float('nan'), 'gflops': float('nan'), 'speedup': float('nan')})
            print(f"  PyTorch MPS:  Not available")

    return results


def print_summary_table(results: Dict) -> None:
    """Print a summary table of benchmark results.

    Args:
        results: Benchmark results dictionary
    """
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Size':<10} {'NumPy (ms)':<15} {'MetalNative (ms)':<20} {'PyTorch MPS (ms)':<20}")
    print("-" * 80)

    for i, size in enumerate(results['sizes']):
        numpy_time = results['numpy'][i]['time_ms']
        mn_time = results['metal_native'][i]['time_ms']
        torch_time = results['torch_mps'][i]['time_ms']

        mn_str = f"{mn_time:.3f}" if not np.isnan(mn_time) else "N/A"
        torch_str = f"{torch_time:.3f}" if not np.isnan(torch_time) else "N/A"

        print(f"{size:<10} {numpy_time:<15.3f} {mn_str:<20} {torch_str:<20}")

    print("=" * 80)


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(description='Benchmark matrix multiplication')
    parser.add_argument('--sizes', type=int, nargs='+',
                       default=[128, 256, 512, 1024, 2048, 4096],
                       help='Matrix sizes to benchmark')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of iterations per benchmark')
    parser.add_argument('--warmup', type=int, default=3,
                       help='Number of warmup iterations')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results (markdown format)')

    args = parser.parse_args()

    # Print configuration
    print("Matrix Multiplication Benchmark")
    print("=" * 80)
    print(f"Sizes: {args.sizes}")
    print(f"Iterations: {args.iterations}")
    print(f"Warmup: {args.warmup}")
    print(f"MetalNative available: {HAS_METAL_NATIVE}")
    print(f"PyTorch MPS available: {HAS_TORCH}")

    # Run benchmarks
    results = run_benchmark(args.sizes, args.iterations, args.warmup)

    # Print summary
    print_summary_table(results)

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write("# Matrix Multiplication Benchmark Results\n\n")
            f.write(f"**Configuration:** {args.iterations} iterations, {args.warmup} warmup\n\n")
            f.write("| Size | NumPy (ms) | MetalNative (ms) | PyTorch MPS (ms) | MN Speedup | MPS Speedup |\n")
            f.write("|------|------------|------------------|------------------|------------|-------------|\n")

            for i, size in enumerate(results['sizes']):
                numpy_time = results['numpy'][i]['time_ms']
                mn_data = results['metal_native'][i]
                torch_data = results['torch_mps'][i]

                mn_time_str = f"{mn_data['time_ms']:.3f}" if not np.isnan(mn_data['time_ms']) else "N/A"
                mn_speedup_str = f"{mn_data['speedup']:.2f}x" if not np.isnan(mn_data['speedup']) else "N/A"
                torch_time_str = f"{torch_data['time_ms']:.3f}" if not np.isnan(torch_data['time_ms']) else "N/A"
                torch_speedup_str = f"{torch_data['speedup']:.2f}x" if not np.isnan(torch_data['speedup']) else "N/A"

                f.write(f"| {size} | {numpy_time:.3f} | {mn_time_str} | {torch_time_str} | {mn_speedup_str} | {torch_speedup_str} |\n")

        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
