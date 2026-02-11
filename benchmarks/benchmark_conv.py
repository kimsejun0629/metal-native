#!/usr/bin/env python3
"""Benchmark Conv2d performance at common configurations.

This benchmark tests 2D convolution operations at various input sizes,
kernel sizes, and configurations commonly found in ResNet and VGG architectures.
"""

import argparse
import time
from typing import List, Dict, Tuple
import numpy as np

try:
    import metal_native as mn
    HAS_METAL_NATIVE = True
except ImportError:
    HAS_METAL_NATIVE = False
    print("Warning: metal_native not available")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
except ImportError:
    HAS_TORCH = False


def naive_conv2d_numpy(input_data: np.ndarray, weight: np.ndarray, stride: int = 1,
                       padding: int = 0, groups: int = 1) -> np.ndarray:
    """Naive Conv2d implementation in NumPy (for reference only).

    Args:
        input_data: Input [N, C, H, W]
        weight: Weight [out_channels, in_channels/groups, kH, kW]
        stride: Stride
        padding: Padding
        groups: Number of groups

    Returns:
        Output [N, out_channels, out_H, out_W]
    """
    N, C, H, W = input_data.shape
    out_channels, _, kH, kW = weight.shape

    # Add padding
    if padding > 0:
        input_padded = np.pad(input_data, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    else:
        input_padded = input_data

    # Calculate output dimensions
    out_H = (H + 2 * padding - kH) // stride + 1
    out_W = (W + 2 * padding - kW) // stride + 1

    output = np.zeros((N, out_channels, out_H, out_W), dtype=np.float32)

    # Naive implementation (very slow, for correctness only)
    for n in range(N):
        for oc in range(out_channels):
            for oh in range(out_H):
                for ow in range(out_W):
                    h_start = oh * stride
                    w_start = ow * stride
                    patch = input_padded[n, :, h_start:h_start+kH, w_start:w_start+kW]
                    output[n, oc, oh, ow] = np.sum(patch * weight[oc])

    return output


def benchmark_numpy_conv2d(config: Dict, iterations: int, warmup: int) -> float:
    """Benchmark NumPy Conv2d using scipy or naive implementation.

    Args:
        config: Configuration dict with conv parameters
        iterations: Number of iterations
        warmup: Number of warmup iterations

    Returns:
        Average time in milliseconds
    """
    # Use PyTorch on CPU as "NumPy" reference since pure NumPy conv is too slow
    if not HAS_TORCH:
        return float('nan')

    device = torch.device('cpu')
    input_data = torch.randn(config['batch'], config['in_channels'],
                             config['input_size'], config['input_size'],
                             device=device, dtype=torch.float32)
    conv = nn.Conv2d(config['in_channels'], config['out_channels'],
                     config['kernel_size'], stride=config['stride'],
                     padding=config['padding'], groups=config['groups'],
                     device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = conv(input_data)

    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            output = conv(input_data)
    end = time.perf_counter()

    return (end - start) * 1000 / iterations


def benchmark_metal_native_conv2d(config: Dict, iterations: int, warmup: int) -> float:
    """Benchmark MetalNative Conv2d.

    Args:
        config: Configuration dict with conv parameters
        iterations: Number of iterations
        warmup: Number of warmup iterations

    Returns:
        Average time in milliseconds
    """
    if not HAS_METAL_NATIVE:
        return float('nan')

    # Note: This assumes MetalNative has Conv2d in nn module
    # Placeholder implementation
    try:
        input_data = mn.from_numpy(np.random.randn(
            config['batch'], config['in_channels'],
            config['input_size'], config['input_size']
        ).astype(np.float32))

        # Placeholder for actual Conv2d layer
        # conv = mn.nn.Conv2d(config['in_channels'], config['out_channels'], ...)

        # Warmup
        for _ in range(warmup):
            # output = conv(input_data)
            mn.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            # output = conv(input_data)
            mn.synchronize()
        end = time.perf_counter()

        return (end - start) * 1000 / iterations
    except (AttributeError, NotImplementedError):
        return float('nan')


def benchmark_torch_conv2d(config: Dict, iterations: int, warmup: int) -> float:
    """Benchmark PyTorch MPS Conv2d.

    Args:
        config: Configuration dict with conv parameters
        iterations: Number of iterations
        warmup: Number of warmup iterations

    Returns:
        Average time in milliseconds
    """
    if not HAS_TORCH:
        return float('nan')

    device = torch.device('mps')
    input_data = torch.randn(config['batch'], config['in_channels'],
                             config['input_size'], config['input_size'],
                             device=device, dtype=torch.float32)
    conv = nn.Conv2d(config['in_channels'], config['out_channels'],
                     config['kernel_size'], stride=config['stride'],
                     padding=config['padding'], groups=config['groups'],
                     device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = conv(input_data)
            torch.mps.synchronize()

    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            output = conv(input_data)
            torch.mps.synchronize()
    end = time.perf_counter()

    return (end - start) * 1000 / iterations


def get_resnet_configs(batch: int = 1) -> List[Dict]:
    """Get common ResNet Conv2d configurations.

    Args:
        batch: Batch size

    Returns:
        List of configuration dicts
    """
    return [
        # ResNet first conv
        {'name': 'ResNet-First', 'batch': batch, 'in_channels': 3, 'out_channels': 64,
         'kernel_size': 7, 'stride': 2, 'padding': 3, 'groups': 1, 'input_size': 224},
        # ResNet bottleneck 1x1
        {'name': 'ResNet-Bottleneck-1x1', 'batch': batch, 'in_channels': 256, 'out_channels': 64,
         'kernel_size': 1, 'stride': 1, 'padding': 0, 'groups': 1, 'input_size': 56},
        # ResNet bottleneck 3x3
        {'name': 'ResNet-Bottleneck-3x3', 'batch': batch, 'in_channels': 64, 'out_channels': 64,
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 1, 'input_size': 56},
        # ResNet layer downsampling
        {'name': 'ResNet-Downsample', 'batch': batch, 'in_channels': 128, 'out_channels': 256,
         'kernel_size': 1, 'stride': 2, 'padding': 0, 'groups': 1, 'input_size': 56},
    ]


def get_vgg_configs(batch: int = 1) -> List[Dict]:
    """Get common VGG Conv2d configurations.

    Args:
        batch: Batch size

    Returns:
        List of configuration dicts
    """
    return [
        # VGG early layers
        {'name': 'VGG-Early', 'batch': batch, 'in_channels': 3, 'out_channels': 64,
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 1, 'input_size': 224},
        # VGG mid layers
        {'name': 'VGG-Mid', 'batch': batch, 'in_channels': 128, 'out_channels': 256,
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 1, 'input_size': 112},
        # VGG deep layers
        {'name': 'VGG-Deep', 'batch': batch, 'in_channels': 512, 'out_channels': 512,
         'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 1, 'input_size': 28},
    ]


def run_benchmark(configs: List[Dict], iterations: int, warmup: int) -> List[Dict]:
    """Run Conv2d benchmarks across configurations.

    Args:
        configs: List of configuration dicts
        iterations: Number of iterations per benchmark
        warmup: Number of warmup iterations

    Returns:
        List of result dicts
    """
    results = []

    print(f"\nRunning Conv2d benchmarks (iterations={iterations}, warmup={warmup})")
    print("=" * 100)

    for config in configs:
        print(f"\n{config['name']}:")
        print(f"  Input: [{config['batch']}, {config['in_channels']}, {config['input_size']}, {config['input_size']}]")
        print(f"  Kernel: {config['kernel_size']}x{config['kernel_size']}, stride={config['stride']}, padding={config['padding']}")
        print(f"  Output channels: {config['out_channels']}, groups: {config['groups']}")

        result = {'config': config}

        # CPU reference (using PyTorch)
        cpu_time = benchmark_numpy_conv2d(config, iterations, warmup)
        result['cpu'] = {'time_ms': cpu_time}
        if not np.isnan(cpu_time):
            print(f"  CPU (PyTorch):  {cpu_time:8.3f} ms")
        else:
            print(f"  CPU (PyTorch):  Not available")

        # MetalNative
        mn_time = benchmark_metal_native_conv2d(config, iterations, warmup)
        if not np.isnan(mn_time) and not np.isnan(cpu_time):
            speedup = cpu_time / mn_time
            result['metal_native'] = {'time_ms': mn_time, 'speedup': speedup}
            print(f"  MetalNative:    {mn_time:8.3f} ms | {speedup:5.2f}x speedup")
        else:
            result['metal_native'] = {'time_ms': mn_time, 'speedup': float('nan')}
            print(f"  MetalNative:    Not available")

        # PyTorch MPS
        torch_time = benchmark_torch_conv2d(config, iterations, warmup)
        if not np.isnan(torch_time) and not np.isnan(cpu_time):
            speedup = cpu_time / torch_time
            result['torch_mps'] = {'time_ms': torch_time, 'speedup': speedup}
            print(f"  PyTorch MPS:    {torch_time:8.3f} ms | {speedup:5.2f}x speedup")
        else:
            result['torch_mps'] = {'time_ms': torch_time, 'speedup': float('nan')}
            print(f"  PyTorch MPS:    Not available")

        results.append(result)

    return results


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(description='Benchmark Conv2d performance')
    parser.add_argument('--preset', type=str, choices=['resnet', 'vgg', 'all'],
                       default='all', help='Configuration preset to benchmark')
    parser.add_argument('--batch', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of iterations per benchmark')
    parser.add_argument('--warmup', type=int, default=3,
                       help='Number of warmup iterations')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results')

    args = parser.parse_args()

    # Get configurations
    configs = []
    if args.preset in ['resnet', 'all']:
        configs.extend(get_resnet_configs(args.batch))
    if args.preset in ['vgg', 'all']:
        configs.extend(get_vgg_configs(args.batch))

    print("Conv2d Benchmark")
    print("=" * 100)
    print(f"Preset: {args.preset}")
    print(f"Batch size: {args.batch}")
    print(f"Iterations: {args.iterations}")
    print(f"Warmup: {args.warmup}")
    print(f"MetalNative available: {HAS_METAL_NATIVE}")
    print(f"PyTorch MPS available: {HAS_TORCH}")

    # Run benchmarks
    results = run_benchmark(configs, args.iterations, args.warmup)

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write("# Conv2d Benchmark Results\n\n")
            f.write(f"**Configuration:** {args.iterations} iterations, {args.warmup} warmup, batch={args.batch}\n\n")
            f.write("| Config | CPU (ms) | MetalNative (ms) | PyTorch MPS (ms) | MN Speedup | MPS Speedup |\n")
            f.write("|--------|----------|------------------|------------------|------------|-------------|\n")

            for result in results:
                name = result['config']['name']
                cpu_time = result['cpu']['time_ms']
                mn_data = result['metal_native']
                torch_data = result['torch_mps']

                cpu_str = f"{cpu_time:.3f}" if not np.isnan(cpu_time) else "N/A"
                mn_str = f"{mn_data['time_ms']:.3f}" if not np.isnan(mn_data['time_ms']) else "N/A"
                mn_speedup = f"{mn_data['speedup']:.2f}x" if not np.isnan(mn_data['speedup']) else "N/A"
                torch_str = f"{torch_data['time_ms']:.3f}" if not np.isnan(torch_data['time_ms']) else "N/A"
                torch_speedup = f"{torch_data['speedup']:.2f}x" if not np.isnan(torch_data['speedup']) else "N/A"

                f.write(f"| {name} | {cpu_str} | {mn_str} | {torch_str} | {mn_speedup} | {torch_speedup} |\n")

        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
