#!/usr/bin/env python3
"""Run all benchmarks and generate a comprehensive report.

This script executes all benchmark modules and outputs results in both
console and markdown formats.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import benchmark modules
try:
    from benchmarks import benchmark_matmul
    from benchmarks import benchmark_attention
    from benchmarks import benchmark_conv
    from benchmarks import benchmark_transformer
    HAS_BENCHMARKS = True
except ImportError as e:
    print(f"Warning: Could not import benchmark modules: {e}")
    HAS_BENCHMARKS = False


def run_all_benchmarks(args: argparse.Namespace) -> Dict:
    """Run all benchmark suites.

    Args:
        args: Command-line arguments

    Returns:
        Dictionary with all benchmark results
    """
    results = {}

    print("\n" + "=" * 100)
    print("METALNATIVE BENCHMARK SUITE")
    print("=" * 100)
    print(f"Iterations: {args.iterations}")
    print(f"Warmup: {args.warmup}")
    print("=" * 100)

    # Matrix multiplication benchmarks
    if not args.skip_matmul:
        print("\n[1/4] Running matrix multiplication benchmarks...")
        start = time.time()
        matmul_results = benchmark_matmul.run_benchmark(
            args.matmul_sizes, args.iterations, args.warmup
        )
        results['matmul'] = matmul_results
        print(f"Completed in {time.time() - start:.1f}s")

    # Attention benchmarks
    if not args.skip_attention:
        print("\n[2/4] Running attention benchmarks...")
        start = time.time()

        # Generate attention configs
        attention_configs = []
        for seq_len in args.attention_seq_lens:
            for head_dim in args.attention_head_dims:
                for heads in args.attention_num_heads:
                    attention_configs.append({
                        'batch': args.attention_batch,
                        'heads': heads,
                        'seq_len': seq_len,
                        'head_dim': head_dim
                    })

        attention_results = benchmark_attention.run_benchmark(
            attention_configs, args.iterations, args.warmup
        )
        results['attention'] = attention_results
        print(f"Completed in {time.time() - start:.1f}s")

    # Convolution benchmarks
    if not args.skip_conv:
        print("\n[3/4] Running convolution benchmarks...")
        start = time.time()

        conv_configs = []
        if args.conv_preset in ['resnet', 'all']:
            conv_configs.extend(benchmark_conv.get_resnet_configs(args.conv_batch))
        if args.conv_preset in ['vgg', 'all']:
            conv_configs.extend(benchmark_conv.get_vgg_configs(args.conv_batch))

        conv_results = benchmark_conv.run_benchmark(
            conv_configs, args.iterations, args.warmup
        )
        results['conv'] = conv_results
        print(f"Completed in {time.time() - start:.1f}s")

    # Transformer benchmarks
    if not args.skip_transformer:
        print("\n[4/4] Running transformer benchmarks...")
        start = time.time()

        transformer_configs = benchmark_transformer.get_model_configs()
        if args.transformer_models and 'all' not in args.transformer_models:
            transformer_configs = [
                c for c in transformer_configs if c['name'] in args.transformer_models
            ]

        transformer_results = benchmark_transformer.run_benchmark(
            transformer_configs, args.iterations, args.warmup
        )
        results['transformer'] = transformer_results
        print(f"Completed in {time.time() - start:.1f}s")

    return results


def generate_markdown_report(results: Dict, output_path: str) -> None:
    """Generate a markdown report of all benchmark results.

    Args:
        results: Benchmark results dictionary
        output_path: Path to output file
    """
    with open(output_path, 'w') as f:
        f.write("# MetalNative Benchmark Report\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # Matrix multiplication results
        if 'matmul' in results:
            f.write("## Matrix Multiplication\n\n")
            f.write("| Size | NumPy (ms) | MetalNative (ms) | PyTorch MPS (ms) | MN Speedup | MPS Speedup |\n")
            f.write("|------|------------|------------------|------------------|------------|-------------|\n")

            matmul_data = results['matmul']
            for i, size in enumerate(matmul_data['sizes']):
                numpy_time = matmul_data['numpy'][i]['time_ms']
                mn_data = matmul_data['metal_native'][i]
                mps_data = matmul_data['torch_mps'][i]

                import numpy as np
                mn_time = f"{mn_data['time_ms']:.3f}" if not np.isnan(mn_data['time_ms']) else "N/A"
                mn_speedup = f"{mn_data['speedup']:.2f}x" if not np.isnan(mn_data['speedup']) else "N/A"
                mps_time = f"{mps_data['time_ms']:.3f}" if not np.isnan(mps_data['time_ms']) else "N/A"
                mps_speedup = f"{mps_data['speedup']:.2f}x" if not np.isnan(mps_data['speedup']) else "N/A"

                f.write(f"| {size} | {numpy_time:.3f} | {mn_time} | {mps_time} | {mn_speedup} | {mps_speedup} |\n")

            f.write("\n")

        # Attention results
        if 'attention' in results:
            f.write("## Attention (FlashAttention)\n\n")
            f.write("| Config | NumPy (ms) | MetalNative (ms) | PyTorch MPS (ms) | MN Speedup |\n")
            f.write("|--------|------------|------------------|------------------|------------|\n")

            for result in results['attention']:
                cfg = result['config']
                config_str = f"B{cfg['batch']}_H{cfg['heads']}_S{cfg['seq_len']}_D{cfg['head_dim']}"
                numpy_time = result['numpy']['time_ms']

                import numpy as np
                mn_str = "N/A"
                mn_speedup = "N/A"
                if result['metal_native']:
                    mn_str = f"{result['metal_native']['time_ms']:.3f}"
                    mn_speedup = f"{result['metal_native']['speedup']:.2f}x"

                mps_str = "N/A"
                if result['torch_mps']:
                    mps_str = f"{result['torch_mps']['time_ms']:.3f}"

                f.write(f"| {config_str} | {numpy_time:.3f} | {mn_str} | {mps_str} | {mn_speedup} |\n")

            f.write("\n")

        # Convolution results
        if 'conv' in results:
            f.write("## Convolution (Conv2d)\n\n")
            f.write("| Config | CPU (ms) | MetalNative (ms) | PyTorch MPS (ms) | MN Speedup | MPS Speedup |\n")
            f.write("|--------|----------|------------------|------------------|------------|-------------|\n")

            for result in results['conv']:
                name = result['config']['name']
                cpu_time = result['cpu']['time_ms']
                mn_data = result['metal_native']
                mps_data = result['torch_mps']

                import numpy as np
                cpu_str = f"{cpu_time:.3f}" if not np.isnan(cpu_time) else "N/A"
                mn_str = f"{mn_data['time_ms']:.3f}" if not np.isnan(mn_data['time_ms']) else "N/A"
                mn_speedup = f"{mn_data['speedup']:.2f}x" if not np.isnan(mn_data['speedup']) else "N/A"
                mps_str = f"{mps_data['time_ms']:.3f}" if not np.isnan(mps_data['time_ms']) else "N/A"
                mps_speedup = f"{mps_data['speedup']:.2f}x" if not np.isnan(mps_data['speedup']) else "N/A"

                f.write(f"| {name} | {cpu_str} | {mn_str} | {mps_str} | {mn_speedup} | {mps_speedup} |\n")

            f.write("\n")

        # Transformer results
        if 'transformer' in results:
            f.write("## Transformer Block\n\n")
            f.write("| Model | NumPy (ms) | PyTorch MPS (ms) | MPS Speedup | Throughput (tok/s) |\n")
            f.write("|-------|------------|------------------|-------------|--------------------|\n")

            for result in results['transformer']:
                name = result['config']['name']
                numpy_time = result['numpy']['time_ms']
                mps_data = result['torch_mps']

                import numpy as np
                mps_str = f"{mps_data['time_ms']:.3f}" if not np.isnan(mps_data['time_ms']) else "N/A"
                speedup_str = f"{mps_data['speedup_numpy']:.2f}x" if not np.isnan(mps_data['speedup_numpy']) else "N/A"
                throughput_str = f"{result.get('throughput', 0):.1f}" if 'throughput' in result else "N/A"

                f.write(f"| {name} | {numpy_time:.3f} | {mps_str} | {speedup_str} | {throughput_str} |\n")

            f.write("\n")

        f.write("---\n\n")
        f.write("*Generated by MetalNative benchmark suite*\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run all MetalNative benchmarks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Global options
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of iterations per benchmark')
    parser.add_argument('--warmup', type=int, default=3,
                       help='Number of warmup iterations')
    parser.add_argument('--output', type=str, default='benchmark_results.md',
                       help='Output file for results (markdown)')

    # Skip options
    parser.add_argument('--skip-matmul', action='store_true',
                       help='Skip matrix multiplication benchmarks')
    parser.add_argument('--skip-attention', action='store_true',
                       help='Skip attention benchmarks')
    parser.add_argument('--skip-conv', action='store_true',
                       help='Skip convolution benchmarks')
    parser.add_argument('--skip-transformer', action='store_true',
                       help='Skip transformer benchmarks')

    # Matrix multiplication options
    parser.add_argument('--matmul-sizes', type=int, nargs='+',
                       default=[128, 256, 512, 1024, 2048],
                       help='Matrix sizes for matmul benchmark')

    # Attention options
    parser.add_argument('--attention-batch', type=int, default=2,
                       help='Batch size for attention benchmark')
    parser.add_argument('--attention-seq-lens', type=int, nargs='+',
                       default=[128, 256, 512, 1024],
                       help='Sequence lengths for attention benchmark')
    parser.add_argument('--attention-head-dims', type=int, nargs='+',
                       default=[64, 128],
                       help='Head dimensions for attention benchmark')
    parser.add_argument('--attention-num-heads', type=int, nargs='+',
                       default=[8, 12],
                       help='Number of heads for attention benchmark')

    # Convolution options
    parser.add_argument('--conv-batch', type=int, default=1,
                       help='Batch size for conv benchmark')
    parser.add_argument('--conv-preset', type=str, choices=['resnet', 'vgg', 'all'],
                       default='resnet',
                       help='Conv configuration preset')

    # Transformer options
    parser.add_argument('--transformer-models', type=str, nargs='+',
                       default=['GPT2-Small', 'GPT2-Medium'],
                       help='Transformer models to benchmark')

    args = parser.parse_args()

    if not HAS_BENCHMARKS:
        print("Error: Could not import benchmark modules", file=sys.stderr)
        sys.exit(1)

    # Run benchmarks
    print("Starting MetalNative benchmark suite...")
    start_time = time.time()

    results = run_all_benchmarks(args)

    total_time = time.time() - start_time

    # Generate report
    print("\n" + "=" * 100)
    print("GENERATING REPORT")
    print("=" * 100)

    generate_markdown_report(results, args.output)

    print(f"\nBenchmark suite completed in {total_time:.1f}s")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
