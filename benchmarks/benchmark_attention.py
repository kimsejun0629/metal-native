#!/usr/bin/env python3
"""Benchmark FlashAttention performance at various configurations.

This benchmark tests scaled dot-product attention (the core operation in
transformer models) at different sequence lengths, head dimensions, and
number of attention heads.
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
    import torch.nn.functional as F
    HAS_TORCH = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
except ImportError:
    HAS_TORCH = False


def naive_attention_numpy(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Naive attention implementation in NumPy.

    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        v: Value tensor [batch, heads, seq_len, head_dim]

    Returns:
        Attention output [batch, heads, seq_len, head_dim]
    """
    head_dim = q.shape[-1]
    scale = 1.0 / np.sqrt(head_dim)

    # Compute attention scores
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale

    # Apply softmax
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    attn_weights = np.exp(scores)
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)

    # Apply attention to values
    output = np.matmul(attn_weights, v)
    return output


def benchmark_numpy_attention(batch: int, heads: int, seq_len: int, head_dim: int,
                               iterations: int, warmup: int, dtype: str = 'float32') -> Tuple[float, float]:
    """Benchmark NumPy attention.

    Args:
        batch: Batch size
        heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        iterations: Number of iterations
        warmup: Number of warmup iterations

    Returns:
        Tuple of (avg_time_ms, memory_mb)
    """
    np_dtype = np.float16 if dtype == 'float16' else np.float32
    q = np.random.randn(batch, heads, seq_len, head_dim).astype(np_dtype)
    k = np.random.randn(batch, heads, seq_len, head_dim).astype(np_dtype)
    v = np.random.randn(batch, heads, seq_len, head_dim).astype(np_dtype)

    # Warmup
    for _ in range(warmup):
        _ = naive_attention_numpy(q, k, v)

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        output = naive_attention_numpy(q, k, v)
    end = time.perf_counter()

    # Estimate memory usage (input + intermediate attention matrix + output)
    elem_bytes = 2 if dtype == 'float16' else 4
    memory_mb = (q.nbytes + k.nbytes + v.nbytes +
                 batch * heads * seq_len * seq_len * elem_bytes +  # attention scores
                 output.nbytes) / (1024 ** 2)

    return (end - start) * 1000 / iterations, memory_mb


def benchmark_metal_native_attention(batch: int, heads: int, seq_len: int, head_dim: int,
                                      iterations: int, warmup: int, dtype: str = 'float32') -> Tuple[float, float]:
    """Benchmark MetalNative FlashAttention.

    Args:
        batch: Batch size
        heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        iterations: Number of iterations
        warmup: Number of warmup iterations

    Returns:
        Tuple of (avg_time_ms, memory_mb)
    """
    if not HAS_METAL_NATIVE:
        return float('nan'), float('nan')

    # Note: This assumes MetalNative has a flash_attention function
    # If not implemented, this will need to use the naive attention
    try:
        np_dtype = np.float16 if dtype == 'float16' else np.float32
        q = mn.from_numpy(np.random.randn(batch, heads, seq_len, head_dim).astype(np_dtype))
        k = mn.from_numpy(np.random.randn(batch, heads, seq_len, head_dim).astype(np_dtype))
        v = mn.from_numpy(np.random.randn(batch, heads, seq_len, head_dim).astype(np_dtype))
    except (RuntimeError, NotImplementedError):
        return float('nan'), float('nan')

    # Use actual Metal FlashAttention kernel via C++ binding
    scale = 1.0 / np.sqrt(head_dim)

    # Warmup
    for _ in range(warmup):
        _ = mn._C.flash_attention(q._handle, k._handle, v._handle, None, scale)
        mn._C.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        output = mn._C.flash_attention(q._handle, k._handle, v._handle, None, scale)
    mn._C.synchronize()
    end = time.perf_counter()

    # Memory usage is much lower with FlashAttention (no materialized attention matrix)
    elem_bytes = 2 if dtype == 'float16' else 4
    total_elements = batch * heads * seq_len * head_dim
    memory_mb = (total_elements * elem_bytes * 4) / (1024 ** 2)  # q, k, v, output only

    return (end - start) * 1000 / iterations, memory_mb


def benchmark_torch_attention(batch: int, heads: int, seq_len: int, head_dim: int,
                               iterations: int, warmup: int, dtype: str = 'float32') -> Tuple[float, float]:
    """Benchmark PyTorch scaled_dot_product_attention.

    Args:
        batch: Batch size
        heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        iterations: Number of iterations
        warmup: Number of warmup iterations

    Returns:
        Tuple of (avg_time_ms, memory_mb)
    """
    if not HAS_TORCH:
        return float('nan'), float('nan')

    device = torch.device('mps')
    torch_dtype = torch.float16 if dtype == 'float16' else torch.float32
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch_dtype)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch_dtype)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch_dtype)

    # Warmup
    for _ in range(warmup):
        if hasattr(F, 'scaled_dot_product_attention'):
            _ = F.scaled_dot_product_attention(q, k, v)
        else:
            # Fallback to naive implementation
            scale = 1.0 / np.sqrt(head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = F.softmax(scores, dim=-1)
            _ = torch.matmul(attn_weights, v)
        torch.mps.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        if hasattr(F, 'scaled_dot_product_attention'):
            output = F.scaled_dot_product_attention(q, k, v)
        else:
            scale = 1.0 / np.sqrt(head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, v)
        torch.mps.synchronize()
    end = time.perf_counter()

    # Memory estimate
    elem_bytes = 2 if dtype == 'float16' else 4
    memory_mb = (q.element_size() * q.nelement() * 4 +  # q, k, v, output
                 batch * heads * seq_len * seq_len * elem_bytes) / (1024 ** 2)  # attention scores

    return (end - start) * 1000 / iterations, memory_mb


def compute_throughput(batch: int, seq_len: int, time_ms: float) -> float:
    """Compute throughput in tokens/second.

    Args:
        batch: Batch size
        seq_len: Sequence length
        time_ms: Execution time in milliseconds

    Returns:
        Throughput in tokens per second
    """
    total_tokens = batch * seq_len
    return total_tokens / (time_ms / 1000)


def run_benchmark(configs: List[Dict], iterations: int, warmup: int) -> Dict:
    """Run attention benchmarks across configurations.

    Args:
        configs: List of configuration dictionaries
        iterations: Number of iterations per benchmark
        warmup: Number of warmup iterations

    Returns:
        Dictionary with benchmark results
    """
    results = []

    print(f"\nRunning attention benchmarks (iterations={iterations}, warmup={warmup})")
    print("=" * 100)

    for config in configs:
        batch = config['batch']
        heads = config['heads']
        seq_len = config['seq_len']
        head_dim = config['head_dim']
        dtype = config.get('dtype', 'float32')

        print(f"\nConfig: batch={batch}, heads={heads}, seq_len={seq_len}, head_dim={head_dim}")

        result = {'config': config}

        # NumPy
        numpy_time, numpy_mem = benchmark_numpy_attention(batch, heads, seq_len, head_dim, iterations, warmup, dtype)
        numpy_throughput = compute_throughput(batch, seq_len, numpy_time)
        result['numpy'] = {'time_ms': numpy_time, 'memory_mb': numpy_mem, 'throughput': numpy_throughput}
        print(f"  NumPy:        {numpy_time:8.3f} ms | {numpy_mem:6.1f} MB | {numpy_throughput:8.1f} tokens/s")

        # MetalNative
        mn_time, mn_mem = benchmark_metal_native_attention(batch, heads, seq_len, head_dim, iterations, warmup, dtype)
        if not np.isnan(mn_time):
            mn_throughput = compute_throughput(batch, seq_len, mn_time)
            speedup = numpy_time / mn_time
            mem_saving = (numpy_mem - mn_mem) / numpy_mem * 100
            result['metal_native'] = {
                'time_ms': mn_time, 'memory_mb': mn_mem, 'throughput': mn_throughput,
                'speedup': speedup, 'mem_saving': mem_saving
            }
            print(f"  MetalNative:  {mn_time:8.3f} ms | {mn_mem:6.1f} MB | {mn_throughput:8.1f} tokens/s | {speedup:5.2f}x | {mem_saving:5.1f}% mem")
        else:
            result['metal_native'] = None
            print(f"  MetalNative:  Not available")

        # PyTorch
        torch_time, torch_mem = benchmark_torch_attention(batch, heads, seq_len, head_dim, iterations, warmup, dtype)
        if not np.isnan(torch_time):
            torch_throughput = compute_throughput(batch, seq_len, torch_time)
            speedup = numpy_time / torch_time
            mem_saving = (numpy_mem - torch_mem) / numpy_mem * 100
            result['torch_mps'] = {
                'time_ms': torch_time, 'memory_mb': torch_mem, 'throughput': torch_throughput,
                'speedup': speedup, 'mem_saving': mem_saving
            }
            print(f"  PyTorch MPS:  {torch_time:8.3f} ms | {torch_mem:6.1f} MB | {torch_throughput:8.1f} tokens/s | {speedup:5.2f}x | {mem_saving:5.1f}% mem")
        else:
            result['torch_mps'] = None
            print(f"  PyTorch MPS:  Not available")

        results.append(result)

    return results


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(description='Benchmark attention performance')
    parser.add_argument('--seq-lens', type=int, nargs='+',
                       default=[128, 256, 512, 1024, 2048],
                       help='Sequence lengths to benchmark')
    parser.add_argument('--head-dims', type=int, nargs='+',
                       default=[64, 128],
                       help='Head dimensions to benchmark')
    parser.add_argument('--num-heads', type=int, nargs='+',
                       default=[8, 12],
                       help='Number of attention heads to benchmark')
    parser.add_argument('--batch', type=int, default=2,
                       help='Batch size')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of iterations per benchmark')
    parser.add_argument('--warmup', type=int, default=3,
                       help='Number of warmup iterations')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float16'],
                       help='Data type for benchmarks (default: float32)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results')

    args = parser.parse_args()

    # Generate configurations
    configs = []
    for seq_len in args.seq_lens:
        for head_dim in args.head_dims:
            for heads in args.num_heads:
                configs.append({
                    'batch': args.batch,
                    'heads': heads,
                    'seq_len': seq_len,
                    'head_dim': head_dim,
                    'dtype': args.dtype
                })

    print("FlashAttention Benchmark")
    print("=" * 100)
    print(f"Configurations: {len(configs)}")
    print(f"Iterations: {args.iterations}")
    print(f"Warmup: {args.warmup}")
    print(f"Dtype: {args.dtype}")
    print(f"MetalNative available: {HAS_METAL_NATIVE}")
    print(f"PyTorch MPS available: {HAS_TORCH}")

    # Run benchmarks
    results = run_benchmark(configs, args.iterations, args.warmup)

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write("# FlashAttention Benchmark Results\n\n")
            f.write(f"**Configuration:** {args.iterations} iterations, {args.warmup} warmup\n\n")
            f.write("| Config | NumPy (ms) | MetalNative (ms) | PyTorch MPS (ms) | MN Speedup | MPS Speedup |\n")
            f.write("|--------|------------|------------------|------------------|------------|-------------|\n")

            for result in results:
                cfg = result['config']
                config_str = f"B{cfg['batch']}_H{cfg['heads']}_S{cfg['seq_len']}_D{cfg['head_dim']}"
                numpy_time = result['numpy']['time_ms']

                mn_str = f"{result['metal_native']['time_ms']:.3f}" if result['metal_native'] else "N/A"
                mn_speedup = f"{result['metal_native']['speedup']:.2f}x" if result['metal_native'] else "N/A"

                torch_str = f"{result['torch_mps']['time_ms']:.3f}" if result['torch_mps'] else "N/A"
                torch_speedup = f"{result['torch_mps']['speedup']:.2f}x" if result['torch_mps'] else "N/A"

                f.write(f"| {config_str} | {numpy_time:.3f} | {mn_str} | {torch_str} | {mn_speedup} | {torch_speedup} |\n")

        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
