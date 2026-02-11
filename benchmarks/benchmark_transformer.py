#!/usr/bin/env python3
"""Benchmark full transformer block performance.

This benchmark tests a complete transformer block (attention + FFN + norms)
at various model sizes matching common LLM configurations.
"""

import argparse
import time
from typing import Dict, List
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
    import torch.nn.functional as F
    HAS_TORCH = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
except ImportError:
    HAS_TORCH = False


class NaiveTransformerBlock:
    """Naive transformer block implementation in NumPy."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        """Initialize transformer block.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.head_dim = d_model // n_heads

        # Initialize weights (random)
        self.q_proj = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.k_proj = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.v_proj = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.o_proj = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.ff_w1 = np.random.randn(d_model, d_ff).astype(np.float32) * 0.02
        self.ff_w2 = np.random.randn(d_ff, d_model).astype(np.float32) * 0.02

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through transformer block.

        Args:
            x: Input [batch, seq_len, d_model]

        Returns:
            Output [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape

        # Attention
        residual = x
        x = self._layer_norm(x)

        # Multi-head attention
        q = np.matmul(x, self.q_proj).reshape(batch, seq_len, self.n_heads, self.head_dim)
        k = np.matmul(x, self.k_proj).reshape(batch, seq_len, self.n_heads, self.head_dim)
        v = np.matmul(x, self.v_proj).reshape(batch, seq_len, self.n_heads, self.head_dim)

        # Transpose for attention computation
        q = q.transpose(0, 2, 1, 3)  # [batch, n_heads, seq_len, head_dim]
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = 1.0 / np.sqrt(self.head_dim)
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        attn_weights = np.exp(scores)
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
        attn_out = np.matmul(attn_weights, v)

        # Reshape and project
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)
        attn_out = np.matmul(attn_out, self.o_proj)

        # Residual
        x = residual + attn_out

        # FFN
        residual = x
        x = self._layer_norm(x)
        x = np.matmul(x, self.ff_w1)
        x = self._gelu(x)
        x = np.matmul(x, self.ff_w2)
        x = residual + x

        return x

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


class TorchTransformerBlock(nn.Module):
    """PyTorch transformer block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, device: torch.device):
        """Initialize transformer block.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            device: Device to place parameters on
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, device=device)
        self.norm1 = nn.LayerNorm(d_model, device=device)
        self.norm2 = nn.LayerNorm(d_model, device=device)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, device=device),
            nn.GELU(),
            nn.Linear(d_ff, d_model, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Attention
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.norm1(x)

        # FFN
        x = x + self.ff(x)
        x = self.norm2(x)

        return x


def benchmark_numpy_transformer(config: Dict, iterations: int, warmup: int) -> float:
    """Benchmark NumPy transformer block.

    Args:
        config: Configuration dict
        iterations: Number of iterations
        warmup: Number of warmup iterations

    Returns:
        Average time in milliseconds
    """
    block = NaiveTransformerBlock(config['d_model'], config['n_heads'], config['d_ff'])
    x = np.random.randn(config['batch'], config['seq_len'], config['d_model']).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        _ = block.forward(x)

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        output = block.forward(x)
    end = time.perf_counter()

    return (end - start) * 1000 / iterations


def benchmark_torch_transformer(config: Dict, iterations: int, warmup: int,
                                device: str = 'mps') -> float:
    """Benchmark PyTorch transformer block.

    Args:
        config: Configuration dict
        iterations: Number of iterations
        warmup: Number of warmup iterations
        device: Device to run on ('cpu' or 'mps')

    Returns:
        Average time in milliseconds
    """
    if not HAS_TORCH:
        return float('nan')

    if device == 'mps' and not HAS_TORCH:
        return float('nan')

    torch_device = torch.device(device)
    block = TorchTransformerBlock(config['d_model'], config['n_heads'],
                                   config['d_ff'], torch_device)
    x = torch.randn(config['batch'], config['seq_len'], config['d_model'],
                    device=torch_device, dtype=torch.float32)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = block(x)
            if device == 'mps':
                torch.mps.synchronize()

    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            output = block(x)
            if device == 'mps':
                torch.mps.synchronize()
    end = time.perf_counter()

    return (end - start) * 1000 / iterations


def get_model_configs() -> List[Dict]:
    """Get common LLM transformer configurations.

    Returns:
        List of configuration dicts
    """
    return [
        # GPT-2 Small (125M params)
        {'name': 'GPT2-Small', 'batch': 1, 'seq_len': 512, 'd_model': 768,
         'n_heads': 12, 'd_ff': 3072, 'params_m': 125},
        # GPT-2 Medium (350M params)
        {'name': 'GPT2-Medium', 'batch': 1, 'seq_len': 512, 'd_model': 1024,
         'n_heads': 16, 'd_ff': 4096, 'params_m': 350},
        # GPT-2 Large (774M params)
        {'name': 'GPT2-Large', 'batch': 1, 'seq_len': 512, 'd_model': 1280,
         'n_heads': 20, 'd_ff': 5120, 'params_m': 774},
        # GPT-2 XL (1.5B params)
        {'name': 'GPT2-XL', 'batch': 1, 'seq_len': 512, 'd_model': 1600,
         'n_heads': 25, 'd_ff': 6400, 'params_m': 1500},
        # Llama-like (7B scale, single layer)
        {'name': 'Llama-7B-Block', 'batch': 1, 'seq_len': 512, 'd_model': 4096,
         'n_heads': 32, 'd_ff': 11008, 'params_m': 7000},
    ]


def run_benchmark(configs: List[Dict], iterations: int, warmup: int) -> List[Dict]:
    """Run transformer block benchmarks.

    Args:
        configs: List of configuration dicts
        iterations: Number of iterations per benchmark
        warmup: Number of warmup iterations

    Returns:
        List of result dicts
    """
    results = []

    print(f"\nRunning transformer block benchmarks (iterations={iterations}, warmup={warmup})")
    print("=" * 100)

    for config in configs:
        print(f"\n{config['name']} (~{config['params_m']}M params per layer):")
        print(f"  d_model={config['d_model']}, n_heads={config['n_heads']}, d_ff={config['d_ff']}")
        print(f"  batch={config['batch']}, seq_len={config['seq_len']}")

        result = {'config': config}

        # NumPy reference
        numpy_time = benchmark_numpy_transformer(config, iterations, warmup)
        result['numpy'] = {'time_ms': numpy_time}
        print(f"  NumPy:        {numpy_time:9.3f} ms")

        # PyTorch CPU
        cpu_time = benchmark_torch_transformer(config, iterations, warmup, device='cpu')
        if not np.isnan(cpu_time):
            speedup = numpy_time / cpu_time
            result['torch_cpu'] = {'time_ms': cpu_time, 'speedup': speedup}
            print(f"  PyTorch CPU:  {cpu_time:9.3f} ms | {speedup:5.2f}x vs NumPy")
        else:
            result['torch_cpu'] = {'time_ms': cpu_time, 'speedup': float('nan')}

        # PyTorch MPS
        mps_time = benchmark_torch_transformer(config, iterations, warmup, device='mps')
        if not np.isnan(mps_time):
            speedup_numpy = numpy_time / mps_time
            speedup_cpu = cpu_time / mps_time if not np.isnan(cpu_time) else float('nan')
            result['torch_mps'] = {'time_ms': mps_time, 'speedup_numpy': speedup_numpy,
                                   'speedup_cpu': speedup_cpu}
            print(f"  PyTorch MPS:  {mps_time:9.3f} ms | {speedup_numpy:5.2f}x vs NumPy | {speedup_cpu:5.2f}x vs CPU")
        else:
            result['torch_mps'] = {'time_ms': mps_time, 'speedup_numpy': float('nan'),
                                   'speedup_cpu': float('nan')}
            print(f"  PyTorch MPS:  Not available")

        # Throughput (tokens/second)
        if not np.isnan(mps_time):
            throughput = (config['batch'] * config['seq_len']) / (mps_time / 1000)
            result['throughput'] = throughput
            print(f"  Throughput:   {throughput:9.1f} tokens/s (MPS)")

        results.append(result)

    return results


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(description='Benchmark transformer block performance')
    parser.add_argument('--models', type=str, nargs='+',
                       choices=['GPT2-Small', 'GPT2-Medium', 'GPT2-Large', 'GPT2-XL', 'Llama-7B-Block', 'all'],
                       default=['all'],
                       help='Model configurations to benchmark')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of iterations per benchmark')
    parser.add_argument('--warmup', type=int, default=3,
                       help='Number of warmup iterations')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results')

    args = parser.parse_args()

    # Get configurations
    all_configs = get_model_configs()
    if 'all' in args.models:
        configs = all_configs
    else:
        configs = [c for c in all_configs if c['name'] in args.models]

    print("Transformer Block Benchmark")
    print("=" * 100)
    print(f"Models: {[c['name'] for c in configs]}")
    print(f"Iterations: {args.iterations}")
    print(f"Warmup: {args.warmup}")
    print(f"PyTorch available: {HAS_TORCH}")
    print(f"PyTorch MPS available: {HAS_TORCH}")

    # Run benchmarks
    results = run_benchmark(configs, args.iterations, args.warmup)

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write("# Transformer Block Benchmark Results\n\n")
            f.write(f"**Configuration:** {args.iterations} iterations, {args.warmup} warmup\n\n")
            f.write("| Model | NumPy (ms) | PyTorch CPU (ms) | PyTorch MPS (ms) | MPS Speedup | Throughput (tok/s) |\n")
            f.write("|-------|------------|------------------|------------------|-------------|--------------------|\n")

            for result in results:
                name = result['config']['name']
                numpy_time = result['numpy']['time_ms']
                cpu_data = result['torch_cpu']
                mps_data = result['torch_mps']

                cpu_str = f"{cpu_data['time_ms']:.3f}" if not np.isnan(cpu_data['time_ms']) else "N/A"
                mps_str = f"{mps_data['time_ms']:.3f}" if not np.isnan(mps_data['time_ms']) else "N/A"
                speedup_str = f"{mps_data['speedup_numpy']:.2f}x" if not np.isnan(mps_data['speedup_numpy']) else "N/A"
                throughput_str = f"{result.get('throughput', float('nan')):.1f}" if 'throughput' in result else "N/A"

                f.write(f"| {name} | {numpy_time:.3f} | {cpu_str} | {mps_str} | {speedup_str} | {throughput_str} |\n")

        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
