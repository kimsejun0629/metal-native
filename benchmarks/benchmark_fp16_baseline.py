#!/usr/bin/env python3
"""FP16 Benchmark Baseline for MetalNative Performance Plan v2.
Sprint 0: Establish FP16 baselines across all kernel categories.
"""

import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# Test matrix sizes (from plan v2):
MATMUL_SIZES = [
    (1, 4096, 4096),      # Decode: single token projection
    (1, 4096, 11008),     # Decode: FFN gate/up
    (32, 4096, 4096),     # Small batch projection
    (128, 4096, 4096),    # Medium batch projection
    (512, 4096, 4096),    # Prefill: full context projection
    (512, 4096, 11008),   # Prefill: FFN gate/up
    (2048, 4096, 4096),   # Large prefill
]

DTYPES = ["float16", "float32"]

ATTENTION_CONFIGS = [
    {"batch": 1, "num_heads": 32, "seq_len_q": 1, "seq_len_k": 512, "head_dim": 128},    # Decode
    {"batch": 1, "num_heads": 32, "seq_len_q": 128, "seq_len_k": 128, "head_dim": 128},   # Short prefill
    {"batch": 1, "num_heads": 32, "seq_len_q": 512, "seq_len_k": 512, "head_dim": 128},   # Medium prefill
    {"batch": 1, "num_heads": 32, "seq_len_q": 2048, "seq_len_k": 2048, "head_dim": 128}, # Long prefill
]

SOFTMAX_SIZES = [
    (1, 32, 4096),   # Small
    (1, 512, 4096),  # Medium
    (1, 2048, 4096), # Large
]

NORM_SIZES = [
    (1, 512, 4096),   # Standard
    (1, 2048, 4096),  # Large
]

@dataclass
class BenchmarkResult:
    name: str
    category: str
    params: dict
    dtype: str
    median_us: float
    mean_us: float
    std_us: float
    min_us: float
    max_us: float
    p95_us: float
    p99_us: float
    throughput_gflops: Optional[float] = None
    bandwidth_gbps: Optional[float] = None
    samples: int = 0

def benchmark_op(fn, warmup=50, iters=200):
    """Run a benchmark with warmup and statistical collection."""
    # Warmup
    for _ in range(warmup):
        fn()

    # Benchmark
    times = []
    for _ in range(iters):
        start = time.perf_counter_ns()
        fn()
        end = time.perf_counter_ns()
        times.append((end - start) / 1000.0)  # Convert to microseconds

    times.sort()
    # Remove outliers (IQR method)
    q1 = np.percentile(times, 25)
    q3 = np.percentile(times, 75)
    iqr = q3 - q1
    filtered = [t for t in times if q1 - 1.5*iqr <= t <= q3 + 1.5*iqr]

    return {
        "median_us": float(np.median(filtered)),
        "mean_us": float(np.mean(filtered)),
        "std_us": float(np.std(filtered)),
        "min_us": float(np.min(filtered)),
        "max_us": float(np.max(filtered)),
        "p95_us": float(np.percentile(filtered, 95)),
        "p99_us": float(np.percentile(filtered, 99)),
        "samples": len(filtered),
    }

def run_matmul_benchmarks():
    """Benchmark matmul operations across sizes and dtypes."""
    results = []

    try:
        import metal_native as mn
    except ImportError:
        print("WARNING: metal_native not importable, using placeholder benchmarks")
        # Create placeholder results for framework validation
        for dtype in DTYPES:
            for M, N, K in MATMUL_SIZES:
                results.append(BenchmarkResult(
                    name=f"matmul_{dtype}_M{M}_N{N}_K{K}",
                    category="matmul",
                    params={"M": M, "N": N, "K": K},
                    dtype=dtype,
                    median_us=0, mean_us=0, std_us=0,
                    min_us=0, max_us=0, p95_us=0, p99_us=0,
                    throughput_gflops=0, bandwidth_gbps=0, samples=0
                ))
        return results

    for dtype_str in DTYPES:
        np_dtype = np.float16 if dtype_str == "float16" else np.float32
        for M, N, K in MATMUL_SIZES:
            print(f"  MatMul {dtype_str} M={M} N={N} K={K}...", end=" ", flush=True)

            # Create tensors
            a = mn.tensor(np.random.randn(M, K).astype(np_dtype))
            b = mn.tensor(np.random.randn(K, N).astype(np_dtype))

            def bench_fn():
                c = mn.matmul(a, b)
                mn.synchronize()

            stats = benchmark_op(bench_fn)

            # Calculate throughput
            flops = 2.0 * M * N * K
            throughput = flops / (stats["median_us"] * 1e-6) / 1e9 if stats["median_us"] > 0 else 0
            bytes_accessed = (M * K + K * N + M * N) * (2 if dtype_str == "float16" else 4)
            bandwidth = bytes_accessed / (stats["median_us"] * 1e-6) / 1e9 if stats["median_us"] > 0 else 0

            result = BenchmarkResult(
                name=f"matmul_{dtype_str}_M{M}_N{N}_K{K}",
                category="matmul",
                params={"M": M, "N": N, "K": K},
                dtype=dtype_str,
                throughput_gflops=throughput,
                bandwidth_gbps=bandwidth,
                **stats
            )
            results.append(result)
            print(f"{stats['median_us']:.1f} us ({throughput:.1f} GFLOPS)")

    return results

def run_attention_benchmarks():
    """Benchmark attention operations."""
    results = []
    try:
        import metal_native as mn
    except ImportError:
        for config in ATTENTION_CONFIGS:
            results.append(BenchmarkResult(
                name=f"attention_fp16_q{config['seq_len_q']}_k{config['seq_len_k']}",
                category="attention", params=config, dtype="float16",
                median_us=0, mean_us=0, std_us=0, min_us=0, max_us=0,
                p95_us=0, p99_us=0, samples=0
            ))
        return results

    for config in ATTENTION_CONFIGS:
        B, H, Sq, Sk, D = config["batch"], config["num_heads"], config["seq_len_q"], config["seq_len_k"], config["head_dim"]
        print(f"  Attention FP16 B={B} H={H} Sq={Sq} Sk={Sk} D={D}...", end=" ", flush=True)

        Q = mn.tensor(np.random.randn(B, H, Sq, D).astype(np.float16))
        K = mn.tensor(np.random.randn(B, H, Sk, D).astype(np.float16))
        V = mn.tensor(np.random.randn(B, H, Sk, D).astype(np.float16))

        def bench_fn():
            out = mn.flash_attention(Q, K, V, scale=1.0/np.sqrt(D))
            mn.synchronize()

        stats = benchmark_op(bench_fn, warmup=30, iters=100)
        result = BenchmarkResult(
            name=f"attention_fp16_q{Sq}_k{Sk}",
            category="attention", params=config, dtype="float16", **stats
        )
        results.append(result)
        print(f"{stats['median_us']:.1f} us")

    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MetalNative FP16 Benchmark Baseline")
    parser.add_argument("--output", default=".omc/benchmarks/baseline-fp16.json")
    parser.add_argument("--categories", nargs="+", default=["matmul", "attention", "softmax", "normalization"])
    args = parser.parse_args()

    print("=" * 60)
    print("MetalNative FP16 Benchmark Baseline (Sprint 0)")
    print("=" * 60)

    all_results = []

    if "matmul" in args.categories:
        print("\n[MatMul Benchmarks]")
        all_results.extend(run_matmul_benchmarks())

    if "attention" in args.categories:
        print("\n[Attention Benchmarks]")
        all_results.extend(run_attention_benchmarks())

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "metadata": {
            "timestamp": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": "Apple Silicon",
        },
        "results": [asdict(r) for r in all_results],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Name':<45} {'Median (us)':>12} {'GFLOPS':>10} {'GB/s':>10}")
    print("-" * 80)
    for r in all_results:
        gflops = f"{r.throughput_gflops:.1f}" if r.throughput_gflops else "N/A"
        gbps = f"{r.bandwidth_gbps:.1f}" if r.bandwidth_gbps else "N/A"
        print(f"{r.name:<45} {r.median_us:>12.1f} {gflops:>10} {gbps:>10}")

if __name__ == "__main__":
    main()
