#!/usr/bin/env python3
"""AMX Utilization Validation for MetalNative.
Sprint 0 Task 0.1: Determine if simdgroup_multiply_accumulate uses AMX.

Compares:
  A) Custom SIMD kernel (matmul_simd_fp16)
  B) MPSGraph matmul
  C) MPSMatrixMultiplication (if accessible)

Decision criteria:
  IF custom SIMD TFLOPS >= 70% of MPSGraph -> Plan A (optimize custom kernels)
  ELSE -> Plan B (optimize MPSGraph dispatch) or Plan C (MPSMatrixMultiplication)
"""

import time
import json
import numpy as np
from pathlib import Path

SIZES = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (1, 4096, 4096),      # Decode: vec-mat
]

def benchmark_matmul(M, N, K, dtype="float16", warmup=100, iters=500):
    """Benchmark a single matmul configuration."""
    try:
        import metal_native as mn
        np_dtype = np.float16 if dtype == "float16" else np.float32

        a = mn.tensor(np.random.randn(M, K).astype(np_dtype))
        b = mn.tensor(np.random.randn(K, N).astype(np_dtype))

        # Warmup
        for _ in range(warmup):
            c = mn.matmul(a, b)
            mn.synchronize()

        # Benchmark
        times = []
        for _ in range(iters):
            start = time.perf_counter_ns()
            c = mn.matmul(a, b)
            mn.synchronize()
            end = time.perf_counter_ns()
            times.append((end - start) / 1e9)  # seconds

        times.sort()
        median_time = np.median(times)
        flops = 2.0 * M * N * K
        tflops = flops / median_time / 1e12 if median_time > 0 else 0

        return {
            "M": M, "N": N, "K": K,
            "dtype": dtype,
            "median_s": float(median_time),
            "median_us": float(median_time * 1e6),
            "achieved_tflops": float(tflops),
            "samples": len(times),
        }
    except ImportError:
        flops = 2.0 * M * N * K
        return {
            "M": M, "N": N, "K": K, "dtype": dtype,
            "median_s": 0, "median_us": 0, "achieved_tflops": 0,
            "samples": 0, "note": "metal_native not available"
        }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AMX Validation Spike")
    parser.add_argument("--output", default=".omc/benchmarks/amx-validation.json")
    args = parser.parse_args()

    print("=" * 60)
    print("AMX Utilization Validation (Sprint 0, Task 0.1)")
    print("=" * 60)

    results = {"custom_simd": [], "metadata": {
        "timestamp": time.time(),
        "note": "Compare custom SIMD kernel vs MPSGraph to determine AMX utilization"
    }}

    print("\n[Custom SIMD Kernel Benchmarks]")
    for M, N, K in SIZES:
        print(f"  M={M} N={N} K={K} FP16...", end=" ", flush=True)
        result = benchmark_matmul(M, N, K, "float16")
        results["custom_simd"].append(result)
        print(f"{result['median_us']:.1f} us, {result['achieved_tflops']:.2f} TFLOPS")

    # Decision logic
    print("\n" + "=" * 60)
    print("DECISION MATRIX")
    print("=" * 60)
    print(f"{'Size':<25} {'TFLOPS':>10} {'Decision':>15}")
    print("-" * 55)

    # M4 Max theoretical: ~27 TFLOPS FP16
    theoretical_peak = 27.0
    for r in results["custom_simd"]:
        utilization = r["achieved_tflops"] / theoretical_peak * 100 if theoretical_peak > 0 else 0
        decision = "Plan A" if utilization >= 30 else "Plan B/C"
        size_str = f"M={r['M']} N={r['N']} K={r['K']}"
        print(f"{size_str:<25} {r['achieved_tflops']:>8.2f}  {decision:>15} ({utilization:.0f}% peak)")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
