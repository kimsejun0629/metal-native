#!/usr/bin/env python3
"""Regression benchmark runner for MetalNative kernels."""

import argparse
import json
import time
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Import baseline manager and report generator
from baseline_manager import BaselineManager
from report_generator import ReportGenerator

# Try to import metal_native
try:
    import metal_native as mn
    HAS_METAL_NATIVE = True
except ImportError:
    HAS_METAL_NATIVE = False
    print("Warning: metal_native not available - using placeholder timing")


def remove_outliers_iqr(samples: List[float]) -> List[float]:
    """Remove outliers using IQR method.

    Args:
        samples: List of timing samples

    Returns:
        Filtered list of samples
    """
    if len(samples) < 4:
        return samples

    q1 = np.percentile(samples, 25)
    q3 = np.percentile(samples, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    filtered = [s for s in samples if lower_bound <= s <= upper_bound]

    # Return original if we filtered too many
    if len(filtered) < len(samples) * 0.5:
        return samples

    return filtered


def get_git_hash() -> str:
    """Get current git commit hash.

    Returns:
        Git commit hash or 'unknown'
    """
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_git_branch() -> str:
    """Get current git branch name.

    Returns:
        Git branch name or 'unknown'
    """
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def benchmark_matmul(params: Dict, warmup_iters: int, bench_iters: int) -> List[float]:
    """Benchmark matrix multiplication.

    Args:
        params: Dictionary with M, N, K, dtype
        warmup_iters: Number of warmup iterations
        bench_iters: Number of benchmark iterations

    Returns:
        List of timing samples in milliseconds
    """
    M, N, K = params["M"], params["N"], params["K"]
    dtype_str = params.get("dtype", "float32")

    if not HAS_METAL_NATIVE:
        # Placeholder timing
        base_time = (M * N * K) / 1e9  # Rough GFLOPS estimate
        samples = [base_time + np.random.normal(0, base_time * 0.05) for _ in range(bench_iters)]
        return samples

    # Convert dtype string to numpy dtype
    dtype = np.float16 if dtype_str == "float16" else np.float32

    # Create tensors
    a = mn.from_numpy(np.random.randn(M, K).astype(dtype))
    b = mn.from_numpy(np.random.randn(K, N).astype(dtype))

    # Warmup
    for _ in range(warmup_iters):
        _ = a @ b
        mn.synchronize()

    # Benchmark
    samples = []
    for _ in range(bench_iters):
        start = time.perf_counter()
        c = a @ b
        mn.synchronize()
        end = time.perf_counter()
        samples.append((end - start) * 1000)

    return samples


def benchmark_attention(params: Dict, warmup_iters: int, bench_iters: int) -> List[float]:
    """Benchmark flash attention.

    Args:
        params: Dictionary with batch, num_heads, seq_len, head_dim, dtype
        warmup_iters: Number of warmup iterations
        bench_iters: Number of benchmark iterations

    Returns:
        List of timing samples in milliseconds
    """
    batch = params["batch"]
    num_heads = params["num_heads"]
    seq_len = params["seq_len"]
    head_dim = params["head_dim"]
    dtype_str = params.get("dtype", "float32")

    if not HAS_METAL_NATIVE:
        # Placeholder timing
        flops = batch * num_heads * seq_len * seq_len * head_dim
        base_time = flops / 1e9
        samples = [base_time + np.random.normal(0, base_time * 0.05) for _ in range(bench_iters)]
        return samples

    dtype = np.float16 if dtype_str == "float16" else np.float32

    # Create tensors (Q, K, V)
    q = mn.from_numpy(np.random.randn(batch, num_heads, seq_len, head_dim).astype(dtype))
    k = mn.from_numpy(np.random.randn(batch, num_heads, seq_len, head_dim).astype(dtype))
    v = mn.from_numpy(np.random.randn(batch, num_heads, seq_len, head_dim).astype(dtype))

    # Check if flash_attention is available
    if hasattr(mn, 'flash_attention'):
        # Warmup
        for _ in range(warmup_iters):
            _ = mn.flash_attention(q, k, v)
            mn.synchronize()

        # Benchmark
        samples = []
        for _ in range(bench_iters):
            start = time.perf_counter()
            out = mn.flash_attention(q, k, v)
            mn.synchronize()
            end = time.perf_counter()
            samples.append((end - start) * 1000)
    else:
        # Fallback to placeholder
        flops = batch * num_heads * seq_len * seq_len * head_dim
        base_time = flops / 1e9
        samples = [base_time + np.random.normal(0, base_time * 0.05) for _ in range(bench_iters)]

    return samples


def benchmark_softmax(params: Dict, warmup_iters: int, bench_iters: int) -> List[float]:
    """Benchmark softmax.

    Args:
        params: Dictionary with batch, rows, cols, dtype
        warmup_iters: Number of warmup iterations
        bench_iters: Number of benchmark iterations

    Returns:
        List of timing samples in milliseconds
    """
    batch = params["batch"]
    rows = params["rows"]
    cols = params["cols"]
    dtype_str = params.get("dtype", "float32")

    if not HAS_METAL_NATIVE:
        # Placeholder timing
        elements = batch * rows * cols
        base_time = elements / 1e8
        samples = [base_time + np.random.normal(0, base_time * 0.05) for _ in range(bench_iters)]
        return samples

    dtype = np.float16 if dtype_str == "float16" else np.float32

    x = mn.from_numpy(np.random.randn(batch * rows, cols).astype(dtype))

    if hasattr(mn, 'softmax'):
        # Warmup
        for _ in range(warmup_iters):
            _ = mn.softmax(x)
            mn.synchronize()

        # Benchmark
        samples = []
        for _ in range(bench_iters):
            start = time.perf_counter()
            out = mn.softmax(x)
            mn.synchronize()
            end = time.perf_counter()
            samples.append((end - start) * 1000)
    else:
        # Placeholder
        elements = batch * rows * cols
        base_time = elements / 1e8
        samples = [base_time + np.random.normal(0, base_time * 0.05) for _ in range(bench_iters)]

    return samples


def benchmark_rmsnorm(params: Dict, warmup_iters: int, bench_iters: int) -> List[float]:
    """Benchmark RMSNorm.

    Args:
        params: Dictionary with batch, seq_len, hidden, dtype
        warmup_iters: Number of warmup iterations
        bench_iters: Number of benchmark iterations

    Returns:
        List of timing samples in milliseconds
    """
    batch = params["batch"]
    seq_len = params["seq_len"]
    hidden = params["hidden"]
    dtype_str = params.get("dtype", "float32")

    if not HAS_METAL_NATIVE:
        # Placeholder timing
        elements = batch * seq_len * hidden
        base_time = elements / 1e8
        samples = [base_time + np.random.normal(0, base_time * 0.05) for _ in range(bench_iters)]
        return samples

    dtype = np.float16 if dtype_str == "float16" else np.float32

    x = mn.from_numpy(np.random.randn(batch * seq_len, hidden).astype(dtype))
    weight = mn.from_numpy(np.ones(hidden).astype(dtype))

    if hasattr(mn, 'rmsnorm'):
        # Warmup
        for _ in range(warmup_iters):
            _ = mn.rmsnorm(x, weight)
            mn.synchronize()

        # Benchmark
        samples = []
        for _ in range(bench_iters):
            start = time.perf_counter()
            out = mn.rmsnorm(x, weight)
            mn.synchronize()
            end = time.perf_counter()
            samples.append((end - start) * 1000)
    else:
        # Placeholder
        elements = batch * seq_len * hidden
        base_time = elements / 1e8
        samples = [base_time + np.random.normal(0, base_time * 0.05) for _ in range(bench_iters)]

    return samples


def benchmark_swiglu(params: Dict, warmup_iters: int, bench_iters: int) -> List[float]:
    """Benchmark SwiGLU activation.

    Args:
        params: Dictionary with batch, seq_len, hidden, dtype
        warmup_iters: Number of warmup iterations
        bench_iters: Number of benchmark iterations

    Returns:
        List of timing samples in milliseconds
    """
    batch = params["batch"]
    seq_len = params["seq_len"]
    hidden = params["hidden"]
    dtype_str = params.get("dtype", "float32")

    if not HAS_METAL_NATIVE:
        # Placeholder timing
        elements = batch * seq_len * hidden
        base_time = elements / 1e8
        samples = [base_time + np.random.normal(0, base_time * 0.05) for _ in range(bench_iters)]
        return samples

    dtype = np.float16 if dtype_str == "float16" else np.float32

    x = mn.from_numpy(np.random.randn(batch * seq_len, hidden * 2).astype(dtype))

    if hasattr(mn, 'swiglu'):
        # Warmup
        for _ in range(warmup_iters):
            _ = mn.swiglu(x)
            mn.synchronize()

        # Benchmark
        samples = []
        for _ in range(bench_iters):
            start = time.perf_counter()
            out = mn.swiglu(x)
            mn.synchronize()
            end = time.perf_counter()
            samples.append((end - start) * 1000)
    else:
        # Placeholder
        elements = batch * seq_len * hidden
        base_time = elements / 1e8
        samples = [base_time + np.random.normal(0, base_time * 0.05) for _ in range(bench_iters)]

    return samples


def run_benchmark(config: Dict, outlier_removal: str = "iqr") -> Dict:
    """Run a single benchmark and return timing statistics.

    Args:
        config: Benchmark configuration dictionary
        outlier_removal: Outlier removal method ("iqr" or "none")

    Returns:
        Dictionary with timing statistics
    """
    op = config["op"]
    params = config["params"]
    warmup_iters = config["warmup_iters"]
    bench_iters = config["bench_iters"]

    # Dispatch to appropriate benchmark function
    if op == "matmul":
        samples = benchmark_matmul(params, warmup_iters, bench_iters)
    elif op == "flash_attention":
        samples = benchmark_attention(params, warmup_iters, bench_iters)
    elif op == "softmax":
        samples = benchmark_softmax(params, warmup_iters, bench_iters)
    elif op == "rmsnorm":
        samples = benchmark_rmsnorm(params, warmup_iters, bench_iters)
    elif op == "swiglu":
        samples = benchmark_swiglu(params, warmup_iters, bench_iters)
    else:
        raise ValueError(f"Unknown op: {op}")

    # Remove outliers
    if outlier_removal == "iqr":
        filtered_samples = remove_outliers_iqr(samples)
    else:
        filtered_samples = samples

    # Compute statistics
    return {
        "name": config["name"],
        "category": config["category"],
        "median_ms": statistics.median(filtered_samples),
        "mean_ms": statistics.mean(filtered_samples),
        "std_ms": statistics.stdev(filtered_samples) if len(filtered_samples) > 1 else 0.0,
        "min_ms": min(filtered_samples),
        "max_ms": max(filtered_samples),
        "p95_ms": np.percentile(filtered_samples, 95),
        "p99_ms": np.percentile(filtered_samples, 99),
        "samples": len(filtered_samples),
        "timestamp": time.time(),
        "git_hash": get_git_hash(),
        "git_branch": get_git_branch(),
    }


def compare_with_baseline(current: Dict, baseline: Dict, threshold: float) -> Dict:
    """Compare current results with baseline.

    Args:
        current: Current benchmark result
        baseline: Baseline benchmark result
        threshold: Regression threshold (e.g., 0.05 for 5%)

    Returns:
        Comparison dictionary with status
    """
    ratio = current["median_ms"] / baseline["median_ms"]
    regression = ratio > (1.0 + threshold)
    improvement = ratio < (1.0 - threshold)

    return {
        "name": current["name"],
        "category": current.get("category", "unknown"),
        "current_ms": current["median_ms"],
        "baseline_ms": baseline["median_ms"],
        "ratio": ratio,
        "change_pct": (ratio - 1.0) * 100,
        "status": "REGRESSION" if regression else "IMPROVEMENT" if improvement else "STABLE",
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Regression benchmark runner for MetalNative kernels")
    parser.add_argument("--config", default="benchmarks/regression_config.json",
                       help="Path to regression config file")
    parser.add_argument("--save-baseline", action="store_true",
                       help="Save results as new baseline")
    parser.add_argument("--baseline-label", type=str,
                       help="Label for saved baseline (default: timestamp)")
    parser.add_argument("--compare", action="store_true",
                       help="Compare with existing baseline")
    parser.add_argument("--baseline", type=str, default="latest",
                       help="Baseline to compare against (default: latest)")
    parser.add_argument("--category", type=str,
                       help="Run only benchmarks in this category")
    parser.add_argument("--report", choices=["text", "markdown", "json"], default="text",
                       help="Report format")
    parser.add_argument("--report-file", type=str,
                       help="Save report to file")
    parser.add_argument("--fail-on-regression", action="store_true",
                       help="Exit with non-zero code if regressions detected")

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Filter benchmarks by category if specified
    benchmarks = config["benchmarks"]
    if args.category:
        benchmarks = [b for b in benchmarks if b.get("category") == args.category]
        if not benchmarks:
            print(f"Error: No benchmarks found for category '{args.category}'")
            sys.exit(1)

    # Initialize managers
    baseline_mgr = BaselineManager(config["baseline_dir"])
    report_gen = ReportGenerator(config["report_dir"])

    print(f"Running {len(benchmarks)} benchmarks...")
    print(f"MetalNative available: {HAS_METAL_NATIVE}")
    print(f"Outlier removal: {config['outlier_removal']}")
    print()

    # Run benchmarks
    results = []
    for i, bench_config in enumerate(benchmarks, 1):
        print(f"[{i}/{len(benchmarks)}] Running {bench_config['name']}...", end=" ", flush=True)
        result = run_benchmark(bench_config, config["outlier_removal"])
        results.append(result)
        print(f"{result['median_ms']:.3f} ms")

    print()

    # Save baseline if requested
    if args.save_baseline:
        label = baseline_mgr.save_baseline(
            results,
            label=args.baseline_label,
            metadata={
                "config_file": str(config_path),
                "category_filter": args.category
            }
        )
        print(f"Baseline saved: {label}")
        print()

    # Compare with baseline if requested
    if args.compare:
        try:
            baseline_data = baseline_mgr.load_baseline(args.baseline)
            baseline_results = {r["name"]: r for r in baseline_data["results"]}

            comparisons = []
            for result in results:
                if result["name"] in baseline_results:
                    threshold = config.get("regression_threshold_default", 0.05)
                    comp = compare_with_baseline(result, baseline_results[result["name"]], threshold)
                    comparisons.append(comp)

            # Generate report
            current_meta = {
                "git_hash": get_git_hash(),
                "git_branch": get_git_branch(),
                "timestamp": time.time()
            }

            baseline_meta = {
                "label": baseline_data.get("label", "unknown"),
                "git_hash": baseline_data.get("git_hash", "unknown"),
                "git_branch": baseline_data.get("git_branch", "unknown"),
                "timestamp": baseline_data.get("timestamp", "unknown")
            }

            if args.report == "markdown":
                report = report_gen.generate_markdown_report(
                    comparisons, current_meta, baseline_meta, args.report_file
                )
                print(report)
            elif args.report == "json":
                report = report_gen.generate_json_report(
                    comparisons, current_meta, baseline_meta, args.report_file
                )
                print(json.dumps(report, indent=2))
            else:  # text
                report = report_gen.generate_text_report(comparisons, current_meta, baseline_meta)
                print(report)
                if args.report_file:
                    output_path = report_gen.report_dir / args.report_file
                    with open(output_path, 'w') as f:
                        f.write(report)

            # Check for regressions
            regressions = [c for c in comparisons if c["status"] == "REGRESSION"]
            if args.fail_on_regression and regressions:
                print(f"\nFAILURE: {len(regressions)} regression(s) detected")
                sys.exit(1)

        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

    print("Done!")


if __name__ == "__main__":
    main()
