#!/usr/bin/env python3
"""Profiling API demonstration.

This example shows how to use MetalNative's profiling capabilities:
- GPU trace capture
- Performance counter reading
- Memory snapshots
- TraceContext usage
"""

import argparse
import time
import numpy as np

try:
    import metal_native as mn
except ImportError:
    print("Error: metal_native is required for this example")
    print("Install with: pip install -e .")
    exit(1)


def demo_basic_profiling():
    """Demonstrate basic profiling operations."""
    print("=" * 80)
    print("Basic Profiling Demo")
    print("=" * 80)

    # Create some tensors
    size = 2048
    print(f"\nCreating {size}x{size} matrices...")
    a = mn.from_numpy(np.random.randn(size, size).astype(np.float32))
    b = mn.from_numpy(np.random.randn(size, size).astype(np.float32))

    # Get initial memory stats
    print("\nInitial memory state:")
    print(f"  Allocated: {mn.memory_allocated() / (1024**2):.2f} MB")
    print(f"  Peak:      {mn.max_memory_allocated() / (1024**2):.2f} MB")

    # Perform operations
    print("\nPerforming matrix operations...")
    start = time.perf_counter()

    # Matrix multiplication
    c = a @ b
    mn.synchronize()

    # Element-wise operations
    d = c + a
    e = d * 2.0
    f = e - b
    mn.synchronize()

    elapsed = time.perf_counter() - start

    print(f"Operations completed in {elapsed*1000:.2f} ms")

    # Get final memory stats
    print("\nFinal memory state:")
    print(f"  Allocated: {mn.memory_allocated() / (1024**2):.2f} MB")
    print(f"  Peak:      {mn.max_memory_allocated() / (1024**2):.2f} MB")

    # Reset peak stats
    mn.reset_peak_stats()
    print("\nPeak memory stats reset")


def demo_trace_context():
    """Demonstrate TraceContext usage (if available)."""
    print("\n" + "=" * 80)
    print("TraceContext Demo")
    print("=" * 80)

    # Note: TraceContext is a placeholder API - actual implementation would be in C++
    # This demonstrates the intended usage pattern

    try:
        # Check if profiling API is available
        # if hasattr(mn, 'TraceContext'):
        #     with mn.TraceContext("matrix_ops") as trace:
        #         a = mn.zeros((1024, 1024))
        #         b = mn.ones((1024, 1024))
        #         c = a @ b
        #         mn.synchronize()
        #
        #     print("\nTrace captured successfully")
        #     print(f"  Duration: {trace.duration_ms():.2f} ms")
        #     print(f"  GPU time: {trace.gpu_time_ms():.2f} ms")
        # else:
        #     print("\nTraceContext API not available in this build")

        print("\nTraceContext API demonstration (placeholder):")
        print("  Usage pattern:")
        print("    with mn.TraceContext('operation_name') as trace:")
        print("        # ... perform operations ...")
        print("        mn.synchronize()")
        print("    print(f'Duration: {trace.duration_ms()} ms')")

    except Exception as e:
        print(f"\nError: {e}")
        print("TraceContext may not be implemented yet")


def demo_performance_counters():
    """Demonstrate performance counter reading (if available)."""
    print("\n" + "=" * 80)
    print("Performance Counters Demo")
    print("=" * 80)

    # Note: Performance counters are a placeholder API
    # Actual implementation would query Metal performance counters

    print("\nPerformance counter API demonstration (placeholder):")
    print("  Available counters might include:")
    print("    - gpu_utilization: GPU busy percentage")
    print("    - memory_bandwidth: Memory bandwidth utilization")
    print("    - shader_cycles: Total shader core cycles")
    print("    - cache_hit_rate: Cache hit rate percentage")
    print("\n  Usage pattern:")
    print("    counters = mn.read_performance_counters()")
    print("    print(f'GPU utilization: {counters['gpu_utilization']}%')")


def demo_memory_snapshot():
    """Demonstrate memory snapshot generation."""
    print("\n" + "=" * 80)
    print("Memory Snapshot Demo")
    print("=" * 80)

    print("\nCreating tensors to generate memory activity...")

    # Create various sized tensors
    tensors = []
    sizes = [64, 128, 256, 512, 1024]

    for size in sizes:
        t = mn.zeros((size, size))
        tensors.append(t)
        print(f"  Created {size}x{size} tensor")

    # Get memory stats
    allocated = mn.memory_allocated()
    peak = mn.max_memory_allocated()

    print(f"\nMemory snapshot:")
    print(f"  Current allocation: {allocated / (1024**2):.2f} MB")
    print(f"  Peak allocation:    {peak / (1024**2):.2f} MB")
    print(f"  Number of tensors:  {len(tensors)}")

    # Calculate fragmentation info
    total_tensor_memory = sum(t.nbytes for t in tensors)
    overhead = allocated - total_tensor_memory
    print(f"  Tensor data:        {total_tensor_memory / (1024**2):.2f} MB")
    print(f"  Overhead:           {overhead / (1024**2):.2f} MB ({100 * overhead / allocated:.1f}%)")

    # Note: Actual snapshot API would be:
    # snapshot = mn.memory_snapshot()
    # snapshot.save("memory_snapshot.json")


def demo_operation_benchmark():
    """Benchmark and profile different operations."""
    print("\n" + "=" * 80)
    print("Operation Benchmark with Profiling")
    print("=" * 80)

    operations = [
        ("Matrix Mul 512x512", lambda a, b: a @ b, 512),
        ("Matrix Mul 1024x1024", lambda a, b: a @ b, 1024),
        ("Matrix Mul 2048x2048", lambda a, b: a @ b, 2048),
        ("Element-wise Add", lambda a, b: a + b, 1024),
        ("Element-wise Mul", lambda a, b: a * b, 1024),
        ("Element-wise Div", lambda a, b: a / b, 1024),
    ]

    print("\nBenchmarking operations...")
    print("-" * 80)

    for name, op, size in operations:
        # Create inputs
        a = mn.from_numpy(np.random.randn(size, size).astype(np.float32))
        b = mn.from_numpy(np.random.randn(size, size).astype(np.float32))

        # Warmup
        for _ in range(3):
            _ = op(a, b)
            mn.synchronize()

        # Reset memory stats
        mn.reset_peak_stats()

        # Benchmark
        iterations = 10
        start = time.perf_counter()

        for _ in range(iterations):
            result = op(a, b)
            mn.synchronize()

        elapsed = time.perf_counter() - start
        avg_time = (elapsed / iterations) * 1000  # ms

        # Get memory usage
        peak_memory = mn.max_memory_allocated() / (1024**2)

        print(f"{name:25s}: {avg_time:7.2f} ms  |  Peak mem: {peak_memory:6.2f} MB")

    print("-" * 80)


def main():
    """Main profiling demo entry point."""
    parser = argparse.ArgumentParser(description='MetalNative profiling demo')
    parser.add_argument('--demo', type=str, choices=['all', 'basic', 'trace', 'counters', 'snapshot', 'benchmark'],
                       default='all', help='Which demo to run')

    args = parser.parse_args()

    print("MetalNative Profiling Demonstration")
    print("=" * 80)

    # Check MetalNative availability
    if not mn.is_available():
        print("Error: Metal is not available on this system")
        return

    print(f"Device: {mn.device_name()}")

    # Get device properties if available
    try:
        props = mn.device_properties()
        if props:
            print(f"Max buffer length: {props.get('max_buffer_length', 'N/A')}")
            print(f"Max threads per group: {props.get('max_threads_per_group', 'N/A')}")
    except:
        pass

    print()

    # Run selected demos
    if args.demo in ['all', 'basic']:
        demo_basic_profiling()

    if args.demo in ['all', 'trace']:
        demo_trace_context()

    if args.demo in ['all', 'counters']:
        demo_performance_counters()

    if args.demo in ['all', 'snapshot']:
        demo_memory_snapshot()

    if args.demo in ['all', 'benchmark']:
        demo_operation_benchmark()

    print("\n" + "=" * 80)
    print("Profiling demo completed")
    print("=" * 80)


if __name__ == "__main__":
    main()
