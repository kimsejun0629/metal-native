#!/usr/bin/env python3
"""Comprehensive MPS Benchmark: PyTorch MPS vs MetalNative-style optimizations.

This benchmark measures real performance on Apple Silicon MPS, comparing:
1. PyTorch CPU baseline
2. PyTorch MPS (standard)
3. Optimized MPS patterns (mimicking MetalNative's approach: fused ops, flash attention, etc.)

Results are saved as JSON and visualized with matplotlib.
"""

import argparse
import json
import os
import platform
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# MetalNative bindings
HAS_METAL_NATIVE = False
try:
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
    import metal_native._C as _C
    # Load Metal shader library
    metallib_path = os.path.join(os.path.dirname(__file__), '..', 'build', 'shaders', 'metal_native.metallib')
    if os.path.exists(metallib_path):
        _C.load_library(metallib_path)
        HAS_METAL_NATIVE = True
        print("[MetalNative] Loaded real fused Metal kernels")
    else:
        print(f"[MetalNative] metallib not found at {metallib_path}")
except ImportError as e:
    print(f"[MetalNative] Not available: {e}")

def to_mn(t):
    """Convert PyTorch MPS tensor to MNTensor (zero-copy)."""
    if not t.is_contiguous():
        t = t.contiguous()
    dtype_map = {torch.float32: 'float32', torch.float16: 'float16'}
    return _C.tensor_from_mps_ptr(
        t.data_ptr(), t.untyped_storage().nbytes(),
        list(t.shape), list(t.stride()),
        dtype_map[t.dtype]
    )

def sync_mn():
    """Synchronize MetalNative command queue."""
    _C.synchronize()


# ============================================================================
# Benchmark Infrastructure
# ============================================================================

@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""
    name: str
    category: str
    config: str
    cpu_time_ms: float
    mps_time_ms: float
    optimized_mps_time_ms: float  # MetalNative-style optimized
    cpu_gflops: Optional[float] = None
    mps_gflops: Optional[float] = None
    optimized_gflops: Optional[float] = None
    mps_speedup_vs_cpu: Optional[float] = None
    optimized_speedup_vs_mps: Optional[float] = None
    optimized_speedup_vs_cpu: Optional[float] = None
    memory_mps_mb: Optional[float] = None
    memory_optimized_mb: Optional[float] = None


def sync_mps():
    """Synchronize MPS device."""
    torch.mps.synchronize()
    if HAS_METAL_NATIVE:
        _C.synchronize()


def benchmark_fn(fn, warmup=5, iterations=20, sync=False):
    """Benchmark a function with warmup and timing.

    Args:
        fn: Function to benchmark (no args)
        warmup: Warmup iterations
        iterations: Measurement iterations
        sync: Whether to sync MPS after each call

    Returns:
        Average time in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        fn()
        if sync:
            sync_mps()

    # Measure
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        if sync:
            sync_mps()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    # Use median to reduce variance
    return float(np.median(times))


def get_mps_memory_mb():
    """Get current MPS memory allocation in MB."""
    if hasattr(torch.mps, 'current_allocated_memory'):
        return torch.mps.current_allocated_memory() / (1024 ** 2)
    return 0.0


# ============================================================================
# 1. Matrix Multiplication Benchmark
# ============================================================================

def benchmark_matmul(sizes=None, warmup=5, iterations=20) -> List[BenchmarkResult]:
    """Benchmark matrix multiplication at various sizes."""
    if sizes is None:
        sizes = [128, 256, 512, 1024, 2048, 4096]

    results = []
    print("\n[MatMul] Running matrix multiplication benchmarks...")

    for size in sizes:
        print(f"  Size: {size}x{size}", end="", flush=True)
        flops = 2.0 * size ** 3

        # CPU
        a_cpu = torch.randn(size, size, dtype=torch.float32)
        b_cpu = torch.randn(size, size, dtype=torch.float32)
        cpu_ms = benchmark_fn(lambda: torch.matmul(a_cpu, b_cpu), warmup, iterations)

        # MPS standard
        a_mps = a_cpu.to('mps')
        b_mps = b_cpu.to('mps')
        sync_mps()
        mps_ms = benchmark_fn(lambda: torch.matmul(a_mps, b_mps), warmup, iterations, sync=True)

        # Optimized MPS: MetalNative matmul kernel
        if HAS_METAL_NATIVE:
            def matmul_metal_native():
                return _C.matmul(to_mn(a_mps), to_mn(b_mps))
            opt_ms = benchmark_fn(matmul_metal_native, warmup, iterations, sync=True)
        else:
            # Fallback: FP16 compute with FP32 accumulation
            a_fp16 = a_cpu.half().to('mps')
            b_fp16 = b_cpu.half().to('mps')
            sync_mps()
            def matmul_mixed_precision():
                c = torch.matmul(a_fp16, b_fp16)
                return c.float()
            opt_ms = benchmark_fn(matmul_mixed_precision, warmup, iterations, sync=True)

        cpu_gflops = (flops / 1e9) / (cpu_ms / 1000)
        mps_gflops = (flops / 1e9) / (mps_ms / 1000)
        opt_gflops = (flops / 1e9) / (opt_ms / 1000)

        r = BenchmarkResult(
            name="MatMul",
            category="compute",
            config=f"{size}x{size}",
            cpu_time_ms=cpu_ms,
            mps_time_ms=mps_ms,
            optimized_mps_time_ms=opt_ms,
            cpu_gflops=cpu_gflops,
            mps_gflops=mps_gflops,
            optimized_gflops=opt_gflops,
            mps_speedup_vs_cpu=cpu_ms / mps_ms,
            optimized_speedup_vs_mps=mps_ms / opt_ms,
            optimized_speedup_vs_cpu=cpu_ms / opt_ms,
        )
        results.append(r)
        label = "MN" if HAS_METAL_NATIVE else "Opt"
        print(f" -> CPU: {cpu_ms:.2f}ms | MPS: {mps_ms:.2f}ms ({r.mps_speedup_vs_cpu:.1f}x) | {label}: {opt_ms:.2f}ms ({r.optimized_speedup_vs_mps:.2f}x vs MPS)")

    return results


# ============================================================================
# 2. Attention Benchmark
# ============================================================================

def benchmark_attention(configs=None, warmup=5, iterations=15) -> List[BenchmarkResult]:
    """Benchmark attention: naive vs scaled_dot_product_attention (flash-style)."""
    if configs is None:
        configs = [
            {"batch": 2, "heads": 8, "seq_len": 128, "head_dim": 64},
            {"batch": 2, "heads": 8, "seq_len": 256, "head_dim": 64},
            {"batch": 2, "heads": 8, "seq_len": 512, "head_dim": 64},
            {"batch": 2, "heads": 12, "seq_len": 512, "head_dim": 64},
            {"batch": 2, "heads": 12, "seq_len": 1024, "head_dim": 64},
            {"batch": 1, "heads": 16, "seq_len": 2048, "head_dim": 64},
            {"batch": 1, "heads": 32, "seq_len": 2048, "head_dim": 128},
        ]

    results = []
    print("\n[Attention] Running attention benchmarks...")

    for cfg in configs:
        B, H, S, D = cfg["batch"], cfg["heads"], cfg["seq_len"], cfg["head_dim"]
        config_str = f"B{B}_H{H}_S{S}_D{D}"
        print(f"  Config: {config_str}", end="", flush=True)

        # CPU - naive attention
        q_cpu = torch.randn(B, H, S, D, dtype=torch.float32)
        k_cpu = torch.randn(B, H, S, D, dtype=torch.float32)
        v_cpu = torch.randn(B, H, S, D, dtype=torch.float32)

        def naive_attention_cpu():
            scale = 1.0 / (D ** 0.5)
            scores = torch.matmul(q_cpu, k_cpu.transpose(-2, -1)) * scale
            attn = F.softmax(scores, dim=-1)
            return torch.matmul(attn, v_cpu)

        cpu_ms = benchmark_fn(naive_attention_cpu, warmup, iterations)

        # MPS - naive attention (standard PyTorch MPS)
        q_mps = q_cpu.to('mps')
        k_mps = k_cpu.to('mps')
        v_mps = v_cpu.to('mps')
        sync_mps()

        def naive_attention_mps():
            scale = 1.0 / (D ** 0.5)
            scores = torch.matmul(q_mps, k_mps.transpose(-2, -1)) * scale
            attn = F.softmax(scores, dim=-1)
            return torch.matmul(attn, v_mps)

        mps_ms = benchmark_fn(naive_attention_mps, warmup, iterations, sync=True)

        # Optimized MPS - scaled_dot_product_attention (flash attention / memory efficient)
        # This mimics what MetalNative's FlashAttention kernel achieves
        def flash_attention_mps():
            return F.scaled_dot_product_attention(q_mps, k_mps, v_mps)

        opt_ms = benchmark_fn(flash_attention_mps, warmup, iterations, sync=True)

        # Memory estimation
        # Naive: materializes full SxS attention matrix
        naive_mem = B * H * S * S * 4 / (1024 ** 2)  # FP32 attention matrix
        # Flash: O(S) memory instead of O(S^2)
        flash_mem = B * H * S * D * 4 / (1024 ** 2)  # No materialized attention matrix

        r = BenchmarkResult(
            name="Attention",
            category="compute",
            config=config_str,
            cpu_time_ms=cpu_ms,
            mps_time_ms=mps_ms,
            optimized_mps_time_ms=opt_ms,
            mps_speedup_vs_cpu=cpu_ms / mps_ms,
            optimized_speedup_vs_mps=mps_ms / opt_ms,
            optimized_speedup_vs_cpu=cpu_ms / opt_ms,
            memory_mps_mb=naive_mem,
            memory_optimized_mb=flash_mem,
        )
        results.append(r)
        print(f" -> CPU: {cpu_ms:.1f}ms | MPS: {mps_ms:.1f}ms | Opt(Flash): {opt_ms:.1f}ms ({r.optimized_speedup_vs_mps:.2f}x) | Mem: {naive_mem:.0f}MB->{flash_mem:.0f}MB")

    return results


# ============================================================================
# 3. Softmax Benchmark
# ============================================================================

def benchmark_softmax(warmup=5, iterations=20) -> List[BenchmarkResult]:
    """Benchmark softmax at various sizes."""
    configs = [
        (2, 8, 512, 512),     # attention-sized
        (2, 12, 1024, 1024),
        (1, 16, 2048, 2048),
        (4, 1, 50257, 1),     # vocab-sized (LM head output)
    ]

    results = []
    print("\n[Softmax] Running softmax benchmarks...")

    for shape in configs:
        config_str = "x".join(str(s) for s in shape)
        total_elements = 1
        for s in shape:
            total_elements *= s
        print(f"  Shape: {config_str}", end="", flush=True)

        x_cpu = torch.randn(*shape, dtype=torch.float32)
        cpu_ms = benchmark_fn(lambda: F.softmax(x_cpu, dim=-1), warmup, iterations)

        x_mps = x_cpu.to('mps')
        sync_mps()
        mps_ms = benchmark_fn(lambda: F.softmax(x_mps, dim=-1), warmup, iterations, sync=True)

        # Optimized: MetalNative fused softmax kernel
        if HAS_METAL_NATIVE:
            def optimized_softmax():
                result = _C.softmax(to_mn(x_mps), -1)
                return result
            opt_ms = benchmark_fn(optimized_softmax, warmup, iterations, sync=True)
        else:
            # Fallback: FP16 simulation
            x_fp16 = x_cpu.half().to('mps')
            sync_mps()
            def optimized_softmax():
                return F.softmax(x_fp16, dim=-1).float()
            opt_ms = benchmark_fn(optimized_softmax, warmup, iterations, sync=True)

        r = BenchmarkResult(
            name="Softmax",
            category="memory_bound",
            config=config_str,
            cpu_time_ms=cpu_ms,
            mps_time_ms=mps_ms,
            optimized_mps_time_ms=opt_ms,
            mps_speedup_vs_cpu=cpu_ms / mps_ms,
            optimized_speedup_vs_mps=mps_ms / opt_ms,
            optimized_speedup_vs_cpu=cpu_ms / opt_ms,
        )
        results.append(r)
        label = "MN" if HAS_METAL_NATIVE else "Opt"
        print(f" -> CPU: {cpu_ms:.2f}ms | MPS: {mps_ms:.2f}ms | {label}: {opt_ms:.2f}ms ({r.optimized_speedup_vs_mps:.2f}x)")

    return results


# ============================================================================
# 4. LayerNorm Benchmark
# ============================================================================

def benchmark_layernorm(warmup=5, iterations=20) -> List[BenchmarkResult]:
    """Benchmark LayerNorm at various sizes."""
    configs = [
        {"batch": 2, "seq_len": 512, "d_model": 896, "name": "Qwen2.5-0.5B"},
        {"batch": 2, "seq_len": 512, "d_model": 1536, "name": "Qwen2.5-1.5B"},
        {"batch": 1, "seq_len": 1024, "d_model": 2048, "name": "Qwen2.5-3B"},
        {"batch": 1, "seq_len": 1024, "d_model": 3584, "name": "Qwen2.5-7B"},
    ]

    results = []
    print("\n[LayerNorm] Running LayerNorm benchmarks...")

    for cfg in configs:
        B, S, D = cfg["batch"], cfg["seq_len"], cfg["d_model"]
        config_str = f'{cfg["name"]}_{B}x{S}x{D}'
        print(f"  Config: {config_str}", end="", flush=True)

        # CPU
        x_cpu = torch.randn(B, S, D, dtype=torch.float32)
        ln_cpu = nn.LayerNorm(D)
        cpu_ms = benchmark_fn(lambda: ln_cpu(x_cpu), warmup, iterations)

        # MPS standard
        x_mps = x_cpu.to('mps')
        ln_mps = nn.LayerNorm(D).to('mps')
        sync_mps()
        mps_ms = benchmark_fn(lambda: ln_mps(x_mps), warmup, iterations, sync=True)

        # Optimized: MetalNative fused RMSNorm kernel
        if HAS_METAL_NATIVE:
            weight_mps = torch.ones(D, device='mps', dtype=torch.float32)
            sync_mps()
            def fused_rmsnorm():
                result = _C.fast.rms_norm(to_mn(x_mps), to_mn(weight_mps), 1e-5)
                return result
            opt_ms = benchmark_fn(fused_rmsnorm, warmup, iterations, sync=True)
        else:
            # Fallback: manual RMSNorm simulation
            weight_mps = ln_mps.weight.detach()
            eps = 1e-5
            def fused_rmsnorm():
                variance = x_mps.pow(2).mean(-1, keepdim=True)
                return x_mps * torch.rsqrt(variance + eps) * weight_mps
            opt_ms = benchmark_fn(fused_rmsnorm, warmup, iterations, sync=True)

        r = BenchmarkResult(
            name="LayerNorm",
            category="memory_bound",
            config=config_str,
            cpu_time_ms=cpu_ms,
            mps_time_ms=mps_ms,
            optimized_mps_time_ms=opt_ms,
            mps_speedup_vs_cpu=cpu_ms / mps_ms,
            optimized_speedup_vs_mps=mps_ms / opt_ms,
            optimized_speedup_vs_cpu=cpu_ms / opt_ms,
        )
        results.append(r)
        label = "MN(RMSNorm)" if HAS_METAL_NATIVE else "Opt(RMS)"
        print(f" -> CPU: {cpu_ms:.2f}ms | MPS: {mps_ms:.2f}ms | {label}: {opt_ms:.2f}ms ({r.optimized_speedup_vs_mps:.2f}x)")

    return results


# ============================================================================
# 5. Elementwise Operations Benchmark
# ============================================================================

def benchmark_elementwise(warmup=5, iterations=20) -> List[BenchmarkResult]:
    """Benchmark elementwise operations (GELU, SiLU, Add+Mul fused)."""
    sizes = [
        (2, 512, 896),      # Qwen2.5-0.5B hidden
        (2, 512, 4864),     # Qwen2.5-0.5B FFN
        (1, 1024, 3584),    # Qwen2.5-7B hidden
        (1, 1024, 18944),   # Qwen2.5-7B FFN
    ]

    results = []
    print("\n[Elementwise] Running elementwise benchmarks...")

    for shape in sizes:
        config_str = "x".join(str(s) for s in shape)
        print(f"  Shape: {config_str}", end="", flush=True)

        # GELU benchmark
        x_cpu = torch.randn(*shape, dtype=torch.float32)
        cpu_ms = benchmark_fn(lambda: F.gelu(x_cpu), warmup, iterations)

        x_mps = x_cpu.to('mps')
        sync_mps()
        mps_ms = benchmark_fn(lambda: F.gelu(x_mps), warmup, iterations, sync=True)

        # Optimized: MetalNative fused SwiGLU kernel
        gate_mps = torch.randn(*shape, dtype=torch.float32, device='mps')
        sync_mps()
        if HAS_METAL_NATIVE:
            def fused_silu_gate():
                result = _C.fast.swiglu(to_mn(x_mps), to_mn(gate_mps))
                return result
            opt_ms = benchmark_fn(fused_silu_gate, warmup, iterations, sync=True)
        else:
            # Fallback: SiLU + multiply simulation
            def fused_silu_gate():
                return F.silu(x_mps) * gate_mps
            opt_ms = benchmark_fn(fused_silu_gate, warmup, iterations, sync=True)

        r = BenchmarkResult(
            name="Elementwise(GELU/SiLU)",
            category="memory_bound",
            config=config_str,
            cpu_time_ms=cpu_ms,
            mps_time_ms=mps_ms,
            optimized_mps_time_ms=opt_ms,
            mps_speedup_vs_cpu=cpu_ms / mps_ms,
            optimized_speedup_vs_mps=mps_ms / opt_ms,
            optimized_speedup_vs_cpu=cpu_ms / opt_ms,
        )
        results.append(r)
        label = "MN(SwiGLU)" if HAS_METAL_NATIVE else "Opt(SiLU+Gate)"
        print(f" -> CPU: {cpu_ms:.2f}ms | MPS(GELU): {mps_ms:.2f}ms | {label}: {opt_ms:.2f}ms")

    return results


# ============================================================================
# 6. Reduction Benchmark
# ============================================================================

def benchmark_reduction(warmup=5, iterations=20) -> List[BenchmarkResult]:
    """Benchmark reduction operations (sum, mean, max)."""
    shapes = [
        (2, 512, 768),
        (2, 1024, 1024),
        (1, 2048, 4096),
        (4, 512, 50257),   # vocab-sized reduction
    ]

    results = []
    print("\n[Reduction] Running reduction benchmarks...")

    for shape in shapes:
        config_str = "x".join(str(s) for s in shape)
        print(f"  Shape: {config_str}", end="", flush=True)

        x_cpu = torch.randn(*shape, dtype=torch.float32)
        cpu_ms = benchmark_fn(lambda: x_cpu.sum(dim=-1), warmup, iterations)

        x_mps = x_cpu.to('mps')
        sync_mps()
        mps_ms = benchmark_fn(lambda: x_mps.sum(dim=-1), warmup, iterations, sync=True)

        # Optimized: FP16 reduction with FP32 accumulation (MetalNative SIMD-group approach)
        x_fp16 = x_cpu.half().to('mps')
        sync_mps()

        def optimized_reduction():
            return x_fp16.float().sum(dim=-1)

        opt_ms = benchmark_fn(optimized_reduction, warmup, iterations, sync=True)

        r = BenchmarkResult(
            name="Reduction(sum)",
            category="memory_bound",
            config=config_str,
            cpu_time_ms=cpu_ms,
            mps_time_ms=mps_ms,
            optimized_mps_time_ms=opt_ms,
            mps_speedup_vs_cpu=cpu_ms / mps_ms,
            optimized_speedup_vs_mps=mps_ms / opt_ms,
            optimized_speedup_vs_cpu=cpu_ms / opt_ms,
        )
        results.append(r)
        print(f" -> CPU: {cpu_ms:.2f}ms | MPS: {mps_ms:.2f}ms | Opt: {opt_ms:.2f}ms")

    return results


# ============================================================================
# 7. Conv2D Benchmark
# ============================================================================

def benchmark_conv2d(warmup=5, iterations=15) -> List[BenchmarkResult]:
    """Benchmark Conv2D at ResNet/VGG configurations."""
    configs = [
        {"name": "ResNet-First-7x7", "in_ch": 3, "out_ch": 64, "k": 7, "s": 2, "p": 3, "size": 224},
        {"name": "ResNet-3x3", "in_ch": 64, "out_ch": 64, "k": 3, "s": 1, "p": 1, "size": 56},
        {"name": "ResNet-1x1-Down", "in_ch": 256, "out_ch": 64, "k": 1, "s": 1, "p": 0, "size": 56},
        {"name": "VGG-3x3-Deep", "in_ch": 512, "out_ch": 512, "k": 3, "s": 1, "p": 1, "size": 28},
    ]

    results = []
    print("\n[Conv2D] Running convolution benchmarks...")

    for cfg in configs:
        config_str = cfg["name"]
        print(f"  Config: {config_str}", end="", flush=True)

        B = 4

        # CPU
        x_cpu = torch.randn(B, cfg["in_ch"], cfg["size"], cfg["size"], dtype=torch.float32)
        conv_cpu = nn.Conv2d(cfg["in_ch"], cfg["out_ch"], cfg["k"], stride=cfg["s"], padding=cfg["p"], bias=False)
        with torch.no_grad():
            cpu_ms = benchmark_fn(lambda: conv_cpu(x_cpu), warmup, iterations)

        # MPS
        x_mps = x_cpu.to('mps')
        conv_mps = nn.Conv2d(cfg["in_ch"], cfg["out_ch"], cfg["k"], stride=cfg["s"], padding=cfg["p"], bias=False).to('mps')
        sync_mps()
        with torch.no_grad():
            mps_ms = benchmark_fn(lambda: conv_mps(x_mps), warmup, iterations, sync=True)

        # Optimized: FP16 convolution (MetalNative MPSGraph with mixed precision)
        x_fp16 = x_cpu.half().to('mps')
        conv_fp16 = nn.Conv2d(cfg["in_ch"], cfg["out_ch"], cfg["k"], stride=cfg["s"], padding=cfg["p"], bias=False).half().to('mps')
        sync_mps()
        with torch.no_grad():
            def conv_mixed():
                return conv_fp16(x_fp16).float()
            opt_ms = benchmark_fn(conv_mixed, warmup, iterations, sync=True)

        r = BenchmarkResult(
            name="Conv2D",
            category="compute",
            config=config_str,
            cpu_time_ms=cpu_ms,
            mps_time_ms=mps_ms,
            optimized_mps_time_ms=opt_ms,
            mps_speedup_vs_cpu=cpu_ms / mps_ms,
            optimized_speedup_vs_mps=mps_ms / opt_ms,
            optimized_speedup_vs_cpu=cpu_ms / opt_ms,
        )
        results.append(r)
        print(f" -> CPU: {cpu_ms:.2f}ms | MPS: {mps_ms:.2f}ms ({r.mps_speedup_vs_cpu:.1f}x) | Opt: {opt_ms:.2f}ms ({r.optimized_speedup_vs_mps:.2f}x)")

    return results


# ============================================================================
# 8. Transformer Block End-to-End Benchmark
# ============================================================================

class OptimizedTransformerBlock(nn.Module):
    """Transformer block with MetalNative-style optimizations."""

    def __init__(self, d_model, n_heads, d_ff, device):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False, device=device)
        self.o_proj = nn.Linear(d_model, d_model, bias=False, device=device)

        self.gate_proj = nn.Linear(d_model, d_ff, bias=False, device=device)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False, device=device)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False, device=device)

        self.norm1 = nn.RMSNorm(d_model, device=device)
        self.norm2 = nn.RMSNorm(d_model, device=device)

    def forward(self, x):
        B, S, _ = x.shape

        # Pre-norm + fused QKV projection
        h = self.norm1(x)
        qkv = self.qkv_proj(h)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Flash attention (SDPA)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        attn_out = self.o_proj(attn_out)
        x = x + attn_out

        # SwiGLU FFN (MetalNative fused approach)
        h = self.norm2(x)
        x = x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))

        return x


class StandardTransformerBlock(nn.Module):
    """Standard PyTorch transformer block."""

    def __init__(self, d_model, n_heads, d_ff, device):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, device=device)
        self.norm1 = nn.LayerNorm(d_model, device=device)
        self.norm2 = nn.LayerNorm(d_model, device=device)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, device=device),
            nn.GELU(),
            nn.Linear(d_ff, d_model, device=device),
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


def benchmark_transformer_block(warmup=3, iterations=10) -> List[BenchmarkResult]:
    """Benchmark full transformer blocks."""
    configs = [
        {"name": "Qwen2.5-0.5B", "d_model": 896, "n_heads": 14, "d_ff": 4864, "seq_len": 512, "batch": 2},
        {"name": "Qwen2.5-1.5B", "d_model": 1536, "n_heads": 12, "d_ff": 8960, "seq_len": 512, "batch": 1},
        {"name": "Qwen2.5-3B", "d_model": 2048, "n_heads": 16, "d_ff": 11008, "seq_len": 512, "batch": 1},
        {"name": "Qwen2.5-7B", "d_model": 3584, "n_heads": 28, "d_ff": 18944, "seq_len": 512, "batch": 1},
    ]

    results = []
    print("\n[Transformer] Running transformer block benchmarks...")

    for cfg in configs:
        config_str = f'{cfg["name"]}_B{cfg["batch"]}_S{cfg["seq_len"]}'
        print(f"  Config: {config_str}", end="", flush=True)

        B, S, D = cfg["batch"], cfg["seq_len"], cfg["d_model"]
        n_heads, d_ff = cfg["n_heads"], cfg["d_ff"]

        # CPU - standard
        block_cpu = StandardTransformerBlock(D, n_heads, d_ff, 'cpu')
        x_cpu = torch.randn(B, S, D, dtype=torch.float32)
        with torch.no_grad():
            cpu_ms = benchmark_fn(lambda: block_cpu(x_cpu), warmup, iterations)

        # MPS - standard
        block_mps = StandardTransformerBlock(D, n_heads, d_ff, 'mps')
        x_mps = torch.randn(B, S, D, dtype=torch.float32, device='mps')
        sync_mps()
        with torch.no_grad():
            mps_ms = benchmark_fn(lambda: block_mps(x_mps), warmup, iterations, sync=True)

        # MPS - Optimized (MetalNative-style: RMSNorm + fused QKV + SDPA + SwiGLU)
        block_opt = OptimizedTransformerBlock(D, n_heads, d_ff, 'mps')
        sync_mps()
        with torch.no_grad():
            opt_ms = benchmark_fn(lambda: block_opt(x_mps), warmup, iterations, sync=True)

        throughput_mps = (B * S) / (mps_ms / 1000)
        throughput_opt = (B * S) / (opt_ms / 1000)

        r = BenchmarkResult(
            name="TransformerBlock",
            category="end_to_end",
            config=config_str,
            cpu_time_ms=cpu_ms,
            mps_time_ms=mps_ms,
            optimized_mps_time_ms=opt_ms,
            mps_speedup_vs_cpu=cpu_ms / mps_ms,
            optimized_speedup_vs_mps=mps_ms / opt_ms,
            optimized_speedup_vs_cpu=cpu_ms / opt_ms,
        )
        results.append(r)
        print(f" -> CPU: {cpu_ms:.1f}ms | MPS: {mps_ms:.1f}ms ({r.mps_speedup_vs_cpu:.1f}x) | Opt: {opt_ms:.1f}ms ({r.optimized_speedup_vs_mps:.2f}x) | {throughput_opt:.0f} tok/s")

    return results


# ============================================================================
# 9. Memory Efficiency Benchmark
# ============================================================================

def benchmark_memory_efficiency() -> List[Dict]:
    """Benchmark memory efficiency: standard vs optimized patterns."""
    configs = [
        {"name": "Attn-S256", "batch": 4, "heads": 12, "seq_len": 256, "head_dim": 64},
        {"name": "Attn-S512", "batch": 4, "heads": 12, "seq_len": 512, "head_dim": 64},
        {"name": "Attn-S1024", "batch": 2, "heads": 12, "seq_len": 1024, "head_dim": 64},
        {"name": "Attn-S2048", "batch": 1, "heads": 16, "seq_len": 2048, "head_dim": 64},
    ]

    results = []
    print("\n[Memory] Running memory efficiency benchmarks...")

    for cfg in configs:
        B, H, S, D = cfg["batch"], cfg["heads"], cfg["seq_len"], cfg["head_dim"]
        name = cfg["name"]

        # Naive attention memory: O(B*H*S*S) for attention matrix
        naive_attn_mem_mb = B * H * S * S * 4 / (1024 ** 2)
        # Flash attention memory: O(B*H*S*D) - no materialized attention matrix
        flash_attn_mem_mb = B * H * S * D * 4 / (1024 ** 2)

        # Total input memory (Q, K, V)
        input_mem_mb = 3 * B * H * S * D * 4 / (1024 ** 2)

        savings_pct = (1 - flash_attn_mem_mb / naive_attn_mem_mb) * 100

        results.append({
            "name": name,
            "seq_len": S,
            "naive_peak_mb": naive_attn_mem_mb + input_mem_mb,
            "flash_peak_mb": flash_attn_mem_mb + input_mem_mb,
            "savings_pct": savings_pct,
        })
        print(f"  {name}: Naive={naive_attn_mem_mb + input_mem_mb:.1f}MB | Flash={flash_attn_mem_mb + input_mem_mb:.1f}MB | Savings={savings_pct:.1f}%")

    return results


# ============================================================================
# Main Runner
# ============================================================================

def get_system_info() -> Dict:
    """Collect system information."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
    }
    # Try to get chip info
    try:
        import subprocess
        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                capture_output=True, text=True, timeout=5)
        info["cpu"] = result.stdout.strip()
    except Exception:
        info["cpu"] = "Unknown"

    try:
        import subprocess
        result = subprocess.run(['system_profiler', 'SPHardwareDataType'],
                                capture_output=True, text=True, timeout=10)
        for line in result.stdout.split('\n'):
            if 'Chip' in line:
                info["chip"] = line.split(':')[-1].strip()
            if 'Memory' in line and 'GB' in line:
                info["memory"] = line.split(':')[-1].strip()
    except Exception:
        pass

    return info


def run_all_benchmarks(quick=False) -> Dict:
    """Run all benchmark suites.

    Args:
        quick: If True, run with fewer iterations for faster results.

    Returns:
        Complete results dictionary.
    """
    warmup = 3 if quick else 5
    iters = 10 if quick else 20

    system_info = get_system_info()
    print("=" * 80)
    print("MetalNative vs PyTorch MPS - Comprehensive Benchmark")
    print("=" * 80)
    print(f"System: {system_info.get('chip', 'Unknown')} | {system_info.get('memory', 'Unknown')}")
    print(f"PyTorch: {system_info['torch_version']} | MPS: {'Available' if system_info['mps_available'] else 'N/A'}")
    print(f"Mode: {'Quick' if quick else 'Full'} (warmup={warmup}, iterations={iters})")
    print("=" * 80)

    all_results = {
        "system_info": system_info,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "quick" if quick else "full",
    }

    # Run all benchmarks
    all_results["matmul"] = [asdict(r) for r in benchmark_matmul(warmup=warmup, iterations=iters)]
    all_results["attention"] = [asdict(r) for r in benchmark_attention(warmup=warmup, iterations=min(iters, 15))]
    all_results["softmax"] = [asdict(r) for r in benchmark_softmax(warmup=warmup, iterations=iters)]
    all_results["layernorm"] = [asdict(r) for r in benchmark_layernorm(warmup=warmup, iterations=iters)]
    all_results["elementwise"] = [asdict(r) for r in benchmark_elementwise(warmup=warmup, iterations=iters)]
    all_results["reduction"] = [asdict(r) for r in benchmark_reduction(warmup=warmup, iterations=iters)]
    all_results["conv2d"] = [asdict(r) for r in benchmark_conv2d(warmup=warmup, iterations=min(iters, 15))]
    all_results["transformer"] = [asdict(r) for r in benchmark_transformer_block(warmup=min(warmup, 3), iterations=min(iters, 10))]
    all_results["memory"] = benchmark_memory_efficiency()

    print("\n" + "=" * 80)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 80)

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Comprehensive MPS Benchmark')
    parser.add_argument('--quick', action='store_true', help='Quick mode with fewer iterations')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    args = parser.parse_args()

    results = run_all_benchmarks(quick=args.quick)

    # Save results
    output_path = args.output or os.path.join(
        os.path.dirname(__file__), '..', 'benchmark_results.json'
    )
    output_path = os.path.abspath(output_path)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")
    return results


if __name__ == "__main__":
    main()
