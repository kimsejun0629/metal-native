#!/usr/bin/env python3
"""Model-based per-operation benchmark: PyTorch MPS vs MetalNative.

Tests individual transformer block operations at real LLM dimensions.
Creates tensors via CPU numpy to avoid MPS data_ptr SIGBUS on large buffers.
"""

import sys
import os
import time
import numpy as np

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import torch
import torch.nn.functional as F

# Load MetalNative
import metal_native._C as _C
metallib = os.path.join(os.path.dirname(__file__), '..', 'build', 'shaders', 'metal_native.metallib')
if os.path.exists(metallib):
    _C.load_library(metallib)

# ── Helpers ──────────────────────────────────────────────────────────────────

def make_mn(shape, dtype='float32'):
    """Create MNTensor from random numpy data."""
    if dtype == 'float32':
        data = np.random.randn(*shape).astype(np.float32)
    elif dtype == 'float16':
        data = np.random.randn(*shape).astype(np.float16)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return _C.tensor_from_cpu_data(data.ctypes.data, data.nbytes, list(data.shape), dtype), data

def make_mps(data_np):
    """Create MPS tensor from numpy data."""
    return torch.from_numpy(data_np.copy()).to('mps')

def bench_mps(fn, warmup=5, iters=20):
    """Benchmark a PyTorch MPS function."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
            torch.mps.synchronize()
        times = []
        for _ in range(iters):
            torch.mps.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.mps.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    return float(np.median(times))

def bench_mn(fn, warmup=5, iters=20):
    """Benchmark a MetalNative function."""
    _C.set_lazy_commit(True)
    for _ in range(warmup):
        fn()
        _C.synchronize()
    times = []
    for _ in range(iters):
        _C.synchronize()
        t0 = time.perf_counter()
        fn()
        _C.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    _C.set_lazy_commit(False)
    return float(np.median(times))

# ── Model Configs ────────────────────────────────────────────────────────────

MODELS = [
    {"name": "Qwen2.5-0.5B",  "d": 896,  "heads": 14, "kv_heads": 2,  "ffn": 4864,  "seq": 512},
    {"name": "Qwen2.5-1.5B",  "d": 1536, "heads": 12, "kv_heads": 2,  "ffn": 8960,  "seq": 512},
    {"name": "Llama-3.2-3B",   "d": 3072, "heads": 24, "kv_heads": 8,  "ffn": 8192,  "seq": 512},
    {"name": "Qwen2.5-7B",    "d": 3584, "heads": 28, "kv_heads": 4,  "ffn": 18944, "seq": 512},
]

# ── Main ─────────────────────────────────────────────────────────────────────

def benchmark_model(cfg, warmup=5, iters=20):
    """Benchmark all operations for one model config."""
    name = cfg["name"]
    d = cfg["d"]
    ffn = cfg["ffn"]
    seq = cfg["seq"]
    heads = cfg["heads"]
    head_dim = d // heads
    B = 1

    print(f"\n{'='*70}")
    print(f"  {name}  (d={d}, heads={heads}, ffn={ffn}, seq={seq})")
    print(f"{'='*70}")

    results = {}

    # ── 1. RMSNorm ────────────────────────────────────────────────────────
    print(f"  RMSNorm [{B},{seq},{d}]...", end="", flush=True)
    x_np = np.random.randn(B, seq, d).astype(np.float32)
    w_np = np.ones(d, dtype=np.float32)

    x_mps = make_mps(x_np)
    w_mps = torch.ones(d, device='mps', dtype=torch.float32)
    rms_mps = torch.nn.RMSNorm(d).to('mps')
    torch.mps.synchronize()
    mps_ms = bench_mps(lambda: rms_mps(x_mps), warmup, iters)

    mn_x, _ = make_mn((B, seq, d))
    mn_w, _ = make_mn((d,))
    # set weight to ones
    mn_w = _C.tensor_from_cpu_data(w_np.ctypes.data, w_np.nbytes, list(w_np.shape), 'float32')
    mn_ms = bench_mn(lambda: _C.fast.rms_norm(mn_x, mn_w, 1e-5), warmup, iters)

    ratio = mps_ms / mn_ms if mn_ms > 0 else 0
    results["RMSNorm"] = {"mps": mps_ms, "mn": mn_ms, "ratio": ratio}
    print(f" MPS={mps_ms:.3f}ms  MN={mn_ms:.3f}ms  -> {ratio:.1f}x")

    # ── 2. QKV Projection (MatMul) ────────────────────────────────────────
    print(f"  QKV Proj [{B},{seq},{d}]@[{d},{3*d}]...", end="", flush=True)
    qkv_w_np = np.random.randn(d, 3 * d).astype(np.float32) * 0.02

    x_mps = make_mps(x_np)
    qkv_w_mps = make_mps(qkv_w_np)
    torch.mps.synchronize()
    mps_ms = bench_mps(lambda: torch.matmul(x_mps, qkv_w_mps), warmup, iters)

    mn_x = _C.tensor_from_cpu_data(x_np.ctypes.data, x_np.nbytes, list(x_np.shape), 'float32')
    mn_qkv_w = _C.tensor_from_cpu_data(qkv_w_np.ctypes.data, qkv_w_np.nbytes, list(qkv_w_np.shape), 'float32')
    mn_ms = bench_mn(lambda: _C.matmul(mn_x, mn_qkv_w), warmup, iters)

    ratio = mps_ms / mn_ms if mn_ms > 0 else 0
    results["QKV_Proj"] = {"mps": mps_ms, "mn": mn_ms, "ratio": ratio}
    print(f" MPS={mps_ms:.3f}ms  MN={mn_ms:.3f}ms  -> {ratio:.2f}x")

    # ── 3. Output Projection ──────────────────────────────────────────────
    print(f"  Out Proj [{B},{seq},{d}]@[{d},{d}]...", end="", flush=True)
    o_w_np = np.random.randn(d, d).astype(np.float32) * 0.02

    o_w_mps = make_mps(o_w_np)
    torch.mps.synchronize()
    mps_ms = bench_mps(lambda: torch.matmul(x_mps, o_w_mps), warmup, iters)

    mn_o_w = _C.tensor_from_cpu_data(o_w_np.ctypes.data, o_w_np.nbytes, list(o_w_np.shape), 'float32')
    mn_ms = bench_mn(lambda: _C.matmul(mn_x, mn_o_w), warmup, iters)

    ratio = mps_ms / mn_ms if mn_ms > 0 else 0
    results["Out_Proj"] = {"mps": mps_ms, "mn": mn_ms, "ratio": ratio}
    print(f" MPS={mps_ms:.3f}ms  MN={mn_ms:.3f}ms  -> {ratio:.2f}x")

    # ── 4. Gate+Up Projection ─────────────────────────────────────────────
    print(f"  Gate+Up [{B},{seq},{d}]@[{d},{ffn}]...", end="", flush=True)
    gate_w_np = np.random.randn(d, ffn).astype(np.float32) * 0.02

    gate_w_mps = make_mps(gate_w_np)
    torch.mps.synchronize()
    mps_ms = bench_mps(lambda: torch.matmul(x_mps, gate_w_mps), warmup, iters)

    mn_gate_w = _C.tensor_from_cpu_data(gate_w_np.ctypes.data, gate_w_np.nbytes, list(gate_w_np.shape), 'float32')
    mn_ms = bench_mn(lambda: _C.matmul(mn_x, mn_gate_w), warmup, iters)

    ratio = mps_ms / mn_ms if mn_ms > 0 else 0
    results["GateUp_Proj"] = {"mps": mps_ms, "mn": mn_ms, "ratio": ratio}
    print(f" MPS={mps_ms:.3f}ms  MN={mn_ms:.3f}ms  -> {ratio:.2f}x")

    # ── 5. SwiGLU Activation ──────────────────────────────────────────────
    print(f"  SwiGLU [{B},{seq},{ffn}]...", end="", flush=True)
    gate_np = np.random.randn(B, seq, ffn).astype(np.float32)
    up_np = np.random.randn(B, seq, ffn).astype(np.float32)

    gate_mps = make_mps(gate_np)
    up_mps = make_mps(up_np)
    torch.mps.synchronize()
    mps_ms = bench_mps(lambda: F.silu(gate_mps) * up_mps, warmup, iters)

    mn_gate = _C.tensor_from_cpu_data(gate_np.ctypes.data, gate_np.nbytes, list(gate_np.shape), 'float32')
    mn_up = _C.tensor_from_cpu_data(up_np.ctypes.data, up_np.nbytes, list(up_np.shape), 'float32')
    mn_ms = bench_mn(lambda: _C.fast.swiglu(mn_gate, mn_up), warmup, iters)

    ratio = mps_ms / mn_ms if mn_ms > 0 else 0
    results["SwiGLU"] = {"mps": mps_ms, "mn": mn_ms, "ratio": ratio}
    print(f" MPS={mps_ms:.3f}ms  MN={mn_ms:.3f}ms  -> {ratio:.1f}x")

    # ── 6. Down Projection ────────────────────────────────────────────────
    print(f"  Down Proj [{B},{seq},{ffn}]@[{ffn},{d}]...", end="", flush=True)
    down_w_np = np.random.randn(ffn, d).astype(np.float32) * 0.02

    ffn_out_mps = make_mps(gate_np)  # reuse as FFN intermediate
    down_w_mps = make_mps(down_w_np)
    torch.mps.synchronize()
    mps_ms = bench_mps(lambda: torch.matmul(ffn_out_mps, down_w_mps), warmup, iters)

    mn_ffn_out = _C.tensor_from_cpu_data(gate_np.ctypes.data, gate_np.nbytes, list(gate_np.shape), 'float32')
    mn_down_w = _C.tensor_from_cpu_data(down_w_np.ctypes.data, down_w_np.nbytes, list(down_w_np.shape), 'float32')
    mn_ms = bench_mn(lambda: _C.matmul(mn_ffn_out, mn_down_w), warmup, iters)

    ratio = mps_ms / mn_ms if mn_ms > 0 else 0
    results["Down_Proj"] = {"mps": mps_ms, "mn": mn_ms, "ratio": ratio}
    print(f" MPS={mps_ms:.3f}ms  MN={mn_ms:.3f}ms  -> {ratio:.2f}x")

    # ── 7. Softmax ────────────────────────────────────────────────────────
    print(f"  Softmax [{B},{heads},{seq},{seq}]...", end="", flush=True)
    attn_np = np.random.randn(B, heads, seq, seq).astype(np.float32)

    attn_mps = make_mps(attn_np)
    torch.mps.synchronize()
    mps_ms = bench_mps(lambda: F.softmax(attn_mps, dim=-1), warmup, iters)

    mn_attn = _C.tensor_from_cpu_data(attn_np.ctypes.data, attn_np.nbytes, list(attn_np.shape), 'float32')
    mn_ms = bench_mn(lambda: _C.softmax(mn_attn, -1), warmup, iters)

    ratio = mps_ms / mn_ms if mn_ms > 0 else 0
    results["Softmax"] = {"mps": mps_ms, "mn": mn_ms, "ratio": ratio}
    print(f" MPS={mps_ms:.3f}ms  MN={mn_ms:.3f}ms  -> {ratio:.1f}x")

    return results


def main():
    print("=" * 70)
    print("  MetalNative vs PyTorch MPS - Model-Based Operation Benchmark")
    print(f"  PyTorch {torch.__version__} | Python {sys.version.split()[0]}")
    print(f"  Lazy commit: enabled for MetalNative fused ops")
    print("=" * 70)

    all_results = {}
    for cfg in MODELS:
        try:
            results = benchmark_model(cfg, warmup=5, iters=20)
            all_results[cfg["name"]] = results
        except Exception as e:
            print(f"\n  ERROR on {cfg['name']}: {e}")
            import traceback
            traceback.print_exc()

    # ── Summary Table ────────────────────────────────────────────────────
    print("\n")
    print("=" * 90)
    print("  SUMMARY: MetalNative Speedup vs PyTorch MPS (higher = better)")
    print("=" * 90)

    ops = ["RMSNorm", "QKV_Proj", "Out_Proj", "GateUp_Proj", "SwiGLU", "Down_Proj", "Softmax"]
    header = f"{'Model':<16}"
    for op in ops:
        short = op.replace("_Proj", "").replace("GateUp", "G+U")
        header += f" {short:>8}"
    print(header)
    print("-" * 90)

    for model_name, results in all_results.items():
        row = f"{model_name:<16}"
        for op in ops:
            if op in results:
                r = results[op]["ratio"]
                if r >= 2.0:
                    row += f" {r:>7.1f}x"
                else:
                    row += f" {r:>7.2f}x"
            else:
                row += f" {'N/A':>8}"
        print(row)

    print("-" * 90)
    print("  > 1.0x = MetalNative faster | < 1.0x = PyTorch MPS faster")
    print("  Fused ops (RMSNorm, SwiGLU, Softmax): single Metal kernel vs multi-op MPS")
    print("  MatMul ops: both use MPSGraph matmul, expect ~1.0x parity")
    print("=" * 90)

    # ── Detailed timing table ────────────────────────────────────────────
    print("\n")
    print("=" * 90)
    print("  DETAILED TIMINGS (ms)")
    print("=" * 90)
    print(f"{'Model':<16} {'Op':<12} {'MPS (ms)':>10} {'MN (ms)':>10} {'Speedup':>8}")
    print("-" * 90)

    for model_name, results in all_results.items():
        for i, (op, data) in enumerate(results.items()):
            label = model_name if i == 0 else ""
            r = data["ratio"]
            print(f"{label:<16} {op:<12} {data['mps']:>10.3f} {data['mn']:>10.3f} {r:>7.2f}x")
        print()

    print("=" * 90)


if __name__ == "__main__":
    main()
