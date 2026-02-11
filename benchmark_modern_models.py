#!/usr/bin/env python3
"""
Benchmark modern open-source LLMs on Apple Silicon MPS.
Tests real pretrained models from HuggingFace with actual weights.

Target: Apple M4 Max 36GB Unified Memory
Models selected: All released 2024+, production-grade open-source LLMs
"""

import torch
import time
import json
import gc
import os
import sys
import traceback
from dataclasses import dataclass, asdict
from typing import Optional

# ── Helpers ──────────────────────────────────────────────────────────────────

@dataclass
class ModelBenchResult:
    model_name: str
    model_id: str
    params: str
    dtype: str
    release_year: int
    # Forward pass
    forward_time_ms: float = 0.0
    forward_std_ms: float = 0.0
    # Generation
    tokens_per_sec: float = 0.0
    gen_time_ms: float = 0.0
    gen_tokens: int = 0
    # Memory
    gpu_memory_mb: float = 0.0
    # vs CPU
    cpu_forward_ms: float = 0.0
    speedup_vs_cpu: float = 0.0
    # Status
    status: str = "pending"
    error: Optional[str] = None


def get_gpu_memory_mb():
    """Get current MPS memory usage in MB."""
    if hasattr(torch.mps, 'current_allocated_memory'):
        return torch.mps.current_allocated_memory() / 1024 / 1024
    return 0.0


def cleanup():
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()


def benchmark_forward(model, input_ids, device, warmup=3, runs=10):
    """Benchmark forward pass latency."""
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(input_ids)
            if device.type == 'mps':
                torch.mps.synchronize()

        # Timed runs
        times = []
        for _ in range(runs):
            if device.type == 'mps':
                torch.mps.synchronize()
            t0 = time.perf_counter()
            _ = model(input_ids)
            if device.type == 'mps':
                torch.mps.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    return avg, std


def benchmark_generation(model, input_ids, device, max_new_tokens=50, warmup=1, runs=3):
    """Benchmark token generation throughput."""
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
            if device.type == 'mps':
                torch.mps.synchronize()

        # Timed runs
        times = []
        for _ in range(runs):
            if device.type == 'mps':
                torch.mps.synchronize()
            t0 = time.perf_counter()
            out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
            if device.type == 'mps':
                torch.mps.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            gen_tokens = out.shape[1] - input_ids.shape[1]

        avg_ms = sum(times) / len(times)
        tokens_per_sec = (gen_tokens / (avg_ms / 1000)) if avg_ms > 0 else 0

    return avg_ms, tokens_per_sec, gen_tokens


def benchmark_cpu_forward(model_id, input_ids_cpu, dtype, timeout_sec=120):
    """Benchmark forward pass on CPU with timeout."""
    from transformers import AutoModelForCausalLM

    # For CPU, always use float32
    try:
        cpu_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
        cpu_model.eval()
    except Exception as e:
        print(f"    [CPU] Failed to load: {e}")
        return -1.0

    with torch.no_grad():
        # Single warmup
        try:
            _ = cpu_model(input_ids_cpu)
        except Exception:
            del cpu_model
            cleanup()
            return -1.0

        # Timed run (single, since CPU is slow)
        t0 = time.perf_counter()
        _ = cpu_model(input_ids_cpu)
        t1 = time.perf_counter()
        cpu_ms = (t1 - t0) * 1000

    del cpu_model
    cleanup()
    return cpu_ms


# ── Model Registry ───────────────────────────────────────────────────────────

MODELS = [
    # ── Lightweight (0.5-1.5B) — All open-access, no auth required ──
    {
        "name": "Qwen2.5-0.5B",
        "id": "Qwen/Qwen2.5-0.5B",
        "params": "0.5B",
        "dtype": "float16",
        "year": 2024,
        "cpu_test": True,
    },
    {
        "name": "SmolLM2-1.7B",
        "id": "HuggingFaceTB/SmolLM2-1.7B",
        "params": "1.7B",
        "dtype": "float16",
        "year": 2024,
        "cpu_test": True,
    },
    {
        "name": "Qwen2.5-1.5B",
        "id": "Qwen/Qwen2.5-1.5B",
        "params": "1.5B",
        "dtype": "float16",
        "year": 2024,
        "cpu_test": True,
    },
    # ── Medium (3-4B) ──
    {
        "name": "Qwen2.5-3B",
        "id": "Qwen/Qwen2.5-3B",
        "params": "3B",
        "dtype": "float16",
        "year": 2024,
        "cpu_test": True,
    },
    {
        "name": "Phi-3.5-Mini-3.8B",
        "id": "microsoft/Phi-3.5-mini-instruct",
        "params": "3.8B",
        "dtype": "float16",
        "year": 2024,
        "cpu_test": False,  # Too slow on CPU
    },
    # ── Large (7B+) ──
    {
        "name": "Qwen2.5-7B",
        "id": "Qwen/Qwen2.5-7B",
        "params": "7B",
        "dtype": "float16",
        "year": 2024,
        "cpu_test": False,
    },
    {
        "name": "Qwen2.5-14B",
        "id": "Qwen/Qwen2.5-14B",
        "params": "14B",
        "dtype": "float16",
        "year": 2024,
        "cpu_test": False,
    },
]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  MetalNative — Modern LLM Benchmark Suite")
    print("  Apple Silicon MPS vs CPU (Real Pretrained Models)")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n  Device: {device}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Python: {sys.version.split()[0]}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    print(f"  Transformers: {transformers.__version__}")
    print(f"  Models to test: {len(MODELS)}")
    print()

    results = []
    prompt = "The future of artificial intelligence will"
    seq_len = 64  # Input sequence length for forward pass

    for i, model_info in enumerate(MODELS, 1):
        name = model_info["name"]
        model_id = model_info["id"]
        dtype_str = model_info["dtype"]
        model_dtype = torch.float16 if dtype_str == "float16" else torch.float32

        result = ModelBenchResult(
            model_name=name,
            model_id=model_id,
            params=model_info["params"],
            dtype=dtype_str,
            release_year=model_info["year"],
        )

        print(f"[{i}/{len(MODELS)}] {name} ({model_info['params']}, {dtype_str})")
        print(f"  Loading from: {model_id}")

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Prepare input
            input_ids_mps = tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=seq_len,
                truncation=True,
            )["input_ids"].to(device)

            input_ids_cpu = input_ids_mps.cpu()

            # Load model to MPS
            print(f"  Loading model to MPS ({dtype_str})...")
            t_load_start = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=model_dtype,
                device_map="mps",
                trust_remote_code=True,
            )
            model.eval()
            t_load = time.perf_counter() - t_load_start
            print(f"  Loaded in {t_load:.1f}s")

            # GPU memory
            if device.type == 'mps':
                torch.mps.synchronize()
            result.gpu_memory_mb = get_gpu_memory_mb()
            print(f"  GPU Memory: {result.gpu_memory_mb:.0f} MB")

            # Forward pass benchmark
            print(f"  Benchmarking forward pass (seq_len={seq_len})...")
            fwd_ms, fwd_std = benchmark_forward(model, input_ids_mps, device, warmup=3, runs=10)
            result.forward_time_ms = round(fwd_ms, 2)
            result.forward_std_ms = round(fwd_std, 2)
            print(f"  Forward: {fwd_ms:.2f} ± {fwd_std:.2f} ms")

            # Generation benchmark
            print(f"  Benchmarking generation (50 tokens)...")
            gen_ms, tok_s, gen_tok = benchmark_generation(
                model, input_ids_mps, device, max_new_tokens=50, warmup=1, runs=3
            )
            result.gen_time_ms = round(gen_ms, 2)
            result.tokens_per_sec = round(tok_s, 1)
            result.gen_tokens = gen_tok
            print(f"  Generation: {tok_s:.1f} tok/s ({gen_tok} tokens in {gen_ms:.0f} ms)")

            # Free MPS model
            del model
            cleanup()

            # CPU comparison (only for smaller models)
            if model_info.get("cpu_test", False):
                print(f"  Benchmarking CPU forward pass...")
                cpu_ms = benchmark_cpu_forward(model_id, input_ids_cpu, dtype_str)
                if cpu_ms > 0:
                    result.cpu_forward_ms = round(cpu_ms, 2)
                    result.speedup_vs_cpu = round(cpu_ms / fwd_ms, 2)
                    print(f"  CPU Forward: {cpu_ms:.2f} ms → {result.speedup_vs_cpu:.2f}x speedup")
                else:
                    print(f"  CPU Forward: skipped (load failed)")
            else:
                print(f"  CPU Forward: skipped (model too large)")

            result.status = "success"
            print(f"  ✓ Done")

        except Exception as e:
            result.status = "error"
            result.error = str(e)
            print(f"  ✗ Error: {e}")
            traceback.print_exc()

        results.append(result)
        cleanup()
        print()

    # ── Summary Table ────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  RESULTS SUMMARY — Modern Open-Source LLMs on Apple Silicon MPS")
    print("=" * 90)
    print(f"{'Model':<22} {'Params':>6} {'Forward':>10} {'tok/s':>8} {'GPU MB':>8} {'Speedup':>8}")
    print("-" * 90)

    for r in results:
        if r.status == "success":
            speedup_str = f"{r.speedup_vs_cpu:.1f}x" if r.speedup_vs_cpu > 0 else "—"
            print(
                f"{r.model_name:<22} {r.params:>6} "
                f"{r.forward_time_ms:>8.2f}ms {r.tokens_per_sec:>7.1f} "
                f"{r.gpu_memory_mb:>7.0f} {speedup_str:>8}"
            )
        else:
            print(f"{r.model_name:<22} {r.params:>6}  FAILED: {r.error}")

    print("-" * 90)
    print(f"  Platform: Apple M4 Max, 36GB Unified Memory")
    print(f"  PyTorch {torch.__version__} | Input seq_len={seq_len} | Generation: 50 tokens")
    print()

    # ── Save JSON ────────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "benchmark": "Modern LLM Benchmark Suite",
            "device": "Apple M4 Max 36GB",
            "pytorch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "input_seq_len": seq_len,
            "gen_tokens": 50,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": [asdict(r) for r in results],
    }

    out_path = os.path.join(os.path.dirname(__file__), "benchmark_modern_models.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
