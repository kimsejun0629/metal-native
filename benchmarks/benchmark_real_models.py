#!/usr/bin/env python3
"""Real model inference benchmark on Apple Silicon MPS.

Tests actual HuggingFace models with token generation on MPS device.
Measures: prefill latency, decode throughput (tok/s), peak memory.
"""

import torch
import time
import gc
import argparse
import json
import sys
from datetime import datetime

def get_device_info():
    """Get system info."""
    import platform
    info = {
        "chip": platform.processor(),
        "os": platform.mac_ver()[0],
        "pytorch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
    }
    # Get memory
    try:
        import subprocess
        result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        info["memory_gb"] = int(result.stdout.strip()) // (1024**3)
    except:
        info["memory_gb"] = "unknown"
    return info


def benchmark_model(model_name, prompt="The future of artificial intelligence is",
                    max_new_tokens=100, num_runs=3, warmup=1):
    """Benchmark a single model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")

    result = {
        "model": model_name,
        "max_new_tokens": max_new_tokens,
        "status": "success",
    }

    try:
        # Load tokenizer
        print(f"  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model to MPS
        print(f"  Loading model to MPS...")
        load_start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="mps",
            trust_remote_code=True,
        )
        model.eval()
        load_time = time.perf_counter() - load_start
        print(f"  Model loaded in {load_time:.1f}s")

        # Model info
        num_params = sum(p.numel() for p in model.parameters()) / 1e9
        result["params_b"] = round(num_params, 2)
        result["load_time_s"] = round(load_time, 1)
        print(f"  Parameters: {num_params:.2f}B")

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to("mps")
        input_len = inputs["input_ids"].shape[1]
        result["input_tokens"] = input_len

        # Warmup
        print(f"  Warming up ({warmup} runs)...")
        for _ in range(warmup):
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    use_cache=True,
                )
            torch.mps.synchronize()

        # Prefill benchmark
        print(f"  Benchmarking prefill...")
        prefill_times = []
        for _ in range(num_runs):
            torch.mps.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = model(**inputs)
            torch.mps.synchronize()
            t1 = time.perf_counter()
            prefill_times.append(t1 - t0)

        avg_prefill = sum(prefill_times) / len(prefill_times)
        result["prefill_ms"] = round(avg_prefill * 1000, 2)

        # Decode benchmark (full generation)
        print(f"  Benchmarking generation ({max_new_tokens} tokens x {num_runs} runs)...")
        gen_times = []
        gen_tokens = []
        for i in range(num_runs):
            torch.mps.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            torch.mps.synchronize()
            t1 = time.perf_counter()

            new_tokens = output_ids.shape[1] - input_len
            gen_times.append(t1 - t0)
            gen_tokens.append(new_tokens)

        avg_time = sum(gen_times) / len(gen_times)
        avg_tokens = sum(gen_tokens) / len(gen_tokens)
        tok_per_sec = avg_tokens / avg_time

        result["generation_time_ms"] = round(avg_time * 1000, 1)
        result["generated_tokens"] = round(avg_tokens)
        result["tokens_per_second"] = round(tok_per_sec, 1)
        result["ms_per_token"] = round(1000 / tok_per_sec, 2)

        # Estimate decode-only throughput (subtract prefill)
        decode_time = avg_time - avg_prefill
        if decode_time > 0 and avg_tokens > 1:
            decode_tok_per_sec = (avg_tokens - 1) / decode_time
            result["decode_tokens_per_second"] = round(decode_tok_per_sec, 1)
            result["decode_ms_per_token"] = round(1000 / decode_tok_per_sec, 2)

        # Memory usage
        try:
            mem_allocated = torch.mps.current_allocated_memory() / (1024**2)
            result["memory_mb"] = round(mem_allocated, 1)
        except:
            pass

        # Print sample output
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        preview = generated_text[:200] + "..." if len(generated_text) > 200 else generated_text
        print(f"\n  Results:")
        print(f"    Prefill:     {result['prefill_ms']:.1f} ms")
        print(f"    Generation:  {result['generation_time_ms']:.0f} ms ({result['generated_tokens']} tokens)")
        print(f"    Throughput:  {result['tokens_per_second']:.1f} tok/s (end-to-end)")
        if "decode_tokens_per_second" in result:
            print(f"    Decode only: {result['decode_tokens_per_second']:.1f} tok/s ({result['decode_ms_per_token']:.1f} ms/tok)")
        if "memory_mb" in result:
            print(f"    Memory:      {result['memory_mb']:.0f} MB")
        print(f"\n  Sample: {preview}")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"  ERROR: {e}")

    finally:
        # Cleanup
        try:
            del model
        except:
            pass
        try:
            del tokenizer
        except:
            pass
        gc.collect()
        torch.mps.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Real model inference benchmark")
    parser.add_argument("--models", nargs="+", default=None,
                       help="Specific models to test")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="Max new tokens to generate")
    parser.add_argument("--runs", type=int, default=3,
                       help="Number of benchmark runs")
    parser.add_argument("--output", type=str, default="benchmark_real_models.json",
                       help="Output JSON file")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: fewer tokens, fewer runs")
    args = parser.parse_args()

    if args.quick:
        args.max_tokens = 50
        args.runs = 2

    # Default model list (ordered by size)
    if args.models is None:
        args.models = [
            "Qwen/Qwen2.5-0.5B",
            "Qwen/Qwen2.5-1.5B",
            "Qwen/Qwen2.5-3B",
            "Qwen/Qwen2.5-7B",
            "HuggingFaceTB/SmolLM2-1.7B",
            "microsoft/Phi-3.5-mini-instruct",
        ]

    print("=" * 70)
    print("Real Model Inference Benchmark - Apple Silicon MPS")
    print("=" * 70)

    device_info = get_device_info()
    print(f"System: Apple {device_info.get('chip', 'unknown')} | {device_info.get('memory_gb', '?')} GB")
    print(f"PyTorch: {device_info['pytorch']} | MPS: {device_info['mps_available']}")
    print(f"Max tokens: {args.max_tokens} | Runs: {args.runs}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "system": device_info,
        "config": {
            "max_new_tokens": args.max_tokens,
            "num_runs": args.runs,
            "prompt": "The future of artificial intelligence is",
        },
        "models": [],
    }

    for model_name in args.models:
        result = benchmark_model(
            model_name,
            max_new_tokens=args.max_tokens,
            num_runs=args.runs,
        )
        results["models"].append(result)

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<35} {'Params':>7} {'Prefill':>9} {'Decode':>10} {'Memory':>8}")
    print(f"{'':35} {'(B)':>7} {'(ms)':>9} {'(tok/s)':>10} {'(MB)':>8}")
    print("-" * 70)

    for r in results["models"]:
        if r["status"] == "success":
            decode = r.get("decode_tokens_per_second", r.get("tokens_per_second", 0))
            mem = r.get("memory_mb", 0)
            name = r["model"].split("/")[-1][:34]
            print(f"{name:<35} {r.get('params_b', 0):>7.2f} {r.get('prefill_ms', 0):>8.1f} {decode:>10.1f} {mem:>8.0f}")
        else:
            name = r["model"].split("/")[-1][:34]
            print(f"{name:<35} {'ERROR':>7} {'-':>9} {'-':>10} {'-':>8}")

    print(f"{'='*70}\n")

    # Save results
    output_path = f"/Users/kimsejun/Documents/projects/pytorch_mps/metal_native/{args.output}"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
