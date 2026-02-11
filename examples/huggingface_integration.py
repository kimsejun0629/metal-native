#!/usr/bin/env python3
"""HuggingFace Transformers integration example.

This example demonstrates how to use MetalNative with HuggingFace transformers
for accelerated inference on Apple Silicon. It shows:
- Loading models from HuggingFace Hub
- Using the accelerate plugin
- Running inference with MetalNative backend
"""

import argparse
import time
from typing import Optional
import numpy as np

try:
    import metal_native as mn
except ImportError:
    print("Error: metal_native is required for this example")
    print("Install with: pip install -e .")
    exit(1)

# Optional dependencies
try:
    import transformers
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available")
    print("Install with: pip install transformers")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: torch not available")


def demo_model_loading():
    """Demonstrate loading a HuggingFace model with MetalNative."""
    print("=" * 80)
    print("Model Loading Demo")
    print("=" * 80)

    if not HAS_TRANSFORMERS:
        print("Error: transformers package is required")
        return

    model_name = "bert-base-uncased"
    print(f"\nLoading model: {model_name}")

    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded")

        # Load model
        print("Loading model...")
        model = AutoModel.from_pretrained(model_name)
        print("Model loaded")

        # Note: Actual integration would use accelerate plugin
        # model = model.to('metal_native')  # Move to MetalNative device

        print("\nModel info:")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        print(f"  Architecture: {model.config.model_type}")
        print(f"  Hidden size: {model.config.hidden_size}")

    except Exception as e:
        print(f"Error loading model: {e}")


def demo_inference():
    """Demonstrate inference with MetalNative backend."""
    print("\n" + "=" * 80)
    print("Inference Demo")
    print("=" * 80)

    if not HAS_TRANSFORMERS:
        print("Error: transformers package is required")
        return

    model_name = "bert-base-uncased"
    print(f"\nRunning inference with: {model_name}")

    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # Prepare input
        text = "MetalNative provides high-performance deep learning on Apple Silicon."
        print(f"\nInput text: {text}")

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        print(f"Input IDs shape: {inputs['input_ids'].shape}")

        # Run inference (CPU for now - actual integration would use MetalNative)
        print("\nRunning inference...")
        start = time.perf_counter()

        with torch.no_grad() if HAS_TORCH else contextlib.nullcontext():
            outputs = model(**inputs)

        elapsed = time.perf_counter() - start

        print(f"Inference completed in {elapsed*1000:.2f} ms")
        print(f"Output shape: {outputs.last_hidden_state.shape}")

        # Note: With MetalNative integration, usage would be:
        # model = model.to('metal_native')
        # inputs = {k: v.to('metal_native') for k, v in inputs.items()}
        # outputs = model(**inputs)

    except Exception as e:
        print(f"Error during inference: {e}")


def demo_accelerate_plugin():
    """Demonstrate using the accelerate plugin."""
    print("\n" + "=" * 80)
    print("Accelerate Plugin Demo")
    print("=" * 80)

    print("\nMetalNative accelerate plugin usage:")
    print("-" * 80)
    print("from metal_native.accelerate_plugin import register_metal_native_backend")
    print("from accelerate import Accelerator")
    print()
    print("# Register MetalNative as an accelerate backend")
    print("register_metal_native_backend()")
    print()
    print("# Create accelerator")
    print("accelerator = Accelerator(device='metal_native')")
    print()
    print("# Prepare model")
    print("model = accelerator.prepare(model)")
    print()
    print("# Run inference")
    print("with torch.no_grad():")
    print("    outputs = model(**inputs)")
    print("-" * 80)

    # Try to actually use it if available
    try:
        from metal_native.accelerate_plugin import register_metal_native_backend
        print("\nAccelerate plugin is available!")
        print("Registering backend...")
        register_metal_native_backend()
        print("Backend registered successfully")
    except ImportError:
        print("\nNote: Accelerate plugin not yet implemented")
    except Exception as e:
        print(f"\nError: {e}")


def demo_batch_inference():
    """Demonstrate batched inference."""
    print("\n" + "=" * 80)
    print("Batch Inference Demo")
    print("=" * 80)

    if not HAS_TRANSFORMERS:
        print("Error: transformers package is required")
        return

    model_name = "bert-base-uncased"
    batch_sizes = [1, 4, 8, 16]

    print(f"\nBenchmarking batched inference with: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # Sample texts
        sample_text = "This is a sample text for benchmarking."

        print("\n" + "-" * 80)
        print(f"{'Batch Size':<15} {'Time (ms)':<15} {'Throughput (samples/s)':<25}")
        print("-" * 80)

        for batch_size in batch_sizes:
            # Prepare batch
            texts = [sample_text] * batch_size
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

            # Warmup
            for _ in range(3):
                with torch.no_grad() if HAS_TORCH else contextlib.nullcontext():
                    _ = model(**inputs)

            # Benchmark
            iterations = 10
            start = time.perf_counter()

            for _ in range(iterations):
                with torch.no_grad() if HAS_TORCH else contextlib.nullcontext():
                    outputs = model(**inputs)

            elapsed = time.perf_counter() - start
            avg_time = (elapsed / iterations) * 1000
            throughput = (batch_size * iterations) / elapsed

            print(f"{batch_size:<15} {avg_time:<15.2f} {throughput:<25.2f}")

        print("-" * 80)

    except Exception as e:
        print(f"Error during batch inference: {e}")


def demo_model_conversion():
    """Demonstrate converting PyTorch models to MetalNative."""
    print("\n" + "=" * 80)
    print("Model Conversion Demo")
    print("=" * 80)

    print("\nConverting PyTorch models to MetalNative:")
    print("-" * 80)
    print("Option 1: Direct tensor conversion")
    print("  for name, param in model.named_parameters():")
    print("      param.data = mn.from_torch(param.data)")
    print()
    print("Option 2: State dict conversion")
    print("  state_dict = model.state_dict()")
    print("  metal_state_dict = {")
    print("      k: mn.from_torch(v) for k, v in state_dict.items()")
    print("  }")
    print("  model.load_state_dict(metal_state_dict)")
    print()
    print("Option 3: Using accelerate (recommended)")
    print("  from metal_native.accelerate_plugin import MetalNativeAccelerator")
    print("  accelerator = MetalNativeAccelerator()")
    print("  model = accelerator.prepare(model)")
    print("-" * 80)


def demo_memory_optimization():
    """Demonstrate memory optimization techniques."""
    print("\n" + "=" * 80)
    print("Memory Optimization Demo")
    print("=" * 80)

    print("\nMemory optimization strategies:")
    print("-" * 80)
    print("1. Gradient checkpointing:")
    print("   model.gradient_checkpointing_enable()")
    print()
    print("2. Mixed precision (bfloat16 on M1/M2):")
    print("   model = model.to(dtype=mn.bfloat16)")
    print()
    print("3. Batch size tuning:")
    print("   # Start with small batch, increase until OOM")
    print("   for batch_size in [1, 2, 4, 8, 16, 32]:")
    print("       try:")
    print("           outputs = model(inputs[:batch_size])")
    print("       except RuntimeError as e:")
    print("           optimal_batch_size = batch_size // 2")
    print("           break")
    print()
    print("4. Monitor memory usage:")
    print("   print(f'Allocated: {mn.memory_allocated() / 1e9:.2f} GB')")
    print("   print(f'Peak: {mn.max_memory_allocated() / 1e9:.2f} GB')")
    print("-" * 80)

    # Show actual memory stats
    print("\nCurrent MetalNative memory stats:")
    print(f"  Allocated: {mn.memory_allocated() / (1024**2):.2f} MB")
    print(f"  Peak:      {mn.max_memory_allocated() / (1024**2):.2f} MB")


def main():
    """Main demo entry point."""
    parser = argparse.ArgumentParser(
        description='HuggingFace integration with MetalNative',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--demo', type=str,
                       choices=['all', 'loading', 'inference', 'accelerate', 'batch', 'conversion', 'memory'],
                       default='all',
                       help='Which demo to run')
    parser.add_argument('--model', type=str, default='bert-base-uncased',
                       help='HuggingFace model to use')

    args = parser.parse_args()

    print("HuggingFace Transformers + MetalNative Integration")
    print("=" * 80)

    # Check dependencies
    if not HAS_TRANSFORMERS:
        print("Warning: transformers not installed")
        print("Install with: pip install transformers")
        print()

    # Check MetalNative
    if not mn.is_available():
        print("Error: Metal is not available on this system")
        return

    print(f"Device: {mn.device_name()}")
    print(f"MetalNative version: {mn.__version__}")
    if HAS_TRANSFORMERS:
        print(f"Transformers version: {transformers.__version__}")
    print()

    # Run selected demos
    if args.demo in ['all', 'loading']:
        demo_model_loading()

    if args.demo in ['all', 'inference']:
        demo_inference()

    if args.demo in ['all', 'accelerate']:
        demo_accelerate_plugin()

    if args.demo in ['all', 'batch']:
        demo_batch_inference()

    if args.demo in ['all', 'conversion']:
        demo_model_conversion()

    if args.demo in ['all', 'memory']:
        demo_memory_optimization()

    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    import contextlib
    main()
