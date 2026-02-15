<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# benchmarks
## Purpose
Performance benchmarking suite comparing MetalNative against NumPy CPU and PyTorch MPS across various operations. Measures execution time, GFLOPS, memory usage, and speedup factors for matmul, attention, convolution, transformers, and real model workloads. Includes comprehensive benchmark runner with markdown report generation and visualization tools.

## Key Files
| File | Description |
|------|-------------|
| benchmark_comprehensive.py | Comprehensive MPS benchmark comparing PyTorch CPU/MPS/optimized patterns. Tests matmul, attention (flash vs naive), softmax, layernorm, elementwise ops, reductions, conv2d, and full transformer blocks. Outputs JSON results for visualization. |
| benchmark_matmul.py | Matrix multiplication benchmarks at sizes 128x128 to 4096x4096. Compares NumPy, MetalNative, PyTorch MPS. Reports GFLOPS and speedup metrics. |
| benchmark_attention.py | Scaled dot-product attention benchmarks with flash attention optimization. Tests various sequence lengths (128-2048), head counts (8-32), and batch sizes. Measures memory savings (O(S²) → O(S)) and throughput (tokens/s). |
| benchmark_conv.py | 2D convolution benchmarks using ResNet and VGG configurations. Tests 7x7, 3x3, 1x1 kernels with various channels and image sizes. |
| benchmark_transformer.py | Full transformer block benchmarks (attention + FFN + norms) for GPT2-Small/Medium/Large/XL and Llama-7B block configurations. End-to-end performance measurement. |
| benchmark_model_ops.py | Real model operation patterns: RMSNorm, SwiGLU, fused residual+norm, fused QKV projection, RoPE. Mirrors production LLM architectures. |
| benchmark_real_models.py | Full model inference benchmarks for production architectures (ResNet, BERT, GPT variants). End-to-end latency and throughput. |
| run_all.py | Master runner script executing all benchmarks with configurable iterations, warmup, and output formats. Generates comprehensive markdown reports with comparison tables. |
| visualize_benchmark.py | Visualization script for benchmark results. Generates charts (speedup, GFLOPS, memory usage, scaling). |
| README.md | Complete benchmark suite documentation with usage examples, metrics explanations, and troubleshooting. |
| CMakeLists.txt | Build configuration for C++ benchmark executables (if any). |

## For AI Agents
### Working In This Directory
- All benchmarks are Python scripts using argparse for CLI options
- Use `--iterations N --warmup N` to control measurement quality vs speed
- Output markdown reports with `--output FILE.md`
- Run quick tests first: `--iterations 5 --warmup 2`
- Check torch.backends.mps.is_available() before running MPS comparisons
- benchmark_comprehensive.py is the main reference implementation showing all optimization patterns

### Testing Requirements
- Requires: numpy, metal_native installed
- Optional: torch (for MPS comparisons - highly recommended)
- Install: `pip install numpy torch && pip install -e ..`
- Validate: Run `python3 benchmark_matmul.py --sizes 128 256 --iterations 5` as smoke test
- Full suite: `python3 run_all.py` takes 5-10 minutes

### Common Patterns
- Use `benchmark_fn(lambda: operation(), warmup, iterations, sync=True)` for timing
- Call `sync_mps()` or `torch.mps.synchronize()` before measurement to ensure GPU completion
- Calculate GFLOPS: `(flops / 1e9) / (time_ms / 1000)`
- Use `np.median(times)` for robust timing (reduces variance)
- Memory estimation: naive attention = B×H×S² bytes, flash = B×H×S×D bytes

## Dependencies
### Internal
- `../python/metal_native`: MetalNative Python bindings (device, tensors, ops)
- `../build/shaders/metal_native.metallib`: Compiled Metal kernels

### External
- numpy: Reference CPU implementations
- torch: PyTorch MPS comparisons (optional but recommended)
- matplotlib: Visualization (visualize_benchmark.py only)
- dataclasses, json, argparse: Standard library

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
