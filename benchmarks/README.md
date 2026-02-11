# MetalNative Benchmark Suite

This directory contains benchmarks comparing MetalNative performance against NumPy and PyTorch MPS across various operations.

## Available Benchmarks

### 1. Matrix Multiplication (`benchmark_matmul.py`)

Benchmarks matrix multiplication at various sizes (128x128 to 4096x4096).

**Usage:**
```bash
python3 benchmark_matmul.py --sizes 512 1024 2048 --iterations 10 --warmup 3
python3 benchmark_matmul.py --output results_matmul.md
```

**Metrics:**
- Execution time (ms)
- GFLOPS (billions of floating-point operations per second)
- Speedup vs NumPy

### 2. Attention (`benchmark_attention.py`)

Benchmarks scaled dot-product attention (the core operation in transformers) at various configurations.

**Usage:**
```bash
python3 benchmark_attention.py --seq-lens 128 256 512 1024 --head-dims 64 128
python3 benchmark_attention.py --num-heads 8 12 16 --batch 4
python3 benchmark_attention.py --output results_attention.md
```

**Metrics:**
- Execution time (ms)
- Memory usage (MB)
- Throughput (tokens/second)
- Memory savings (FlashAttention vs naive)

### 3. Convolution (`benchmark_conv.py`)

Benchmarks 2D convolution at common ResNet and VGG configurations.

**Usage:**
```bash
python3 benchmark_conv.py --preset resnet --batch 1
python3 benchmark_conv.py --preset vgg --batch 4
python3 benchmark_conv.py --preset all --output results_conv.md
```

**Presets:**
- `resnet`: ResNet-18/50 configurations (7x7, 3x3, 1x1 convs)
- `vgg`: VGG configurations (3x3 convs at various depths)
- `all`: Both presets

### 4. Transformer Block (`benchmark_transformer.py`)

Benchmarks full transformer blocks (attention + FFN + norms) at various model scales.

**Usage:**
```bash
python3 benchmark_transformer.py --models GPT2-Small GPT2-Medium
python3 benchmark_transformer.py --models all --output results_transformer.md
```

**Model Configurations:**
- `GPT2-Small`: 768 dim, 12 heads (125M params)
- `GPT2-Medium`: 1024 dim, 16 heads (350M params)
- `GPT2-Large`: 1280 dim, 20 heads (774M params)
- `GPT2-XL`: 1600 dim, 25 heads (1.5B params)
- `Llama-7B-Block`: 4096 dim, 32 heads (7B scale)

### 5. Run All Benchmarks (`run_all.py`)

Runner script that executes all benchmarks and generates a comprehensive report.

**Usage:**
```bash
# Run all benchmarks with default settings
python3 run_all.py

# Customize iterations and output
python3 run_all.py --iterations 20 --warmup 5 --output my_results.md

# Skip specific benchmarks
python3 run_all.py --skip-attention --skip-transformer

# Customize configurations
python3 run_all.py --matmul-sizes 256 512 1024 --conv-preset resnet
```

**Options:**
- `--iterations N`: Number of iterations per benchmark (default: 10)
- `--warmup N`: Number of warmup iterations (default: 3)
- `--output FILE`: Output file for markdown report
- `--skip-matmul`, `--skip-attention`, `--skip-conv`, `--skip-transformer`: Skip specific benchmarks

## Dependencies

**Required:**
- `metal_native`: The MetalNative framework
- `numpy`: For baseline comparisons

**Optional:**
- `torch`: For PyTorch MPS comparisons (highly recommended)

**Install dependencies:**
```bash
pip install numpy torch
pip install -e ..  # Install metal_native from parent directory
```

## Output Format

All benchmarks support `--output` flag to save results in markdown format with comparison tables.

**Example output:**
```markdown
| Size | NumPy (ms) | MetalNative (ms) | PyTorch MPS (ms) | MN Speedup | MPS Speedup |
|------|------------|------------------|------------------|------------|-------------|
| 512  | 15.234     | 2.456            | 2.891            | 6.20x      | 5.27x       |
| 1024 | 125.678    | 12.345           | 13.456           | 10.18x     | 9.34x       |
```

## Tips

1. **Warmup**: Use at least 3 warmup iterations to ensure Metal kernels are compiled and cached.

2. **Consistency**: Close other applications and disable background tasks for consistent results.

3. **Memory**: Monitor memory usage with `--iterations 1` first for large benchmarks.

4. **Comparison**: Compare against PyTorch MPS when available for a fair GPU-to-GPU comparison.

5. **Reproducibility**: Results may vary between runs due to system load, thermal throttling, etc.

## Example Workflow

```bash
# Quick test with small sizes
python3 benchmark_matmul.py --sizes 128 256 512 --iterations 5

# Full benchmark suite
python3 run_all.py --iterations 20 --warmup 5 --output full_benchmark.md

# Focus on transformer workloads
python3 benchmark_attention.py --seq-lens 512 1024 2048 --output attn.md
python3 benchmark_transformer.py --models GPT2-Small GPT2-Medium --output trans.md
```

## Interpreting Results

- **Speedup > 1.0**: MetalNative is faster than the baseline
- **GFLOPS**: Higher is better (more operations per second)
- **Memory (MB)**: Lower is better (FlashAttention reduces memory by avoiding materialized attention matrix)
- **Throughput (tokens/s)**: Higher is better

## Troubleshooting

**Issue**: "metal_native not available"
- **Solution**: Install MetalNative with `pip install -e ..` from this directory

**Issue**: "PyTorch MPS not available"
- **Solution**: Install PyTorch 2.0+ or ignore MPS comparisons (NumPy comparison still works)

**Issue**: Out of memory errors
- **Solution**: Reduce batch size, sequence length, or model size

**Issue**: Very slow benchmarks
- **Solution**: Reduce `--iterations` for quick testing
