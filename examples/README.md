# MetalNative Examples

This directory contains example scripts demonstrating various features and use cases of the MetalNative framework.

## Available Examples

### 1. MNIST Training (`mnist_training.py`)

Train a simple 2-layer MLP classifier on MNIST digits using MetalNative.

**Architecture:**
- Linear(784, 256) → ReLU → Linear(256, 10) → CrossEntropy

**Usage:**
```bash
# Basic training
python3 mnist_training.py

# Customize hyperparameters
python3 mnist_training.py --epochs 10 --batch-size 128 --hidden-size 512 --lr 0.01
```

**Options:**
- `--epochs N`: Number of training epochs (default: 5)
- `--batch-size N`: Batch size (default: 128)
- `--hidden-size N`: Hidden layer size (default: 256)
- `--lr FLOAT`: Learning rate (default: 0.01)

**Output:**
```
Epoch 1/5 (2.45s):
  Train Loss: 0.3456, Train Acc: 0.8923
  Test Loss:  0.2891, Test Acc:  0.9134
```

---

### 2. ResNet-18 Inference (`resnet_inference.py`)

Demonstrate inference with a ResNet-18 model for image classification.

**Components:**
- Conv2d, BatchNorm, ReLU, MaxPool, AvgPool
- Basic residual blocks
- Global average pooling
- Fully connected classifier

**Usage:**
```bash
# Basic inference
python3 resnet_inference.py

# Benchmark with multiple iterations
python3 resnet_inference.py --warmup 10 --iterations 50 --num-classes 1000
```

**Options:**
- `--num-classes N`: Number of output classes (default: 1000)
- `--warmup N`: Number of warmup runs (default: 5)
- `--iterations N`: Number of inference iterations (default: 10)

**Output:**
```
Inference Statistics:
  Average: 25.34 ± 1.23 ms
  Min:     23.45 ms
  Max:     28.67 ms
  Throughput: 39.46 images/second
```

---

### 3. LLM Text Generation (`llm_generation.py`)

Autoregressive text generation with a mini GPT-like transformer model.

**Components:**
- Token and position embeddings
- Multi-head self-attention with causal masking
- Feed-forward networks
- Layer normalization
- Language modeling head
- Top-k sampling

**Usage:**
```bash
# Basic generation
python3 llm_generation.py

# Customize model architecture
python3 llm_generation.py --d-model 256 --n-heads 8 --n-layers 4

# Control generation
python3 llm_generation.py --max-new-tokens 50 --temperature 0.8 --top-k 20
```

**Options:**
- `--vocab-size N`: Vocabulary size (default: 100)
- `--d-model N`: Model dimension (default: 128)
- `--n-heads N`: Number of attention heads (default: 4)
- `--d-ff N`: Feed-forward dimension (default: 512)
- `--n-layers N`: Number of transformer layers (default: 2)
- `--max-seq-len N`: Maximum sequence length (default: 128)
- `--max-new-tokens N`: Maximum tokens to generate (default: 20)
- `--temperature FLOAT`: Sampling temperature (default: 1.0)
- `--top-k N`: Top-k sampling (default: 10)

**Output:**
```
Prompt: the cat sat on mat
Generated text: the cat sat on mat dog ran in park quick brown fox
Generated 7 tokens in 0.45s
Throughput: 15.56 tokens/s
```

---

### 4. Profiling Demo (`profiling_demo.py`)

Demonstrate MetalNative's profiling and performance monitoring capabilities.

**Features:**
- Memory tracking (allocated, peak)
- Operation benchmarking
- TraceContext usage patterns
- Performance counter APIs
- Memory snapshots

**Usage:**
```bash
# Run all demos
python3 profiling_demo.py

# Run specific demo
python3 profiling_demo.py --demo basic
python3 profiling_demo.py --demo benchmark
python3 profiling_demo.py --demo memory
```

**Options:**
- `--demo {all,basic,trace,counters,snapshot,benchmark}`: Which demo to run

**Demos:**
- `basic`: Basic profiling operations and memory tracking
- `trace`: TraceContext usage (captures GPU traces)
- `counters`: Performance counter reading
- `snapshot`: Memory snapshot generation
- `benchmark`: Benchmark different operations with profiling

**Output:**
```
Operation Benchmark with Profiling
----------------------------------------
Matrix Mul 512x512      :   3.45 ms  |  Peak mem:  2.05 MB
Matrix Mul 1024x1024    :  15.67 ms  |  Peak mem:  8.19 MB
Element-wise Add        :   0.23 ms  |  Peak mem:  4.10 MB
```

---

### 5. HuggingFace Integration (`huggingface_integration.py`)

Integration with HuggingFace Transformers for accelerated inference.

**Features:**
- Loading models from HuggingFace Hub
- Using the accelerate plugin
- Model conversion (PyTorch → MetalNative)
- Batched inference
- Memory optimization techniques

**Usage:**
```bash
# Run all demos
python3 huggingface_integration.py

# Run specific demo
python3 huggingface_integration.py --demo loading
python3 huggingface_integration.py --demo inference
python3 huggingface_integration.py --demo batch

# Use different model
python3 huggingface_integration.py --model bert-large-uncased
```

**Options:**
- `--demo {all,loading,inference,accelerate,batch,conversion,memory}`: Which demo to run
- `--model MODEL`: HuggingFace model to use (default: bert-base-uncased)

**Demos:**
- `loading`: Load a HuggingFace model
- `inference`: Run inference with MetalNative backend
- `accelerate`: Demonstrate accelerate plugin usage
- `batch`: Batched inference benchmarking
- `conversion`: Model conversion patterns
- `memory`: Memory optimization strategies

---

## Dependencies

**Required:**
- `metal_native`: The MetalNative framework
- `numpy`: For numerical operations

**Optional (per example):**
- `torch`: For PyTorch interoperability (resnet_inference, huggingface_integration)
- `transformers`: For HuggingFace integration (huggingface_integration)

**Install dependencies:**
```bash
# Core dependencies
pip install numpy

# Optional dependencies
pip install torch transformers

# Install MetalNative
pip install -e ..
```

## Running Examples

### Quick Start

```bash
# Test installation
python3 profiling_demo.py --demo basic

# Try simple training
python3 mnist_training.py --epochs 3

# Benchmark inference
python3 resnet_inference.py --iterations 20
```

### Advanced Usage

```bash
# Train with custom architecture
python3 mnist_training.py --hidden-size 1024 --epochs 10 --lr 0.001

# Generate longer sequences
python3 llm_generation.py --n-layers 6 --d-model 512 --max-new-tokens 100

# Profile operations in detail
python3 profiling_demo.py --demo benchmark
```

## Code Structure

All examples follow these conventions:

1. **Shebang line**: `#!/usr/bin/env python3` for direct execution
2. **Module docstring**: Explains what the example demonstrates
3. **Imports with guards**: Optional dependencies wrapped in try/except
4. **Argparse**: Command-line options for customization
5. **`if __name__ == "__main__"`**: Guard for standalone execution
6. **Type hints**: For clarity and IDE support

## Learning Path

**Beginners:**
1. Start with `profiling_demo.py` to understand basic operations
2. Try `mnist_training.py` to see a complete training loop
3. Explore `resnet_inference.py` for inference patterns

**Intermediate:**
1. Study `llm_generation.py` for transformer implementation
2. Experiment with `huggingface_integration.py` for production workflows

**Advanced:**
1. Modify examples for your own models
2. Combine patterns from multiple examples
3. Add custom profiling and optimization

## Common Patterns

### Loading Data
```python
# NumPy → MetalNative (copy)
tensor = mn.from_numpy(numpy_array)

# PyTorch → MetalNative (zero-copy via DLPack)
tensor = mn.from_torch(torch_tensor)
```

### Model Definition
```python
# Initialize weights
self.weight = mn.from_numpy(np.random.randn(in_dim, out_dim).astype(np.float32))

# Forward pass
output = input @ self.weight + self.bias
```

### Training Loop
```python
for epoch in range(epochs):
    for batch in dataloader:
        # Forward
        logits = model.forward(batch_x)
        loss = criterion(logits, batch_y)

        # Backward (when autograd available)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Synchronization
```python
# Ensure GPU operations complete
output = model(input)
mn.synchronize()

# Now safe to read output
result = output.numpy()
```

## Troubleshooting

**Issue**: "metal_native is required for this example"
- **Solution**: Install with `pip install -e ..` from examples directory

**Issue**: "transformers not available"
- **Solution**: Install with `pip install transformers` (only needed for huggingface_integration.py)

**Issue**: Slow performance
- **Solution**: Ensure warmup iterations are used, check for thermal throttling

**Issue**: Out of memory
- **Solution**: Reduce batch size, model size, or sequence length

**Issue**: Import errors
- **Solution**: Check that metal_native C extension is built (`pip install -e ..` in parent directory)

## Tips

1. **Start small**: Test with small models/batches first
2. **Profile early**: Use profiling_demo to understand performance characteristics
3. **Check memory**: Monitor `mn.memory_allocated()` regularly
4. **Synchronize**: Always call `mn.synchronize()` before timing or reading results
5. **Guard imports**: Use try/except for optional dependencies

## Contributing

To add a new example:

1. Follow the code structure conventions above
2. Add comprehensive docstrings and comments
3. Include `--help` documentation via argparse
4. Test with and without optional dependencies
5. Update this README with the new example
6. Add to `__init__.py` for discoverability
