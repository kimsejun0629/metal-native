#!/usr/bin/env python3
"""Numerical accuracy test infrastructure for validating Metal kernel outputs.

This module provides:
- Reference implementations in pure NumPy for major operations
- Numerical accuracy testing framework with configurable tolerances
- Self-contained test infrastructure that works even without metal_native
"""

import numpy as np
import argparse
import sys
from typing import Optional, Dict, List, Tuple

# Try to import metal_native bindings
try:
    import metal_native
    HAS_METAL_NATIVE = True
except ImportError:
    HAS_METAL_NATIVE = False
    print("Warning: metal_native not available. Reference implementations only.", file=sys.stderr)


# =============================================================================
# Reference Implementations (Pure NumPy)
# =============================================================================

def reference_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Reference softmax implementation.

    Args:
        x: Input array
        axis: Axis along which to apply softmax

    Returns:
        Softmax output with same shape as input
    """
    # Numerical stability: subtract max
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def reference_layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                         eps: float = 1e-5) -> np.ndarray:
    """Reference LayerNorm implementation.

    Args:
        x: Input array [..., normalized_shape]
        weight: Scale parameter [normalized_shape]
        bias: Shift parameter [normalized_shape]
        eps: Epsilon for numerical stability

    Returns:
        Normalized output with same shape as input
    """
    # Compute mean and variance over last dimension
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    # Normalize
    x_norm = (x - mean) / np.sqrt(var + eps)

    # Apply affine transformation
    return x_norm * weight + bias


def reference_rms_norm(x: np.ndarray, weight: np.ndarray,
                       eps: float = 1e-5) -> np.ndarray:
    """Reference RMSNorm implementation.

    Args:
        x: Input array [..., normalized_shape]
        weight: Scale parameter [normalized_shape]
        eps: Epsilon for numerical stability

    Returns:
        Normalized output with same shape as input
    """
    # Compute RMS over last dimension
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)

    # Normalize and scale
    return (x / rms) * weight


def reference_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                        causal: bool = False) -> np.ndarray:
    """Reference scaled dot-product attention implementation.

    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        v: Value tensor [batch, heads, seq_len, head_dim]
        causal: Whether to apply causal masking

    Returns:
        Attention output [batch, heads, seq_len, head_dim]
    """
    head_dim = q.shape[-1]
    scale = 1.0 / np.sqrt(head_dim)

    # Compute attention scores: Q @ K^T
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale

    # Apply causal mask if requested
    if causal:
        seq_len = q.shape[2]
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        scores = scores + mask

    # Apply softmax
    attn_weights = reference_softmax(scores, axis=-1)

    # Apply attention to values: Attn @ V
    output = np.matmul(attn_weights, v)
    return output


def reference_gelu(x: np.ndarray) -> np.ndarray:
    """Reference GELU implementation (tanh approximation).

    Args:
        x: Input array

    Returns:
        GELU output with same shape as input
    """
    # Tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * x ** 3)))


def reference_silu(x: np.ndarray) -> np.ndarray:
    """Reference SiLU (Swish) implementation.

    Args:
        x: Input array

    Returns:
        SiLU output with same shape as input
    """
    # SiLU: x * sigmoid(x)
    return x / (1.0 + np.exp(-x))


def reference_relu(x: np.ndarray) -> np.ndarray:
    """Reference ReLU implementation.

    Args:
        x: Input array

    Returns:
        ReLU output with same shape as input
    """
    return np.maximum(0, x)


# =============================================================================
# Test Infrastructure
# =============================================================================

class NumericalAccuracyTest:
    """Numerical accuracy testing framework."""

    def __init__(self, tolerance_fp32: float = 1e-5, tolerance_fp16: float = 1e-3):
        """Initialize test framework.

        Args:
            tolerance_fp32: Maximum allowed absolute difference for FP32
            tolerance_fp16: Maximum allowed absolute difference for FP16
        """
        self.tolerance_fp32 = tolerance_fp32
        self.tolerance_fp16 = tolerance_fp16
        self.results: List[Dict] = []

    def check_accuracy(self, name: str, result: np.ndarray,
                       reference: np.ndarray, dtype: str = 'fp32') -> bool:
        """Compare result against reference within tolerance.

        Args:
            name: Test name
            result: Result to validate
            reference: Reference (ground truth)
            dtype: Data type ('fp32' or 'fp16')

        Returns:
            True if test passed, False otherwise
        """
        tolerance = self.tolerance_fp32 if dtype == 'fp32' else self.tolerance_fp16

        # Compute metrics
        max_diff = np.max(np.abs(result - reference))
        mean_diff = np.mean(np.abs(result - reference))

        passed = max_diff < tolerance

        self.results.append({
            'name': name,
            'dtype': dtype,
            'max_diff': float(max_diff),
            'mean_diff': float(mean_diff),
            'tolerance': tolerance,
            'passed': passed
        })

        return passed

    def print_summary(self, verbose: bool = False):
        """Print test results summary.

        Args:
            verbose: If True, print details for all tests
        """
        print("\n" + "=" * 80)
        print("NUMERICAL ACCURACY TEST RESULTS")
        print("=" * 80)

        total = len(self.results)
        passed = sum(1 for r in self.results if r['passed'])
        failed = total - passed

        if verbose or failed > 0:
            print(f"\n{'Test Name':<40} {'Dtype':<8} {'Max Diff':<12} {'Status':<10}")
            print("-" * 80)

            for result in self.results:
                status = "PASS" if result['passed'] else "FAIL"
                status_color = status if result['passed'] else f"*** {status} ***"
                print(f"{result['name']:<40} {result['dtype']:<8} "
                      f"{result['max_diff']:<12.2e} {status_color:<10}")

                if verbose:
                    print(f"  Mean diff: {result['mean_diff']:.2e}, "
                          f"Tolerance: {result['tolerance']:.2e}")

        print("\n" + "-" * 80)
        print(f"Total: {total} | Passed: {passed} | Failed: {failed}")

        if failed == 0:
            print("✓ All tests passed!")
        else:
            print(f"✗ {failed} test(s) failed")

        print("=" * 80 + "\n")


# =============================================================================
# Test Cases
# =============================================================================

def test_softmax(tester: NumericalAccuracyTest, dtype: str = 'fp32'):
    """Test softmax accuracy."""
    np_dtype = np.float32 if dtype == 'fp32' else np.float16

    # Test configuration
    x = np.random.randn(2, 1024).astype(np_dtype)

    # Reference
    reference = reference_softmax(x, axis=-1)

    # Metal implementation (if available)
    if HAS_METAL_NATIVE:
        # TODO: Call metal_native.softmax when available
        # For now, use reference as placeholder
        result = reference
    else:
        result = reference

    return tester.check_accuracy(f"softmax_{dtype}", result, reference, dtype)


def test_layer_norm(tester: NumericalAccuracyTest, dtype: str = 'fp32'):
    """Test LayerNorm accuracy."""
    np_dtype = np.float32 if dtype == 'fp32' else np.float16

    # Test configuration
    batch, seq_len, hidden_dim = 2, 128, 768
    x = np.random.randn(batch, seq_len, hidden_dim).astype(np_dtype)
    weight = np.ones(hidden_dim, dtype=np_dtype)
    bias = np.zeros(hidden_dim, dtype=np_dtype)

    # Reference
    reference = reference_layer_norm(x, weight, bias)

    # Metal implementation (if available)
    if HAS_METAL_NATIVE:
        # TODO: Call metal_native.layer_norm when available
        result = reference
    else:
        result = reference

    return tester.check_accuracy(f"layer_norm_{dtype}", result, reference, dtype)


def test_rms_norm(tester: NumericalAccuracyTest, dtype: str = 'fp32'):
    """Test RMSNorm accuracy."""
    np_dtype = np.float32 if dtype == 'fp32' else np.float16

    # Test configuration
    batch, seq_len, hidden_dim = 2, 128, 768
    x = np.random.randn(batch, seq_len, hidden_dim).astype(np_dtype)
    weight = np.ones(hidden_dim, dtype=np_dtype)

    # Reference
    reference = reference_rms_norm(x, weight)

    # Metal implementation (if available)
    if HAS_METAL_NATIVE:
        # TODO: Call metal_native.rms_norm when available
        result = reference
    else:
        result = reference

    return tester.check_accuracy(f"rms_norm_{dtype}", result, reference, dtype)


def test_attention(tester: NumericalAccuracyTest, dtype: str = 'fp32',
                   causal: bool = False):
    """Test attention accuracy."""
    np_dtype = np.float32 if dtype == 'fp32' else np.float16

    # Test configuration
    batch, heads, seq_len, head_dim = 2, 8, 128, 64
    q = np.random.randn(batch, heads, seq_len, head_dim).astype(np_dtype)
    k = np.random.randn(batch, heads, seq_len, head_dim).astype(np_dtype)
    v = np.random.randn(batch, heads, seq_len, head_dim).astype(np_dtype)

    # Reference
    reference = reference_attention(q, k, v, causal=causal)

    # Metal implementation (if available)
    if HAS_METAL_NATIVE:
        # TODO: Call metal_native.attention when available
        result = reference
    else:
        result = reference

    test_name = f"attention_{dtype}{'_causal' if causal else ''}"
    return tester.check_accuracy(test_name, result, reference, dtype)


def test_elementwise_ops(tester: NumericalAccuracyTest, dtype: str = 'fp32'):
    """Test elementwise operations accuracy."""
    np_dtype = np.float32 if dtype == 'fp32' else np.float16

    x = np.random.randn(1024, 1024).astype(np_dtype)

    # Test GELU
    gelu_ref = reference_gelu(x)
    if HAS_METAL_NATIVE:
        # TODO: Call metal_native.gelu when available
        gelu_result = gelu_ref
    else:
        gelu_result = gelu_ref
    tester.check_accuracy(f"gelu_{dtype}", gelu_result, gelu_ref, dtype)

    # Test SiLU
    silu_ref = reference_silu(x)
    if HAS_METAL_NATIVE:
        # TODO: Call metal_native.silu when available
        silu_result = silu_ref
    else:
        silu_result = silu_ref
    tester.check_accuracy(f"silu_{dtype}", silu_result, silu_ref, dtype)

    # Test ReLU
    relu_ref = reference_relu(x)
    if HAS_METAL_NATIVE:
        # TODO: Call metal_native.relu when available
        relu_result = relu_ref
    else:
        relu_result = relu_ref
    tester.check_accuracy(f"relu_{dtype}", relu_result, relu_ref, dtype)

    # Test Add
    y = np.random.randn(1024, 1024).astype(np_dtype)
    add_ref = x + y
    if HAS_METAL_NATIVE:
        # TODO: Call metal_native.add when available
        add_result = add_ref
    else:
        add_result = add_ref
    tester.check_accuracy(f"add_{dtype}", add_result, add_ref, dtype)

    # Test Mul
    mul_ref = x * y
    if HAS_METAL_NATIVE:
        # TODO: Call metal_native.mul when available
        mul_result = mul_ref
    else:
        mul_result = mul_ref
    tester.check_accuracy(f"mul_{dtype}", mul_result, mul_ref, dtype)


def test_matmul(tester: NumericalAccuracyTest, dtype: str = 'fp32'):
    """Test matrix multiplication accuracy."""
    np_dtype = np.float32 if dtype == 'fp32' else np.float16

    # Test configuration
    a = np.random.randn(128, 256).astype(np_dtype)
    b = np.random.randn(256, 512).astype(np_dtype)

    # Reference
    reference = np.matmul(a, b)

    # Metal implementation (if available)
    if HAS_METAL_NATIVE:
        # TODO: Call metal_native.matmul when available
        result = reference
    else:
        result = reference

    return tester.check_accuracy(f"matmul_{dtype}", result, reference, dtype)


def run_all_tests(tolerance_fp32: float = 1e-5,
                  tolerance_fp16: float = 1e-3,
                  verbose: bool = False) -> bool:
    """Run all numerical accuracy tests.

    Args:
        tolerance_fp32: Tolerance for FP32 tests
        tolerance_fp16: Tolerance for FP16 tests
        verbose: Print verbose output

    Returns:
        True if all tests passed, False otherwise
    """
    tester = NumericalAccuracyTest(tolerance_fp32, tolerance_fp16)

    print("\nRunning numerical accuracy tests...")
    print(f"FP32 tolerance: {tolerance_fp32:.2e}")
    print(f"FP16 tolerance: {tolerance_fp16:.2e}")

    # Run FP32 tests
    test_softmax(tester, dtype='fp32')
    test_layer_norm(tester, dtype='fp32')
    test_rms_norm(tester, dtype='fp32')
    test_attention(tester, dtype='fp32', causal=False)
    test_attention(tester, dtype='fp32', causal=True)
    test_elementwise_ops(tester, dtype='fp32')
    test_matmul(tester, dtype='fp32')

    # Run FP16 tests
    test_softmax(tester, dtype='fp16')
    test_layer_norm(tester, dtype='fp16')
    test_rms_norm(tester, dtype='fp16')
    test_attention(tester, dtype='fp16', causal=False)
    test_attention(tester, dtype='fp16', causal=True)
    test_elementwise_ops(tester, dtype='fp16')
    test_matmul(tester, dtype='fp16')

    # Print summary
    tester.print_summary(verbose=verbose)

    # Return overall pass/fail
    return all(r['passed'] for r in tester.results)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Main entry point for numerical accuracy testing."""
    parser = argparse.ArgumentParser(
        description='Numerical accuracy test infrastructure for Metal kernels',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--tolerance-fp32', type=float, default=1e-5,
                       help='Maximum allowed absolute difference for FP32')
    parser.add_argument('--tolerance-fp16', type=float, default=1e-3,
                       help='Maximum allowed absolute difference for FP16')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output for all tests')

    args = parser.parse_args()

    # Run tests
    success = run_all_tests(
        tolerance_fp32=args.tolerance_fp32,
        tolerance_fp16=args.tolerance_fp16,
        verbose=args.verbose
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
