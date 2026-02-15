#!/usr/bin/env python3
"""Test texture-backed attention implementation."""

import sys
import os
import numpy as np

# Add build directory to path for importing metal_native
build_path = os.path.join(os.path.dirname(__file__), '../../build/python')
sys.path.insert(0, build_path)

import metal_native as mn


def test_texture_attention_fp32():
    """Test texture-backed attention with FP32 MHA."""
    print("Testing texture-backed attention (FP32 MHA)...")

    # Small test case: batch=1, heads=2, seq=8, head_dim=64
    batch = 1
    num_heads = 2
    seq_len = 8
    head_dim = 64

    # Create random inputs
    np.random.seed(42)
    q_data = np.random.randn(batch, num_heads, seq_len, head_dim).astype(np.float32)
    k_data = np.random.randn(batch, num_heads, seq_len, head_dim).astype(np.float32)
    v_data = np.random.randn(batch, num_heads, seq_len, head_dim).astype(np.float32)

    # Create tensors
    q = mn.Tensor.from_numpy(q_data)
    k = mn.Tensor.from_numpy(k_data)
    v = mn.Tensor.from_numpy(v_data)

    scale = 1.0 / np.sqrt(head_dim)

    # Run with standard path
    mn.set_texture_attention(False)
    assert not mn.texture_attention_enabled(), "Texture attention should be disabled"
    out_standard = mn.flash_attention(q, k, v, None, scale)
    out_standard_np = out_standard.numpy()

    # Run with texture path
    mn.set_texture_attention(True)
    assert mn.texture_attention_enabled(), "Texture attention should be enabled"
    out_texture = mn.flash_attention(q, k, v, None, scale)
    out_texture_np = out_texture.numpy()

    # Disable texture attention for subsequent tests
    mn.set_texture_attention(False)

    # Compare results (should be very close, allowing for minor numerical differences)
    max_diff = np.max(np.abs(out_standard_np - out_texture_np))
    mean_diff = np.mean(np.abs(out_standard_np - out_texture_np))

    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")

    # Allow for small numerical differences due to different execution order
    tolerance = 1e-4
    if max_diff < tolerance:
        print("  ✓ PASS: Results match within tolerance")
        return True
    else:
        print(f"  ✗ FAIL: Results differ by more than {tolerance}")
        print(f"  Standard output sample: {out_standard_np[0, 0, 0, :4]}")
        print(f"  Texture output sample:  {out_texture_np[0, 0, 0, :4]}")
        return False


def test_texture_attention_toggle():
    """Test enabling/disabling texture attention."""
    print("\nTesting texture attention toggle...")

    # Default should be disabled
    assert not mn.texture_attention_enabled(), "Default should be disabled"
    print("  ✓ Default state: disabled")

    # Enable
    mn.set_texture_attention(True)
    assert mn.texture_attention_enabled(), "Should be enabled"
    print("  ✓ Enable works")

    # Disable
    mn.set_texture_attention(False)
    assert not mn.texture_attention_enabled(), "Should be disabled"
    print("  ✓ Disable works")

    return True


if __name__ == '__main__':
    print("=" * 60)
    print("Texture-Backed Attention Test Suite")
    print("=" * 60)

    all_passed = True

    # Test toggle functionality
    all_passed &= test_texture_attention_toggle()

    # Test FP32 attention with texture backend
    all_passed &= test_texture_attention_fp32()

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests PASSED")
        sys.exit(0)
    else:
        print("✗ Some tests FAILED")
        sys.exit(1)
