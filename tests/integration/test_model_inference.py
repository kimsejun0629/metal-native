"""Integration tests for simple model inference."""
import pytest
import numpy as np
from .conftest import requires_metal


@pytest.mark.integration
@requires_metal
class TestLinearModel:
    """Test simple linear model inference."""

    def test_linear_forward_pass(self):
        """Test creating a simple linear layer and running forward pass."""
        import metal_native as mn

        # Create a simple linear layer: y = Wx + b
        batch_size = 4
        in_features = 10
        out_features = 5

        # Initialize weights and bias
        W = mn.randn(in_features, out_features) * 0.1
        b = mn.zeros((out_features,))

        # Create input
        x = mn.randn(batch_size, in_features)

        # Forward pass: y = xW + b
        y = x @ W

        # Add bias (broadcasting)
        for i in range(batch_size):
            y_i = y.numpy()[i] + b.numpy()
            # Just verify we can perform the operation
            assert y_i.shape == (out_features,)

        # Check output shape
        assert y.shape == (batch_size, out_features)

        # Verify results are reasonable (not NaN or Inf)
        y_np = y.numpy()
        assert not np.isnan(y_np).any(), "Output contains NaN"
        assert not np.isinf(y_np).any(), "Output contains Inf"

    def test_two_layer_network(self):
        """Test a simple two-layer neural network."""
        import metal_native as mn

        batch_size = 8
        input_dim = 20
        hidden_dim = 15
        output_dim = 10

        # Layer 1: input -> hidden
        W1 = mn.randn(input_dim, hidden_dim) * 0.1
        b1 = mn.zeros((hidden_dim,))

        # Layer 2: hidden -> output
        W2 = mn.randn(hidden_dim, output_dim) * 0.1
        b2 = mn.zeros((output_dim,))

        # Input
        x = mn.randn(batch_size, input_dim)

        # Forward pass
        # Layer 1
        h = x @ W1

        # ReLU activation (manual implementation)
        h_np = h.numpy()
        h_np = np.maximum(h_np, 0)
        h = mn.from_numpy(h_np)

        # Layer 2
        y = h @ W2

        # Check output
        assert y.shape == (batch_size, output_dim)
        y_np = y.numpy()
        assert not np.isnan(y_np).any()
        assert not np.isinf(y_np).any()

    def test_batch_matrix_multiplication(self):
        """Test batch matrix multiplication for model inference."""
        import metal_native as mn

        # Simulate multiple batch items
        batch_size = 16
        seq_len = 10
        hidden_dim = 64

        # Create input: [batch, seq_len, hidden_dim]
        # We'll test this as [batch*seq_len, hidden_dim]
        x = mn.randn(batch_size * seq_len, hidden_dim)

        # Weight matrix
        W = mn.randn(hidden_dim, hidden_dim) * 0.02

        # Forward pass
        y = x @ W

        assert y.shape == (batch_size * seq_len, hidden_dim)

        y_np = y.numpy()
        assert not np.isnan(y_np).any()
        assert not np.isinf(y_np).any()


@pytest.mark.integration
@requires_metal
class TestAttentionMechanism:
    """Test simple attention mechanism components."""

    def test_scaled_dot_product_attention(self):
        """Test scaled dot-product attention calculation."""
        import metal_native as mn

        batch_size = 2
        seq_len = 8
        d_model = 32

        # Query, Key, Value matrices
        # For simplicity, use [seq_len, d_model]
        Q = mn.randn(seq_len, d_model) * 0.1
        K = mn.randn(seq_len, d_model) * 0.1
        V = mn.randn(seq_len, d_model) * 0.1

        # Attention scores: QK^T / sqrt(d_model)
        # K^T would be [d_model, seq_len]
        # For testing, we'll compute Q @ K^T manually
        scores = Q @ K.numpy().T  # Note: using numpy for transpose

        # In a real implementation, we'd apply softmax here
        # For now, just verify the shape
        assert scores.shape == (seq_len, seq_len)

        scores_np = scores.numpy()
        assert not np.isnan(scores_np).any()
        assert not np.isinf(scores_np).any()

    def test_multi_head_projection(self):
        """Test multi-head attention projections."""
        import metal_native as mn

        seq_len = 10
        d_model = 64
        num_heads = 8
        d_k = d_model // num_heads  # 8

        # Input
        x = mn.randn(seq_len, d_model) * 0.1

        # Projection matrices for Q, K, V
        W_q = mn.randn(d_model, d_model) * 0.02
        W_k = mn.randn(d_model, d_model) * 0.02
        W_v = mn.randn(d_model, d_model) * 0.02

        # Project
        Q = x @ W_q
        K = x @ W_k
        V = x @ W_v

        # Verify shapes
        assert Q.shape == (seq_len, d_model)
        assert K.shape == (seq_len, d_model)
        assert V.shape == (seq_len, d_model)

        # Verify no numerical issues
        for tensor in [Q, K, V]:
            t_np = tensor.numpy()
            assert not np.isnan(t_np).any()
            assert not np.isinf(t_np).any()


@pytest.mark.integration
@requires_metal
class TestModelNumericalStability:
    """Test numerical stability of model operations."""

    def test_large_matrix_multiplication(self):
        """Test that large matrix multiplications remain stable."""
        import metal_native as mn

        # Large matrices
        m, n, k = 512, 512, 512

        A = mn.randn(m, k) * 0.01  # Small values to avoid overflow
        B = mn.randn(k, n) * 0.01

        C = A @ B

        assert C.shape == (m, n)

        C_np = C.numpy()
        assert not np.isnan(C_np).any(), "Large matmul produced NaN"
        assert not np.isinf(C_np).any(), "Large matmul produced Inf"

        # Check values are in reasonable range
        assert np.abs(C_np).max() < 100.0, "Values unexpectedly large"

    def test_fp16_precision(self):
        """Test FP16 operations maintain acceptable precision."""
        import metal_native as mn

        # Create FP16 tensors
        a = mn.ones((10, 10), dtype=mn.float16)
        b = mn.ones((10, 10), dtype=mn.float16) * 2.0

        c = a + b

        result = c.numpy()
        # FP16 has lower precision, use larger tolerance
        assert np.allclose(result, 3.0, atol=1e-2), "FP16 addition failed"

    def test_mixed_operations_stability(self):
        """Test stability of mixed arithmetic operations."""
        import metal_native as mn

        # Perform a series of operations
        x = mn.randn(100, 100)

        # y = (x + 1) * 2 - 0.5
        y = (x + 1.0) * 2.0 - 0.5

        # z = y @ y^T (simplified)
        z = y @ y.numpy().T

        z_np = z.numpy()
        assert not np.isnan(z_np).any()
        assert not np.isinf(z_np).any()

    def test_chained_operations(self):
        """Test numerical stability of chained operations."""
        import metal_native as mn

        x = mn.ones((50, 50)) * 0.5

        # Chain of operations
        result = x
        for _ in range(5):
            result = result + 0.1
            result = result * 0.9

        result_np = result.numpy()
        assert not np.isnan(result_np).any()
        assert not np.isinf(result_np).any()

        # Values should be positive and reasonable
        assert (result_np > 0).all()
        assert (result_np < 10.0).all()
