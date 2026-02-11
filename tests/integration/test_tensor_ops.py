"""Integration tests for tensor operations."""
import pytest
import numpy as np
from .conftest import requires_metal


@pytest.mark.integration
@requires_metal
class TestTensorCreation:
    """Test tensor creation functions."""

    def test_zeros(self):
        """Test creating a tensor filled with zeros."""
        import metal_native as mn

        t = mn.zeros((3, 3))
        assert t.shape == (3, 3)
        assert t.dtype == mn.float32

        np_data = t.numpy()
        assert np.allclose(np_data, 0.0, atol=1e-5)

    def test_ones(self):
        """Test creating a tensor filled with ones."""
        import metal_native as mn

        t = mn.ones((2, 4))
        assert t.shape == (2, 4)
        assert t.dtype == mn.float32

        np_data = t.numpy()
        assert np.allclose(np_data, 1.0, atol=1e-5)

    def test_randn(self):
        """Test creating a tensor with random normal values."""
        import metal_native as mn

        t = mn.randn(100, 100)
        assert t.shape == (100, 100)
        assert t.dtype == mn.float32

        np_data = t.numpy()
        # Check that mean is close to 0 and std is close to 1
        assert abs(np_data.mean()) < 0.2
        assert abs(np_data.std() - 1.0) < 0.2

    def test_tensor_from_list(self):
        """Test creating a tensor from a Python list."""
        import metal_native as mn

        data = [[1.0, 2.0], [3.0, 4.0]]
        t = mn.tensor(data)

        assert t.shape == (2, 2)
        assert t.dtype == mn.float32

        np_data = t.numpy()
        expected = np.array(data, dtype=np.float32)
        assert np.allclose(np_data, expected, atol=1e-5)

    def test_tensor_from_numpy(self):
        """Test creating a tensor from a NumPy array."""
        import metal_native as mn

        np_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        t = mn.from_numpy(np_data)

        assert t.shape == (2, 3)
        assert t.dtype == mn.float32

        result = t.numpy()
        assert np.allclose(result, np_data, atol=1e-5)


@pytest.mark.integration
@requires_metal
class TestArithmeticOperations:
    """Test arithmetic operations on tensors."""

    def test_add_tensors(self):
        """Test element-wise addition of two tensors."""
        import metal_native as mn

        a = mn.ones((3, 3))
        b = mn.ones((3, 3)) * 2.0
        c = a + b

        assert c.shape == (3, 3)
        result = c.numpy()
        assert np.allclose(result, 3.0, atol=1e-5)

    def test_add_scalar(self):
        """Test adding a scalar to a tensor."""
        import metal_native as mn

        a = mn.ones((2, 2))
        b = a + 5.0

        result = b.numpy()
        assert np.allclose(result, 6.0, atol=1e-5)

    def test_sub_tensors(self):
        """Test element-wise subtraction of two tensors."""
        import metal_native as mn

        a = mn.ones((3, 3)) * 5.0
        b = mn.ones((3, 3)) * 2.0
        c = a - b

        result = c.numpy()
        assert np.allclose(result, 3.0, atol=1e-5)

    def test_sub_scalar(self):
        """Test subtracting a scalar from a tensor."""
        import metal_native as mn

        a = mn.ones((2, 2)) * 10.0
        b = a - 3.0

        result = b.numpy()
        assert np.allclose(result, 7.0, atol=1e-5)

    def test_mul_tensors(self):
        """Test element-wise multiplication of two tensors."""
        import metal_native as mn

        a = mn.ones((3, 3)) * 3.0
        b = mn.ones((3, 3)) * 4.0
        c = a * b

        result = c.numpy()
        assert np.allclose(result, 12.0, atol=1e-5)

    def test_mul_scalar(self):
        """Test multiplying a tensor by a scalar."""
        import metal_native as mn

        a = mn.ones((2, 2)) * 2.0
        b = a * 3.0

        result = b.numpy()
        assert np.allclose(result, 6.0, atol=1e-5)

    def test_div_tensors(self):
        """Test element-wise division of two tensors."""
        import metal_native as mn

        a = mn.ones((3, 3)) * 12.0
        b = mn.ones((3, 3)) * 4.0
        c = a / b

        result = c.numpy()
        assert np.allclose(result, 3.0, atol=1e-5)

    def test_div_scalar(self):
        """Test dividing a tensor by a scalar."""
        import metal_native as mn

        a = mn.ones((2, 2)) * 10.0
        b = a / 2.0

        result = b.numpy()
        assert np.allclose(result, 5.0, atol=1e-5)

    def test_matmul(self):
        """Test matrix multiplication."""
        import metal_native as mn

        # Create simple matrices for testing
        a = mn.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = mn.tensor([[2.0, 0.0], [1.0, 2.0]])
        c = a @ b

        result = c.numpy()
        expected = np.array([[4.0, 4.0], [10.0, 8.0]], dtype=np.float32)
        assert np.allclose(result, expected, atol=1e-5)


@pytest.mark.integration
@requires_metal
class TestNumpyInterop:
    """Test NumPy interoperability."""

    def test_numpy_roundtrip(self):
        """Test converting to NumPy and back."""
        import metal_native as mn

        # Create original NumPy array
        np_original = np.random.randn(5, 5).astype(np.float32)

        # Convert to metal_native tensor
        t = mn.from_numpy(np_original)

        # Convert back to NumPy
        np_result = t.numpy()

        assert np.allclose(np_original, np_result, atol=1e-5)

    def test_numpy_operations_compatibility(self):
        """Test that operations produce results consistent with NumPy."""
        import metal_native as mn

        # Create test data
        np_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np_b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

        # metal_native operations
        mn_a = mn.from_numpy(np_a)
        mn_b = mn.from_numpy(np_b)
        mn_result = (mn_a + mn_b).numpy()

        # NumPy operations
        np_result = np_a + np_b

        assert np.allclose(mn_result, np_result, atol=1e-5)


@pytest.mark.integration
@requires_metal
class TestDtypeConversions:
    """Test data type conversions."""

    def test_float32_creation(self):
        """Test creating a float32 tensor."""
        import metal_native as mn

        t = mn.zeros((3, 3), dtype=mn.float32)
        assert t.dtype == mn.float32
        assert t.numpy().dtype == np.float32

    def test_float16_creation(self):
        """Test creating a float16 tensor."""
        import metal_native as mn

        t = mn.ones((2, 2), dtype=mn.float16)
        assert t.dtype == mn.float16

        # Note: tolerance is higher for FP16
        result = t.numpy()
        assert np.allclose(result, 1.0, atol=1e-2)

    def test_int32_creation(self):
        """Test creating an int32 tensor."""
        import metal_native as mn

        t = mn.tensor([1, 2, 3, 4], dtype=mn.int32)
        assert t.dtype == mn.int32

        result = t.numpy()
        expected = np.array([1, 2, 3, 4], dtype=np.int32)
        assert np.array_equal(result, expected)

    def test_dtype_inference(self):
        """Test automatic dtype inference from input data."""
        import metal_native as mn

        # Float data should infer float32
        t_float = mn.tensor([1.0, 2.0, 3.0])
        assert t_float.dtype == mn.float32

        # Integer data should infer float32 (default behavior)
        t_int = mn.tensor([1, 2, 3])
        assert t_int.dtype == mn.float32
