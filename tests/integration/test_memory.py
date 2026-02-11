"""Integration tests for memory management."""
import pytest
from .conftest import requires_metal


@pytest.mark.integration
@requires_metal
class TestMemoryTracking:
    """Test memory allocation tracking."""

    def test_memory_allocated_nonzero(self):
        """Test that memory_allocated reports non-zero after allocation."""
        import metal_native as mn

        # Reset stats
        mn.reset_peak_stats()

        # Allocate a tensor
        t = mn.zeros((1000, 1000))

        # Check memory is allocated
        allocated = mn.memory_allocated()
        assert allocated > 0, "memory_allocated should report non-zero after allocation"

        # Expected size: 1000 * 1000 * 4 bytes (float32) = 4MB
        expected_min = 4_000_000
        assert allocated >= expected_min, f"Expected at least {expected_min} bytes, got {allocated}"

    def test_empty_cache_reduces_memory(self):
        """Test that empty_cache reduces memory usage."""
        import metal_native as mn

        # Allocate and then delete tensors
        for _ in range(5):
            t = mn.randn(500, 500)
            del t

        # Memory might still be cached
        before_cache = mn.memory_allocated()

        # Clear cache
        mn.empty_cache()

        after_cache = mn.memory_allocated()

        # After cache clear, memory should be less or equal
        assert after_cache <= before_cache, "empty_cache should reduce or maintain memory usage"

    def test_multiple_large_allocations(self):
        """Test allocating multiple large tensors."""
        import metal_native as mn

        mn.reset_peak_stats()

        # Allocate several large tensors
        tensors = []
        for i in range(3):
            t = mn.ones((500, 500))
            tensors.append(t)

        # Check memory increased
        allocated = mn.memory_allocated()

        # Expected: at least 3 * 500 * 500 * 4 = 3MB
        expected_min = 3_000_000
        assert allocated >= expected_min

        # Clean up
        del tensors
        mn.empty_cache()

    def test_max_memory_allocated(self):
        """Test that max_memory_allocated tracks peak usage."""
        import metal_native as mn

        mn.reset_peak_stats()

        # Allocate a large tensor
        t = mn.zeros((2000, 2000))
        max_mem = mn.max_memory_allocated()

        # Delete it
        del t

        # Max memory should still reflect the peak
        current_mem = mn.memory_allocated()
        assert max_mem >= current_mem, "max_memory_allocated should track peak"

        # Expected peak: at least 2000 * 2000 * 4 = 16MB
        expected_min = 16_000_000
        assert max_mem >= expected_min

    def test_memory_cleanup_after_operations(self):
        """Test that temporary memory from operations is cleaned up."""
        import metal_native as mn

        mn.reset_peak_stats()
        mn.empty_cache()

        a = mn.ones((100, 100))
        b = mn.ones((100, 100))

        # Perform operations that create temporaries
        for _ in range(10):
            c = a + b
            d = c * 2.0
            e = d - a
            del c, d, e

        # Synchronize and clear cache
        mn.synchronize()
        mn.empty_cache()

        # Memory should be close to just storing a and b
        final_mem = mn.memory_allocated()

        # Expected: roughly 2 * 100 * 100 * 4 = 80KB
        # Allow some overhead for allocator bookkeeping
        expected_max = 200_000
        assert final_mem < expected_max, f"Memory usage {final_mem} exceeds expected {expected_max}"


@pytest.mark.integration
@requires_metal
class TestMemorySynchronization:
    """Test memory synchronization utilities."""

    def test_synchronize(self):
        """Test that synchronize completes without error."""
        import metal_native as mn

        # Create and operate on tensors
        a = mn.randn(100, 100)
        b = mn.randn(100, 100)
        c = a @ b

        # Synchronize should wait for all operations to complete
        mn.synchronize()

        # Should be able to access result
        result = c.numpy()
        assert result.shape == (100, 100)

    def test_synchronize_before_memory_check(self):
        """Test synchronizing before checking memory stats."""
        import metal_native as mn

        mn.reset_peak_stats()

        # Create tensors and perform operations
        tensors = []
        for i in range(5):
            t = mn.randn(200, 200)
            t = t + 1.0
            t = t * 2.0
            tensors.append(t)

        # Synchronize before checking memory
        mn.synchronize()

        allocated = mn.memory_allocated()
        assert allocated > 0

        # Clean up
        del tensors
        mn.empty_cache()
