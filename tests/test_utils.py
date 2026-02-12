"""Tests for utils.py functions."""

import pytest
import torch
import numpy as np
from utils import compute_distances


class TestComputeDistances:
    """Test suite for compute_distances function."""

    def test_basic_distance_computation(self):
        """Test basic distance computation with simple data."""
        # Create simple test data: 2 samples, 3 training points
        samples = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0]],  # Sample 1
            [[5.0, 6.0], [7.0, 8.0]]   # Sample 2
        ])
        
        training_data = torch.tensor([
            [[0.0, 0.0], [0.0, 0.0]],  # Training 1
            [[1.0, 2.0], [3.0, 4.0]],  # Training 2 (identical to sample 1)
            [[10.0, 10.0], [10.0, 10.0]]  # Training 3
        ])
        
        distances = compute_distances(samples, training_data)
        
        # Check output shape
        assert distances.shape == (2,), f"Expected shape (2,), got {distances.shape}"
        
        # Check that distance to identical point is zero
        assert distances[0].item() == pytest.approx(0.0, abs=1e-5), \
            "Distance to identical training sample should be 0"
        
        # Check that all distances are non-negative
        assert torch.all(distances >= 0), "All distances should be non-negative"

    def test_single_sample(self):
        """Test with a single sample."""
        samples = torch.randn(1, 10, 10)
        training_data = torch.randn(5, 10, 10)
        
        distances = compute_distances(samples, training_data)
        
        assert distances.shape == (1,)
        assert distances[0] >= 0

    def test_output_type(self):
        """Test that output is a torch tensor."""
        samples = torch.randn(3, 5, 5)
        training_data = torch.randn(4, 5, 5)
        
        distances = compute_distances(samples, training_data)
        
        assert isinstance(distances, torch.Tensor)
        assert distances.dtype == torch.float32

    def test_minimum_distance(self):
        """Test that function returns minimum distance to any training sample."""
        # Create a sample and training data where we know the distances
        sample = torch.zeros(1, 2, 2)  # [0, 0, 0, 0]
        
        training_data = torch.tensor([
            [[1.0, 1.0], [1.0, 1.0]],  # Distance = 2.0
            [[2.0, 2.0], [2.0, 2.0]],  # Distance = 4.0
            [[0.5, 0.5], [0.5, 0.5]],  # Distance = 1.0 (minimum)
        ])
        
        distances = compute_distances(sample, training_data)
        
        # Minimum should be approximately 1.0 (Euclidean distance)
        assert distances[0].item() == pytest.approx(1.0, abs=1e-5)

    def test_real_image_dimensions(self):
        """Test with realistic image dimensions (101x91)."""
        # Simulate actual image data
        samples = torch.randn(10, 101, 91)
        training_data = torch.randn(20, 101, 91)
        
        distances = compute_distances(samples, training_data)
        
        assert distances.shape == (10,)
        assert torch.all(distances >= 0)
        assert torch.all(torch.isfinite(distances))

    def test_identical_samples_and_training(self):
        """Test when samples are identical to training data."""
        data = torch.randn(5, 10, 10)
        samples = data.clone()
        training_data = data.clone()
        
        distances = compute_distances(samples, training_data)
        
        # All distances should be zero
        assert torch.allclose(distances, torch.zeros_like(distances), atol=1e-5)

    def test_4d_input_with_channels(self):
        """Test with 4D input (batch, channels, height, width)."""
        samples = torch.randn(3, 1, 10, 10)  # 3 samples, 1 channel
        training_data = torch.randn(5, 1, 10, 10)  # 5 training samples
        
        distances = compute_distances(samples, training_data)
        
        assert distances.shape == (3,)

    def test_batch_processing(self):
        """Test that function can handle large batches."""
        samples = torch.randn(100, 50, 50)
        training_data = torch.randn(200, 50, 50)
        
        distances = compute_distances(samples, training_data)
        
        assert distances.shape == (100,)
        assert not torch.any(torch.isnan(distances))

    def test_deterministic(self):
        """Test that function is deterministic."""
        samples = torch.randn(5, 10, 10)
        training_data = torch.randn(10, 10, 10)
        
        distances1 = compute_distances(samples, training_data)
        distances2 = compute_distances(samples, training_data)
        
        assert torch.allclose(distances1, distances2)

    def test_distance_symmetry(self):
        """Test basic distance properties."""
        samples = torch.randn(3, 10, 10)
        training_data = torch.randn(5, 10, 10)
        
        # Compute distances
        distances = compute_distances(samples, training_data)
        
        # If we swap samples and training, distances should be related
        # (not equal due to min operation, but all should be finite)
        distances_swapped = compute_distances(training_data, samples)
        
        assert torch.all(torch.isfinite(distances))
        assert torch.all(torch.isfinite(distances_swapped))


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
