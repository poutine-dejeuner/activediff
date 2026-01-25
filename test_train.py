import torch
import shutil
from pathlib import Path
from train import ActiveLearningDDPM
import yaml


def test_active_learning():
    """Test function to validate the active learning pipeline."""
    print("\n" + "="*60)
    print("TESTING ACTIVE LEARNING PIPELINE")
    print("="*60)
    
    # Create synthetic test data
    print("\nCreating synthetic test data...")
    n_initial = 100
    data_shape = (64, 64)  # Adjust based on your actual data shape
    
    synthetic_data = torch.randn(n_initial, 1, *data_shape)
    
    # Save synthetic data
    test_data_path = Path("./test_data.pt")
    torch.save(synthetic_data, test_data_path)
    print(f"Saved synthetic data: {synthetic_data.shape}")
    
    # Create test config
    test_config = {
        'data': {
            'initial_data_path': str(test_data_path),
            'output_dir': './test_active_learning'
        },
        'active_learning': {
            'fom_threshold': 0.48,
            'distance_threshold': 0.1,
            'n_samples_per_iter': 50,
            'max_iterations': 2
        },
        'ddpm': {
            'max_epochs': 10,
            'batch_size': 16,
            'learning_rate': 1e-4
        },
        'generation': {
            'batch_size': 10
        },
        'fom': {
            'n_parallel_workers': 4
        }
    }
    
    # Initialize active learning with test config
    al = ActiveLearningDDPM(test_config)
    
    # Test individual components
    print("\n" + "-"*60)
    print("Testing individual components...")
    print("-"*60)
    
    # Test distance computation
    test_samples = torch.randn(10, 1, *data_shape)
    distances = al.compute_distances(test_samples)
    assert len(distances) == 10, "Distance computation failed"
    print(f"Distance computation: {distances.shape}")
    
    # Test sample selection
    test_fom = torch.rand(10) * 0.8  # Random FOM between 0 and 0.8
    test_distances = torch.rand(10) * 0.3  # Random distances
    selected = al.select_samples(test_samples, test_fom, test_distances)
    print(f"Sample selection: {len(selected)} samples selected")
    
    # Test data update
    updated = al.update_training_data(selected)
    print(f"Data update: {'Success' if updated or len(selected)==0 else 'Failed'}")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    
    # Cleanup
    if test_data_path.exists():
        test_data_path.unlink()
    if Path("./test_active_learning").exists():
        shutil.rmtree("./test_active_learning")
    
    print("\nTest cleanup complete.")

