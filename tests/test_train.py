import numpy as np
import torch
import shutil
from pathlib import Path
import tempfile
import yaml
import pytest
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_class, instantiate


def test_checkpoint_loading_with_get_class():
    """Test that loading checkpoint with get_class works correctly."""
    print("\n" + "="*60)
    print("TESTING CHECKPOINT LOADING WITH get_class")
    print("="*60)

    # Create a minimal config
    cfg = OmegaConf.create({
        'model': {
            '_target_': 'models.unet.UNet',
            'time_steps': 100,
            'image_shape': [64, 64],
            'lr': 0.0001,
            'ema_decay': 0.9999,
            'num_layers': 6,
            'first_channels': 32,
            'input_channels': 1,
            'output_channels': 1,
            'num_groups': 8,
            'dropout_prob': 0.1,
            'num_heads': 4,
            'Attentions': [False, False, True, False, True, False],
            'Upscales': [False, False, False, True, True, True],
        },
        'dtype': 'float32'
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        checkpoint_path = tmpdir / "test_checkpoint.ckpt"

        # Create and save a model
        print("\n1. Creating initial model...")
        model = instantiate(cfg.model)
        trainer = __import__('pytorch_lightning').Trainer(
            max_epochs=1,
            fast_dev_run=True,
            enable_progress_bar=False,
            logger=False,
            accelerator='auto'  # Use GPU if available
        )

        # Create dummy data
        dummy_data = torch.randn(10, 1, 64, 64)
        dummy_dataset = torch.utils.data.TensorDataset(dummy_data)
        dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=2)

        # Train briefly
        print("2. Training briefly...")
        trainer.fit(model, dummy_loader, dummy_loader)

        # Save checkpoint
        print(f"3. Saving checkpoint to {checkpoint_path}...")
        trainer.save_checkpoint(checkpoint_path)

        assert checkpoint_path.exists(), "Checkpoint was not saved"
        print(f"   Checkpoint saved: {checkpoint_path.stat().st_size} bytes")

        # Load checkpoint using get_class (like in train.py)
        print("\n4. Loading checkpoint with get_class...")
        model_class = get_class(cfg.model._target_)
        loaded_model = model_class.load_from_checkpoint(checkpoint_path)

        assert loaded_model is not None, "Model failed to load"
        assert isinstance(loaded_model, model_class), "Loaded model has wrong type"
        print(f"   ✓ Model loaded successfully: {type(loaded_model).__name__}")

        # Verify parameters match
        print("\n5. Verifying model parameters...")
        assert loaded_model.time_steps == 100, "time_steps mismatch"
        assert loaded_model.lr == 0.0001, "lr mismatch"
        assert loaded_model.num_layers == 6, "num_layers mismatch"
        print("   ✓ All parameters match")

    print("\n" + "="*60)
    print("CHECKPOINT LOADING TEST PASSED!")
    print("="*60)


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

def test_meep_compute_fom():
    from nanophoto.meep_compute_fom import compute_FOM_parallele
    imagespath = Path("/home/vincent/repos/photo/activediff/active_learning_output/iter_0/images/images.npy")
    images = np.load(imagespath)[:4]
    fom = compute_FOM_parallele(images[0])
    print(f"FOM single: {fom}")
    fom = compute_FOM_parallele(images)
    print(f"FOM batch: {fom}")

