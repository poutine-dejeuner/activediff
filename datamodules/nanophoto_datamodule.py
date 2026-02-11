"""Datamodule for nanophotonic design data."""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, random_split
from omegaconf import DictConfig


class NanophotoDataModule(pl.LightningDataModule):
    """Datamodule for loading and managing nanophotonic design data.

    This datamodule handles:
    - Loading initial training data from disk
    - Managing new samples added during active learning
    - Combining initial and new data for training
    - Saving/loading checkpoints

    Args:
        initial_data_path: Path to initial training data (.npy file)
        output_dir: Directory for saving outputs and checkpoints
        output_file: Filename for saving new samples
        batch_size: Batch size for dataloaders
        val_split: Fraction of data to use for validation
        num_workers: Number of workers for dataloaders
    """

    def __init__(
        self,
        initial_data_path: Union[str, Path],
        output_dir: Union[str, Path],
        output_file: str = "new_images.npy",
        batch_size: int = 32,
        val_split: float = 0.1,
        num_workers: int = 4
    ):
        super().__init__()
        super().__init__()
        self.initial_data_path = Path(initial_data_path)
        self.output_dir = Path(output_dir)
        self.output_file = output_file
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for data
        self._initial_training_data: Optional[torch.Tensor] = None
        self._new_samples: list[torch.Tensor] = []  # List of tensors, one per iteration
        self._train_dataset: Optional[torch.utils.data.Dataset] = None
        self._val_dataset: Optional[torch.utils.data.Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Load initial training data from disk and restore previous state if available."""
        if self._initial_training_data is not None:
            return  # Already loaded

        print(f"Loading data from: {self.initial_data_path}")
        data = np.load(self.initial_data_path)
        self._initial_training_data = torch.from_numpy(data)

        # Handle dict format if needed
        if isinstance(self._initial_training_data, dict):
            self._initial_training_data = self._initial_training_data['data']

        print(f"Loaded initial training data: {self._initial_training_data.shape}")

        # Try to load checkpoint
        last_iteration = self.load_checkpoint()
        if last_iteration is not None:
            print(f"Resuming from iteration {last_iteration}")

        # Try to load and merge saved samples
        self.load_and_merge_saved_samples()

        # Prepare datasets for training
        self.prepare_data_splits()

    def prepare_data_splits(self) -> None:
        """Prepare train/val splits from current training data."""
        data = self.training_data.numpy()
        if data.ndim == 3:
            data = data[:, None, :, :]  # Add channel dimension
        data = torch.tensor(data, dtype=torch.float32)
        dataset = TensorDataset(data)

        # Split into train/val
        train_size = int(len(data) * (1 - self.val_split))
        val_size = len(data) - train_size

        train_set, val_set = random_split(dataset, [train_size, val_size])

        self._train_dataset = train_set
        self._val_dataset = val_set

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    @property
    def initial_data(self) -> torch.Tensor:
        """Get initial training data."""
        if self._initial_training_data is None:
            self.setup()
        return self._initial_training_data

    @property
    def new_samples(self) -> list[torch.Tensor]:
        """Get list of new samples added during active learning."""
        return self._new_samples

    @property
    def training_data(self) -> torch.Tensor:
        """Get combined training data (initial + new samples)."""
        if len(self._new_samples) == 0:
            return self.initial_data
        return torch.cat([self.initial_data] + self._new_samples, dim=0)

    def add_samples(self, samples: torch.Tensor) -> None:
        """Add new samples to the training dataset.

        Args:
            samples: Tensor of new samples to add
        """
        if len(samples) > 0:
            self._new_samples.append(samples)
            print(f"Added {len(samples)} new samples. Total dataset size: {len(self.training_data)}")
            # Re-prepare data splits with new samples
            self.prepare_data_splits()

    def save_new_samples(self, filepath: Optional[Path] = None) -> None:
        """Save all new samples to disk.

        Args:
            filepath: Optional custom filepath. If None, uses default output path.
        """
        if len(self._new_samples) == 0:
            print("No new samples to save.")
            return

        all_new_samples = torch.cat(self._new_samples, dim=0)
        save_path = filepath or (self.output_dir / self.output_file)

        np.save(save_path, all_new_samples.numpy())
        print(f"Saved {len(all_new_samples)} new samples to: {save_path}")

    def load_checkpoint(self) -> Optional[int]:
        """Load the latest checkpoint if it exists.

        Returns:
            Last completed iteration number, or None if no checkpoint exists
        """
        # Find the latest checkpoint
        checkpoint_files = sorted(self.output_dir.glob("checkpoint_iter_*.pt"))

        if not checkpoint_files:
            return None

        latest_checkpoint = checkpoint_files[-1]
        print(f"Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)

        # Load new_samples from individual iteration files
        iteration = checkpoint.get('iteration', -1)
        self._new_samples = []

        for i in range(iteration + 1):
            selected_path = self.output_dir / f"selected_samples_iter_{i}.pt"
            if selected_path.exists():
                selected = torch.load(selected_path)
                self._new_samples.append(selected)

        if self._new_samples:
            print(f"Restored {len(self._new_samples)} iterations with "
                  f"{checkpoint['n_new_samples']} total new samples")

        return iteration

    def load_and_merge_saved_samples(self) -> None:
        """Load saved new_images.npy and merge into initial data."""
        new_images_path = self.output_dir / self.output_file

        if not new_images_path.exists():
            return

        print(f"Found saved samples: {new_images_path}")
        new_images = np.load(new_images_path)
        new_images_tensor = torch.from_numpy(new_images)

        print(f"Initial data shape: {self._initial_training_data.shape}")
        print(f"New images shape: {new_images_tensor.shape}")

        # Merge into initial data
        self._initial_training_data = torch.cat(
            [self._initial_training_data, new_images_tensor], dim=0
        )

        # Clear new_samples since they're now part of initial data
        self._new_samples = []

        print(f"Merged new images into initial data. New shape: {self._initial_training_data.shape}")

    def save_checkpoint(self, iteration: int) -> None:
        """Save checkpoint with current state.

        Args:
            iteration: Current iteration number
        """
        if len(self._new_samples) == 0:
            return

        all_new_samples = torch.cat(self._new_samples, dim=0)
        checkpoint = {
            'iteration': iteration,
            'new_samples': all_new_samples,
            'n_new_samples': len(all_new_samples),
            'n_total_samples': len(self.training_data)
        }

        checkpoint_path = self.output_dir / f"checkpoint_iter_{iteration}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to: {checkpoint_path}")

    def __repr__(self) -> str:
        """String representation."""
        n_initial = len(self._initial_training_data) if self._initial_training_data is not None else 0
        n_new = sum(len(s) for s in self._new_samples)
        return (f"NanophotoDataModule(\n"
                f"  initial_data_path={self.initial_data_path},\n"
                f"  n_initial_samples={n_initial},\n"
                f"  n_new_samples={n_new},\n"
                f"  total_samples={n_initial + n_new}\n"
                f")")
