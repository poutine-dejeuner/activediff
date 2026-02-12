import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import numpy as np
from PIL import Image
from pathlib import Path


class ThresholdStopping(pl.Callback):
    """Stop training when a monitored metric reaches a threshold value.

    Args:
        monitor: Metric to monitor (e.g., 'val_loss')
        threshold: Threshold value to compare against
        mode: 'min' or 'max' - whether we want the metric below or above threshold
        verbose: Whether to print messages
    """

    def __init__(self, monitor: str, threshold: float, mode: str = 'min', verbose: bool = True):
        super().__init__()
        self.monitor = monitor
        self.threshold = threshold
        self.mode = mode
        self.verbose = verbose

        if mode not in ['min', 'max']:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Check if threshold is reached after validation."""
        logs = trainer.callback_metrics
        current = logs.get(self.monitor)

        if current is None:
            return

        current_value = current.item() if hasattr(current, 'item') else current

        if self.mode == 'min' and current_value <= self.threshold:
            if self.verbose:
                print(f"\nThreshold reached! {self.monitor}={current_value:.6f} <= {self.threshold}")
            trainer.should_stop = True
        elif self.mode == 'max' and current_value >= self.threshold:
            if self.verbose:
                print(f"\nThreshold reached! {self.monitor}={current_value:.6f} >= {self.threshold}")
            trainer.should_stop = True


class GenerateImageCallback(pl.Callback):
    """Generate and save a single sample image using full reverse diffusion.

    Args:
        save_dir: Directory to save images
        every_n_epochs: Generate image every N epochs (default: 10)
    """

    def __init__(self, save_dir: Path, every_n_epochs: int = 10):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Generate and save image using full reverse diffusion."""
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        # Verify required attributes exist
        if not hasattr(pl_module, 'image_shape') or not hasattr(pl_module, 'time_steps'):
            print("Warning: pl_module missing required attributes for image generation")
            return
        if not hasattr(pl_module, 'scheduler_ddpm'):
            print("Warning: pl_module missing scheduler_ddpm")
            return

        pl_module.eval()

        with torch.no_grad():
            # Start from random noise
            image_shape = getattr(pl_module, 'image_shape')
            z = torch.randn(1, 1, *image_shape, device=pl_module.device)

            # Apply padding if model has it
            pad_fn = getattr(pl_module, 'pad_fn', None)
            if pad_fn is not None:
                z = pad_fn(z)

            # Use EMA model if available
            ema = getattr(pl_module, 'ema', None)
            model = ema.module if ema is not None else pl_module
            scheduler = pl_module.scheduler_ddpm
            time_steps = getattr(pl_module, 'time_steps')

            # Reverse diffusion process
            for t in reversed(range(1, time_steps)):
                t_list = [t]
                beta_t = scheduler.beta[t]
                alpha_t = scheduler.alpha[t]
                
                temp = (beta_t / (torch.sqrt(1 - alpha_t) * torch.sqrt(1 - beta_t)))
                z = (1 / torch.sqrt(1 - beta_t)) * z - (temp * model(z, t_list))

                # Add noise
                e = torch.randn_like(z)
                z = z + (e * torch.sqrt(beta_t))

            # Final denoising step (t=0)
            beta_0 = scheduler.beta[0]
            alpha_0 = scheduler.alpha[0]
            temp = (beta_0 / (torch.sqrt(1 - alpha_0) * torch.sqrt(1 - beta_0)))
            x = (1 / torch.sqrt(1 - beta_0)) * z - (temp * model(z, [0]))

            # Remove padding if applied
            if pad_fn is not None and hasattr(pad_fn, 'unpad'):
                x = pad_fn.unpad(x)

        pl_module.train()

        # Convert to image
        img = x[0, 0].cpu().numpy()
        img = np.clip(img, 0, 1)  # Clip to valid range
        img = (img * 255).astype(np.uint8)

        # Save as PNG
        save_path = self.save_dir / f"generated_epoch_{trainer.current_epoch:04d}.png"
        Image.fromarray(img).save(save_path)
        print(f"Generated sample saved to: {save_path}")




def get_training_callbacks(cfg, checkpoint_dir):
    """Create and return list of training callbacks.

    Args:
        cfg: Configuration object
        checkpoint_dir: Directory to save checkpoints

    Returns:
        List of PyTorch Lightning callbacks
    """
    callbacks = []

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='checkpoint',
        save_top_k=1,
        monitor='val/loss',
        mode='min'
    )
    callbacks.append(checkpoint_callback)

    # Image generation callback
    image_dir = checkpoint_dir / "samples"
    generate_callback = GenerateImageCallback(
        save_dir=image_dir,
        every_n_epochs=cfg.callbacks.get('generate_image_every_n_epochs', 10)
    )
    callbacks.append(generate_callback)

    # Early stopping callback
    if cfg.callbacks.early_stopping.get('enabled', True):
        early_stop_callback = EarlyStopping(
            monitor=cfg.callbacks.early_stopping.monitor,
            patience=cfg.callbacks.early_stopping.patience,
            mode=cfg.callbacks.early_stopping.mode,
            min_delta=cfg.callbacks.early_stopping.min_delta,
            verbose=True
        )
        callbacks.append(early_stop_callback)

    # Threshold stopping callback
    if cfg.callbacks.threshold_stopping.get('enabled', False):
        threshold_callback = ThresholdStopping(
            monitor=cfg.callbacks.threshold_stopping.monitor,
            threshold=cfg.callbacks.threshold_stopping.threshold,
            mode=cfg.callbacks.threshold_stopping.mode,
            verbose=True
        )
        callbacks.append(threshold_callback)

    return callbacks
