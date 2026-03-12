import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from activediff.utils import binarisation
from activediff.models.unet_utils import UNetPad


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
        unet_depth: Depth of UNet for padding calculation
    """

    def __init__(self, save_dir: Path, every_n_epochs: int = 10, unet_depth: int = 3):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.every_n_epochs = every_n_epochs
        self.unet_depth = unet_depth
        self.pad_fn = None

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
            # Get image shape and compute padded shape
            image_shape = tuple(getattr(pl_module, 'image_shape'))
            
            # Initialize padding function if needed
            if self.pad_fn is None:
                self.pad_fn = UNetPad(depth=self.unet_depth, shape=image_shape)
            
            # Compute padded shape
            padded_h = image_shape[0] + self.pad_fn.pad[3]
            padded_w = image_shape[1] + self.pad_fn.pad[1]
            
            # Start from random noise at padded size
            z = torch.randn(1, 1, padded_h, padded_w, device=pl_module.device)

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

            # Remove padding
            x = self.pad_fn.inverse(x)

        pl_module.train()

        # Convert to image
        img = x[0, 0].cpu().numpy()
        img = np.clip(img, 0, 1)  # Clip to valid range
        img = (img * 255).astype(np.uint8)

        # Save as PNG
        save_path = self.save_dir / f"generated_epoch_{trainer.current_epoch:04d}.png"
        Image.fromarray(img).save(save_path)
        print(f"Generated sample saved to: {save_path}")


class BinarizationMetricCallback(pl.Callback):
    """Compute and log binarization metric for generated samples.
    
    Args:
        save_dir: Directory to save metrics
        every_n_epochs: Compute metric every N epochs (default: 10)
        unet_depth: Depth of UNet for padding calculation
    """

    def __init__(self, save_dir: Path = None, every_n_epochs: int = 10, unet_depth: int = 3):
        super().__init__()
        self.save_dir = Path(save_dir) if save_dir else None
        self.every_n_epochs = every_n_epochs
        self.unet_depth = unet_depth
        self.metrics_history = []
        self.initialized = False
        self.pad_fn = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Initialize val/bin with a high value so EarlyStopping can monitor it."""
        if not self.initialized:
            pl_module.log("val/bin", 1.0, prog_bar=True, on_step=False, on_epoch=True)
            self.initialized = True

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Compute binarization metric after training epoch."""
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        # Verify required attributes exist
        if not hasattr(pl_module, 'image_shape') or not hasattr(pl_module, 'time_steps'):
            return
        if not hasattr(pl_module, 'scheduler_ddpm'):
            return

        pl_module.eval()

        with torch.no_grad():
            # Get image shape and compute padded shape
            image_shape = tuple(getattr(pl_module, 'image_shape'))
            
            # Initialize padding function if needed
            if self.pad_fn is None:
                self.pad_fn = UNetPad(depth=self.unet_depth, shape=image_shape)
            
            # Compute padded shape
            padded_h = image_shape[0] + self.pad_fn.pad[3]
            padded_w = image_shape[1] + self.pad_fn.pad[1]
            
            # Generate multiple samples to compute average binarization
            num_samples = 4
            samples = []
            
            for _ in range(num_samples):
                # Start from random noise at padded size
                z = torch.randn(1, 1, padded_h, padded_w, device=pl_module.device)

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

                # Remove padding
                x = self.pad_fn.inverse(x)

                samples.append(x)

            # Concatenate samples and compute binarization
            samples = torch.cat(samples, dim=0)
            # Squeeze channel dimension if present
            if samples.shape[1] == 1:
                samples = samples.squeeze(1)
            
            bin_scores = binarisation(samples)
            bin_metric = bin_scores.mean().item()

        pl_module.train()

        # Log metric using pl_module.log so other callbacks can access it
        pl_module.log("val/bin", bin_metric, prog_bar=True, on_step=False, on_epoch=True)
        self.metrics_history.append((trainer.current_epoch, bin_metric))
        
        print(f"Binarization metric at epoch {trainer.current_epoch}: {bin_metric:.4f}")


def get_training_callbacks(cfg, checkpoint_dir):
    """Create and return list of training callbacks from config.

    Args:
        cfg: Configuration object with callbacks config
        checkpoint_dir: Directory to save checkpoints

    Returns:
        List of PyTorch Lightning callbacks
    """
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    
    callbacks = []
    
    # Get callbacks config
    callbacks_cfg = cfg.callbacks
    
    # Set dynamic paths for callbacks that need them
    image_dir = checkpoint_dir / "samples"
    
    # Instantiate each callback from config
    for callback_name, callback_cfg in callbacks_cfg.items():
        # Skip non-dict entries (like old-style config keys)
        if not OmegaConf.is_dict(callback_cfg):
            continue
            
        # Skip if no _target_ (not a callback definition)
        if '_target_' not in callback_cfg:
            continue
        
        # Add dynamic parameters based on callback type
        if callback_name == 'model_checkpoint':
            callback = instantiate(callback_cfg, dirpath=checkpoint_dir)
        elif callback_name == 'generate_image':
            callback = instantiate(callback_cfg, save_dir=image_dir)
        elif callback_name == 'binarization_metric':
            callback = instantiate(callback_cfg, save_dir=checkpoint_dir)
        else:
            callback = instantiate(callback_cfg)
        
        callbacks.append(callback)
    
    return callbacks
