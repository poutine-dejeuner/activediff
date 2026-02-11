import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


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
        monitor='val_loss',
        mode='min'
    )
    callbacks.append(checkpoint_callback)

    # Early stopping callback
    if cfg.training.early_stopping.get('enabled', True):
        early_stop_callback = EarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            patience=cfg.training.early_stopping.patience,
            mode=cfg.training.early_stopping.mode,
            min_delta=cfg.training.early_stopping.min_delta,
            verbose=True
        )
        callbacks.append(early_stop_callback)

    # Threshold stopping callback
    if cfg.training.threshold_stopping.get('enabled', False):
        threshold_callback = ThresholdStopping(
            monitor=cfg.training.threshold_stopping.monitor,
            threshold=cfg.training.threshold_stopping.threshold,
            mode=cfg.training.threshold_stopping.mode,
            verbose=True
        )
        callbacks.append(threshold_callback)

    return callbacks
