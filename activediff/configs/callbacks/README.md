# Callbacks Configuration

This directory contains callback configurations for PyTorch Lightning training.

## Available Configurations

### default.yaml
Full production configuration with all callbacks:
- **model_checkpoint**: Saves best model checkpoints based on validation loss
- **generate_image**: Generates sample images during training
- **binarization_metric**: Computes and logs binarization metrics
- **binarization_early_stopping**: Stops training when binarization threshold is reached
- **early_stopping**: Stops training when validation loss plateaus
- **threshold_stopping**: Stops training when validation loss reaches a threshold

### minimal.yaml
Lightweight configuration for debugging/fast iterations:
- **model_checkpoint**: Basic checkpoint saving
- **early_stopping**: Quick early stopping for development

## Usage

Select a callback configuration in your main config:

```yaml
defaults:
  - callbacks: default  # or minimal
```

Or override from command line:

```bash
python -m activediff.main callbacks=minimal
```

## Customizing Callbacks

You can override individual callback parameters:

```bash
# Change early stopping patience
python -m activediff.main callbacks.early_stopping.patience=100

# Disable a specific callback
python -m activediff.main ~callbacks.binarization_early_stopping

# Change image generation frequency
python -m activediff.main callbacks.generate_image.every_n_epochs=50
```

## Creating Custom Callback Configs

Create a new YAML file in this directory with your callback definitions:

```yaml
my_callback:
  _target_: path.to.MyCallback
  param1: value1
  param2: value2
```

Each callback must have a `_target_` field pointing to the callback class.
