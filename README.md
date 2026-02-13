# ActiveDiff

Active learning pipeline for nanophotonic design using diffusion models (DDPM/UNet).

## Features

- **PyTorch Lightning** integration for training
- **Active Learning** pipeline with distance-based and FOM-based sample selection
- **Diffusion Models** (DDPM) for generating nanophotonic designs
- **Weights & Biases** logging and experiment tracking
- **Hydra** configuration management
- Custom callbacks for early stopping and image generation

## Installation

```bash
# Install package
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with all dependencies
pip install -e ".[all]"
```

## Usage

### Training

```bash
# Run active learning pipeline
python train.py

# With custom config
python train.py trainer.max_epochs=50 active_learning.n_samples_per_iter=500

# Debug mode
python train.py debug=true
```

### Evaluation

```bash
python eval.py
```

## Configuration

All configuration is managed via Hydra. See `configs/` directory:

- `config.yaml` - Main configuration
- `model/unet.yaml` - UNet model configuration
- `datamodule/nanophoto.yaml` - Data module configuration
- `train/unet.yaml` - Training parameters

## Project Structure

```
activediff/
├── models/          # UNet and model utilities
├── datamodules/     # Data loading and management
├── algos/           # Active learning algorithms
├── configs/         # Hydra configuration files
├── tests/           # Unit tests
├── callbacks.py     # Custom Lightning callbacks
├── train.py         # Main training script
├── eval.py          # Evaluation script
└── utils.py         # Utility functions
```

## Testing

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_utils.py -v

# Run with coverage
pytest --cov=. tests/
```

## License

MIT
