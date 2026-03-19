from typing import Any
from collections import defaultdict
from pathlib import Path

import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner

import hydra
from hydra.utils import instantiate, get_class
from omegaconf import OmegaConf, DictConfig, open_dict
import wandb

from activediff.datamodules import NanophotoDataModule
from activediff.utils import compute_fom_scores, compute_distances, dist_select, fom_select, filter_similar_samples
from activediff.callbacks import get_training_callbacks, binarisation


def train_and_generate_samples(datamodule, logger, cfg, iteration):
    """Train DDPM model and generate samples using PyTorch Lightning.
    
    Args:
        datamodule: Data module with training data
        logger: WandbLogger instance (will prefix metrics with iter_{iteration}/train/)
        cfg: Configuration
        iteration: Active learning iteration number
    """
    output_dir = datamodule.output_dir

    print(f"Active Learning Iteration {iteration}: Training DDPM model and generating samples")

    # Prepare checkpoint and output paths
    checkpoint_dir = output_dir / f"iter_{iteration}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint.ckpt"
    images_dir = checkpoint_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Ensure data splits are up to date
    datamodule.prepare_data_splits()

    # Check if we should resume from previous checkpoint
    resume_training = cfg.active_learning.get('resume_training', True)
    prev_checkpoint_path = None
    if resume_training and iteration > 0:
        prev_checkpoint_path = output_dir / f"iter_{iteration-1}" / "checkpoint.ckpt"
        if not prev_checkpoint_path.exists():
            print(f"Previous checkpoint not found at {prev_checkpoint_path}, training from scratch")
            prev_checkpoint_path = None
        else:
            print(f"Resuming training from previous checkpoint: {prev_checkpoint_path}")

    # Initialize model
    dtype = getattr(torch, cfg.dtype)
    if prev_checkpoint_path is not None:
        # Load from checkpoint using dynamic class loading
        model_class = get_class(cfg.model._target_)
        model = model_class.load_from_checkpoint(prev_checkpoint_path)
    else:
        model = instantiate(cfg.model)
    model = model.to(dtype=dtype)

    # Compile model for faster training
    if cfg.get('compile_model', False) and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Setup callbacks
    callbacks = get_training_callbacks(cfg, checkpoint_dir)

    # Trainer
    max_epochs = (cfg.train.n_compute_steps //
                  len(datamodule.train_dataloader().dataset))
    with open_dict(cfg):
        cfg.trainer.max_epochs = max_epochs
        # 16-mixed si le GPU a des tensor cores (compute capability >= 7.0)
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
            cfg.trainer.precision = "16-mixed"
        else:
            cfg.trainer.precision = 32
        
        # Add DDP strategy if multiple GPUs are available
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            cfg.trainer.strategy = "ddp"
            print(f"Multiple GPUs detected ({num_gpus}). Using DDP strategy.")
    
    trainer = pl.Trainer(**dict(cfg.trainer), callbacks=callbacks, logger=logger)
    
    # Set logger prefix for training metrics if using wandb
    if logger is not None and isinstance(logger, WandbLogger):
        logger.experiment.define_metric("global_step", step_metric="trainer/global_step")
        # Store iteration in logger for metric prefixing
        logger._iteration = iteration

    # Auto-find batch size if enabled
    if cfg.trainer.get('auto_scale_batch_size', False):
        print("Finding optimal batch size...")
        tuner = Tuner(trainer)
        tuner.scale_batch_size(model, datamodule=datamodule, mode='binsearch')
        print(f"Optimal batch size found: {datamodule.batch_size}")

    print(f"Training DDPM for up to {max_epochs} epochs...")

    # Train the model with datamodule
    trainer.fit(model, datamodule)

    # Save final checkpoint
    trainer.save_checkpoint(checkpoint_path)

    # Generate samples
    inference = instantiate(cfg.model.inference)
    samples = inference(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        savepath=images_dir,
        meep_eval=False
    )

    # Convert to torch tensor with configured dtype
    dtype = getattr(torch, cfg.dtype)
    samples = torch.from_numpy(samples).to(dtype=dtype)

    print(f"Generated {len(samples)} samples with shape {samples.shape}")

    return samples


def _compute_padded_image_shape(image_shape, unet_depth):
    """Compute the padded image shape that is divisible by 2**unet_depth."""
    padded = []
    for dim in image_shape:
        padded.append(((dim - 1) // 2**unet_depth + 1) * 2**unet_depth)
    return padded


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:

    # Compute padded image shape once and inject into config so all components
    # (inference, etc.) can create correctly-sized tensors without manual padding.
    with open_dict(cfg):
        cfg.data.padded_image_shape = _compute_padded_image_shape(
            cfg.data.image_shape, cfg.datamodule.unet_depth
        )

    # Initialize wandb if enabled
    use_wandb = cfg.wandb.get('enabled', False) and not cfg.debug
    logger = None
    if use_wandb and not cfg.trainer.fast_dev_run:
        # Convert config to dict for wandb logging
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        logger = WandbLogger( project=cfg.wandb.project,
                                    entity=cfg.wandb.entity,
                                    name=cfg.wandb.name,
                                    config=config_dict)

    datamodule: NanophotoDataModule = instantiate(cfg.datamodule)
    datamodule.setup()
    print(datamodule)

    # Load pre-selected generated images from a previous run
    gen_images_path = cfg.active_learning.get('gen_images_path', None)
    if gen_images_path is not None:
        gen_images_path = Path(gen_images_path)
        if not gen_images_path.exists():
            print(f"Warning: gen_images_path={gen_images_path} does not exist, skipping")
        else:
            dtype = getattr(torch, cfg.dtype)
            gen_images = torch.load(gen_images_path, weights_only=False).to(dtype=dtype)
            print(f"Loaded {len(gen_images)} pre-selected generated images from {gen_images_path}")
            datamodule.add_samples(gen_images)

    fom_threshold = cfg.active_learning.fom_threshold
    distance_threshold = cfg.active_learning.distance_threshold
    max_iterations = cfg.active_learning.max_iterations
    start_iteration = datamodule.start_iteration

    for iteration in range(start_iteration, start_iteration + max_iterations):
        print(f"# ITERATION {iteration} (#{iteration - start_iteration + 1}/{max_iterations})")

        # Step 1: Train DDPM model and generate samples
        skip_initial = cfg.active_learning.get('skip_initial_training', False)
        if iteration == start_iteration and skip_initial:
            # Skip training, just generate from existing checkpoint
            checkpoint_dir = datamodule.output_dir / f"iter_{iteration}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            images_dir = checkpoint_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            
            # Use load_ckpt_path if provided, otherwise look in iter_0
            load_ckpt_path = cfg.active_learning.get('load_ckpt_path', None)
            if load_ckpt_path:
                checkpoint_path = load_ckpt_path
            else:
                checkpoint_path = checkpoint_dir / "checkpoint.ckpt"
            
            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(
                    f"skip_initial_training=true but no checkpoint at {checkpoint_path}"
                )
            
            print(f"Skipping initial training, generating from {checkpoint_path}")
            inference = instantiate(cfg.model.inference)
            samples = inference(
                cfg=cfg,
                checkpoint_path=checkpoint_path,
                savepath=images_dir,
                meep_eval=False
            )
            dtype = getattr(torch, cfg.dtype)
            samples = torch.from_numpy(samples).to(dtype=dtype)
        else:
            samples = train_and_generate_samples(
                datamodule, logger, cfg, iteration)

        # Step 2: Select samples based on distance
        assert samples.shape[0] > 0, "No samples generated"
        distances = compute_distances(samples, datamodule.training_data)
        samples_after_dist = dist_select(samples, distances, distance_threshold)
        if use_wandb:
            metrics = {
                'distance_mean': distances.mean().item(),
                'distance_min': distances.min().item(),
                'distance_max': distances.max().item(),
                'binarization': binarisation(samples),
            }

        # Step 3: Compute FOM scores and filter similar samples
        fom_scores = compute_fom_scores(samples_after_dist, cfg)
        samples_after_fom, fom_scores = filter_similar_samples(samples_after_dist, fom_scores, distance_threshold)
        selected_samples, selected_fom = fom_select(samples_after_fom, fom_scores, fom_threshold)
        assert selected_samples.shape[0] > 0, "No samples generated"
        selected_dist = compute_distances(selected_samples,
                                      datamodule.training_data)

        #TODO: start from previous model checkpoint
        #TODO: add patience for training loss and early stopping
        #TODO: if the above is not enough, replace the ddpm by denoising
        #diffusion implicit model

        # Log metrics to wandb
        if use_wandb and len(fom_scores) > 0:
            metrics.update({
                'fom_mean': fom_scores.mean().item(),
                'fom_min': fom_scores.min().item(),
                'fom_max': fom_scores.max().item(),
                'fom_std': fom_scores.std().item(),
            })
            if len(selected_samples) > 0:
                selected_indices = (fom_scores > fom_threshold).nonzero(as_tuple=True)[0]
                if len(selected_indices) > 0:
                    metrics.update({
                        'selected_fom_mean': selected_fom.mean().item(),
                        'selected_fom_std': selected_fom.std().item(),
                        'selected_fom_min': selected_fom.min().item(),
                        'selected_fom_max': selected_fom.max().item(),
                        'selected_dist_mean': selected_dist.mean().item(),
                        'selected_fom_std': selected_dist.std().item(),
                        'selected_fom_min': selected_dist.min().item(),
                        'selected_fom_max': selected_dist.max().item(),

                    })
            wandb.log(metrics)

        # Save selected samples
        if len(selected_samples) > 0:
            selected_samples_path = datamodule.output_dir / f"selected_samples_iter_{iteration}.pt"
            torch.save(selected_samples, selected_samples_path)
            selected_fom_scores_path = datamodule.output_dir / f"selected_fom_scores_iter_{iteration}.pt"
            torch.save(selected_fom_scores, selected_fom_scores_path)
            print(f"Saved {len(selected_samples)} selected samples to {selected_samples_path}")

        # Step 5: Update training data
        if len(selected_samples) == 0:
            print("\nNo new samples selected. Stopping active learning.")
            break

        datamodule.add_samples(selected_samples)
        datamodule.save_checkpoint(iteration)

    datamodule.save_new_samples()

    print(f"Initial dataset size: {len(datamodule.initial_data)}")
    print(f"New samples added: {sum(len(s) for s in datamodule.new_samples)}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
