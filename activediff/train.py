from typing import Any
from collections import defaultdict

import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig, open_dict
import wandb

from activediff.datamodules import NanophotoDataModule
from activediff.utils import compute_fom_scores, compute_distances, dist_select, fom_select
from activediff.callbacks import get_training_callbacks


def train_and_generate_samples(datamodule, logger, cfg, iteration):
    """Train DDPM model and generate samples using PyTorch Lightning."""
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

    # Setup callbacks
    callbacks = get_training_callbacks(cfg, checkpoint_dir)

    # Trainer
    max_epochs = (cfg.train.n_compute_steps //
                  len(datamodule.train_dataloader().dataset))
    with open_dict(cfg):
        cfg.trainer.max_epochs = max_epochs
    
    trainer = pl.Trainer(**dict(cfg.trainer), callbacks=callbacks, logger=logger)

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


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:

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

    fom_threshold = cfg.active_learning.fom_threshold
    distance_threshold = cfg.active_learning.distance_threshold
    max_iterations = cfg.active_learning.max_iterations

    for iteration in range(max_iterations):
        print(f"# ITERATION {iteration + 1}/{max_iterations}")

        # Step 1: Train DDPM model and generate samples
        samples = train_and_generate_samples(
            datamodule, logger, cfg, iteration)

        # Step 2: Select samples based on distance
        distances = compute_distances(samples, datamodule.training_data)
        samples_after_dist = dist_select(samples, distances, distance_threshold)

        # Step 3: Select samples based on FOM
        fom_scores = compute_fom_scores(samples_after_dist, cfg)
        selected_samples = fom_select(samples_after_dist, fom_scores, fom_threshold)

        #TODO: start from previous model checkpoint
        #TODO: add patience for training loss and early stopping
        #TODO: if the above is not enough, replace the ddpm by denoising
        #diffusion implicit model

        # Log metrics to wandb
        if use_wandb:
            metrics = {
                'distance_mean': distances.mean().item(),
                'distance_min': distances.min().item(),
                'distance_max': distances.max().item(),
            }
            if len(fom_scores) > 0:
                metrics.update({
                    'fom_mean': fom_scores.mean().item(),
                    'fom_min': fom_scores.min().item(),
                    'fom_max': fom_scores.max().item(),
                })
            if len(selected_samples) > 0:
                selected_indices = (fom_scores > fom_threshold).nonzero(as_tuple=True)[0]
                if len(selected_indices) > 0:
                    metrics.update({
                        'selected_fom_mean': fom_scores[selected_indices].mean().item(),
                    })
            wandb.log(metrics)

        # Save selected samples
        if len(selected_samples) > 0:
            selected_samples_path = datamodule.output_dir / f"selected_samples_iter_{iteration}.pt"
            torch.save(selected_samples, selected_samples_path)
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
