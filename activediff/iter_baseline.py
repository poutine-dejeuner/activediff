"""Baseline experiment: load a checkpoint, generate samples, compute FOM."""

import argparse
from pathlib import Path

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf, open_dict
from timm.utils.model_ema import ModelEmaV3
from tqdm import tqdm

from activediff.models.unet_utils import DDPM_Scheduler
from activediff.utils import compute_fom_scores, set_seed


def load_model_from_checkpoint(cfg, checkpoint_path, device):
    """Load a UNet model + EMA from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        model_state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("ema.")
        }
        ema_state = checkpoint.get("ema")
        if ema_state is None:
            ema_keys = {
                k.replace("ema.", ""): v
                for k, v in state_dict.items()
                if k.startswith("ema.")
            }
            ema_state = ema_keys if ema_keys else None
    elif "weights" in checkpoint:
        model_state_dict = checkpoint["weights"]
        ema_state = checkpoint.get("ema")
    else:
        model_state_dict = checkpoint
        ema_state = None

    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)
    model.load_state_dict(model_state_dict)

    ema = ModelEmaV3(model, decay=cfg.train.ema_decay)
    if ema_state is not None:
        ema.load_state_dict(ema_state)

    return ema


def generate_samples(cfg, ema, n_samples, batch_size, device, output_path):
    """Generate samples via DDPM reverse diffusion."""
    imagespath = Path(output_path) / 'images.npy'
    if imagespath.exists():
        images = np.load(imagespath)
        return images

    num_time_steps = cfg.model.time_steps
    image_shape = tuple(cfg.data.image_shape)
    padded_image_shape = tuple(cfg.data.padded_image_shape)
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps, device=device)

    model = ema.module.eval()
    all_samples = []
    num_batches = (n_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating samples"):
            cur_bs = min(batch_size, n_samples - batch_idx * batch_size)
            z = torch.randn((cur_bs, 1) + padded_image_shape, device=device)

            for t in reversed(range(1, num_time_steps)):
                t_batch = [t] * cur_bs
                temp = scheduler.beta[t] / (
                    torch.sqrt(1 - scheduler.alpha[t])
                    * torch.sqrt(1 - scheduler.beta[t])
                )
                z = (1 / torch.sqrt(1 - scheduler.beta[t])) * z - temp * model(
                    z, t_batch
                )
                z = z + torch.randn_like(z) * torch.sqrt(scheduler.beta[t])

            temp = scheduler.beta[0] / (
                torch.sqrt(1 - scheduler.alpha[0])
                * torch.sqrt(1 - scheduler.beta[0])
            )
            x = (1 / torch.sqrt(1 - scheduler.beta[0])) * z - temp * model(
                z, [0] * cur_bs
            )
            x = x[..., : image_shape[0], : image_shape[1]]
            all_samples.append(x.cpu())

    samples = torch.cat(all_samples, dim=0).squeeze(1)
    samples = samples.numpy()
    samples = (samples - samples.min()) / (samples.max() - samples.min())
    return samples


def _compute_padded_image_shape(image_shape, unet_depth):
    return [((d - 1) // 2**unet_depth + 1) * 2**unet_depth for d in image_shape]


def run(checkpoint_path: str, n_samples: int, batch_size: int = 32,
        output_dir: str = "baseline_output", skip_meep: bool = False,
        seed: int = 42):
    """Load checkpoint, generate samples, compute FOM, save results."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load Hydra config
    with hydra.initialize(version_base="1.3", config_path="configs"):
        cfg = hydra.compose(config_name="config")

    unet_depth = OmegaConf.select(cfg, "datamodule.unet_depth", default=3)
    with open_dict(cfg):
        cfg.data.padded_image_shape = _compute_padded_image_shape(
            cfg.data.image_shape, unet_depth
        )
        cfg.fom.skip_meep = skip_meep

    print(f"Checkpoint : {checkpoint_path}")
    print(f"Samples    : {n_samples}")
    print(f"Device     : {device}")

    # 1. Load model
    ema = load_model_from_checkpoint(cfg, checkpoint_path, device)
    print("Model loaded.")

    # 2. Generate samples
    samples = generate_samples(cfg, ema, n_samples, batch_size, device, output_path)
    np.save(output_path / "images.npy", samples)
    print(f"Generated {len(samples)} samples — shape {samples.shape}")

    # 3. Compute FOM
    samples_tensor = torch.from_numpy(samples)
    fom_scores = compute_fom_scores(samples_tensor, cfg)
    np.save(output_path / "fom.npy", fom_scores.numpy())

    print(f"\n--- Results ---")
    print(f"FOM  mean={fom_scores.mean():.4f}  std={fom_scores.std():.4f}  "
          f"min={fom_scores.min():.4f}  max={fom_scores.max():.4f}")

    return samples, fom_scores


def main():
    parser = argparse.ArgumentParser(description="Baseline: generate & evaluate FOM")
    parser.add_argument("-c", "--checkpoint", type=str, help="Path to model checkpoint", default="outputs/checkpoints/checkpoint.ckpt")
    parser.add_argument("-n", "--n-samples", type=int, default=600,
                        help="Number of samples to generate")
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("-o", "--output-dir", type=str, default="baseline_output")
    parser.add_argument("--skip-meep", action="store_true",
                        help="Use random FOM scores (for testing)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        checkpoint_path=args.checkpoint,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        skip_meep=args.skip_meep,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
