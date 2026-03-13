import sys
import os
import warnings
from typing import List
import random
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import device
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from einops import rearrange
import hydra
from timm.utils.model_ema import ModelEmaV3
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from activediff.models.unet_utils import (UNetPad, display_reverse,
                                        compute_unet_channels, DDPM_Scheduler)
from activediff.utils import set_seed


class ResBlock(nn.Module):
    def __init__(self, C: int, num_groups: int, dropout_prob: float):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)

    def forward(self, x, embeddings):
        x = x + embeddings[:, :x.shape[1], :, :]
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        return r + x


class Attention(nn.Module):
    def __init__(self, C: int, num_heads: int, dropout_prob: float):
        super().__init__()
        self.proj1 = nn.Linear(C, C*3)
        self.proj2 = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        h, w = x.shape[2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj1(x)
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
        q, k, v = x[0], x[1], x[2]
        x = F.scaled_dot_product_attention(
            q, k, v, is_causal=False, dropout_p=self.dropout_prob)
        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        x = self.proj2(x)
        return rearrange(x, 'b h w C -> b C h w')


class UnetLayer(nn.Module):
    def __init__(self,
                 upscale: bool,
                 attention: bool,
                 num_groups: int,
                 dropout_prob: float,
                 num_heads: int,
                 C: int):
        super().__init__()
        self.ResBlock1 = ResBlock(
            C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        self.ResBlock2 = ResBlock(
            C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        if upscale:
            self.conv = nn.ConvTranspose2d(
                C, C//2, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(C, C*2, kernel_size=3, stride=2, padding=1)
        if attention:
            self.attention_layer = Attention(
                C, num_heads=num_heads, dropout_prob=dropout_prob)

    def forward(self, x, embeddings):
        x = self.ResBlock1(x, embeddings)
        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)
        x = self.ResBlock2(x, embeddings)
        return self.conv(x), x


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps: int, embed_dim: int, device: device):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float()
                        * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings.to(device)

    def forward(self, t):
        embeds = self.embeddings[t]
        return embeds[:, :, None, None]


class UNet(pl.LightningModule):
    def __init__(self,
                 # default number of channels [32, 64, 128, 256, 256, 192]
                 Channels: List | None = None,
                 Attentions: List = [False, True, False, False, False, True],
                 first_channels: int | None = None,
                 num_layers: int | None = None,
                 Upscales: List = [False, False, False, True, True, True],
                 num_groups: int = 32,
                 dropout_prob: float = 0.0,
                 num_heads: int = 8,
                 input_channels: int = 1,
                 output_channels: int = 1,
                 device: device = 'cuda',
                 time_steps: int = 1000,
                 image_shape: List | None = None,
                 lr: float = 1e-4,
                 ema_decay: float = 0.9999,
                 ema_update_every: int = 10,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if Channels is None:
            Channels = compute_unet_channels(first_channels, num_layers)
        if Channels[0] < num_groups:
            warnings.warn(f"""Channels[0]=={Channels[0]}<{num_groups}==num_groups
            and therefore was lowered""")
            first_channels = num_groups
            Channels = compute_unet_channels(first_channels, num_layers)
        self.num_layers = len(Channels)
        self.shallow_conv = nn.Conv2d(
            input_channels, Channels[0], kernel_size=3, padding=1)
        out_channels = (Channels[-1]//2)+Channels[0]
        self.late_conv = nn.Conv2d(
            out_channels, out_channels//2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(
            out_channels//2, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.embeddings = SinusoidalEmbeddings(
            time_steps=time_steps, embed_dim=max(Channels), device=device)
        for i in range(self.num_layers):
            layer = UnetLayer(
                upscale=Upscales[i],
                attention=Attentions[i],
                num_groups=num_groups,
                dropout_prob=dropout_prob,
                C=Channels[i],
                num_heads=num_heads
            )
            setattr(self, f'Layer{i+1}', layer)

        # Training parameters
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema_update_every = ema_update_every
        self.time_steps = time_steps
        self.image_shape = list(image_shape) if image_shape is not None else None
        self.ema = None
        self.criterion = nn.MSELoss(reduction='mean')
        self.scheduler_ddpm = DDPM_Scheduler(num_time_steps=time_steps, device=device)

    def forward(self, x, t):
        x = self.shallow_conv(x)
        residuals = []
        for i in range(self.num_layers//2):
            layer = getattr(self, f'Layer{i+1}')
            embeddings = self.embeddings(t)
            x, r = layer(x, embeddings)
            residuals.append(r)
        for i in range(self.num_layers//2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            x = torch.concat(
                (layer(x, embeddings)[0], residuals[self.num_layers-i-1]), dim=1)
        return self.output_conv(self.relu(self.late_conv(x)))

    def training_step(self, batch, batch_idx):
        x = batch[0]

        # Generate random timesteps and noise
        t = torch.randint(0, self.time_steps, (x.size(0),), device=self.device)
        e = torch.randn_like(x, device=self.device)
        a = self.scheduler_ddpm.alpha[t].view(-1, 1, 1, 1).to(self.device)

        # Forward diffusion
        x_noisy = (torch.sqrt(a) * x) + (torch.sqrt(1 - a) * e)

        # Predict noise
        output = self(x_noisy, t)
        loss = self.criterion(output, e)
        if self.global_step % self.ema_update_every == 0:
            self.ema.update(self)

        self.log('train/loss', loss, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]

        t = torch.randint(0, self.time_steps, (x.size(0),), device=self.device)
        e = torch.randn_like(x, device=self.device)
        a = self.scheduler_ddpm.alpha[t].view(-1, 1, 1, 1).to(self.device)

        x_noisy = (torch.sqrt(a) * x) + (torch.sqrt(1 - a) * e)
        output = self(x_noisy, t)
        loss = self.criterion(output, e)

        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        # Initialize EMA after optimizer
        if self.ema is None:
            self.ema = ModelEmaV3(self, decay=self.ema_decay)

        return optimizer

    def on_save_checkpoint(self, checkpoint):
        # Save EMA state
        if self.ema is not None:
            checkpoint['ema'] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        # Load EMA state
        if 'ema' in checkpoint:
            if self.ema is None:
                self.ema = ModelEmaV3(self, decay=self.ema_decay)
            self.ema.load_state_dict(checkpoint['ema'])


def train(data: np.ndarray, cfg, checkpoint_path: os.PathLike, savedir: os.PathLike,
          run=None):
    seed = -1
    n_epochs = cfg.n_epochs
    lr = cfg.lr
    batch_size = cfg.batch_size
    num_time_steps = cfg.model.time_steps
    ema_decay = cfg.ema_decay
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert data.shape[-2:] == cfg.data.image_shape
    print("TRAINING")
    print(f"{n_epochs} epochs total")
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)
    dtype = torch.float32

    data = torch.tensor(data, dtype=dtype)
    if data.ndim == 3:
        data = data.unsqueeze(1)
    assert data.ndim == 4

    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)

    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)
    depth = model.num_layers//2

    pad_fn = UNetPad(data[0:1], depth=depth)

    assert data.shape[-2:] == cfg.data.image_shape

    train_dataset = TensorDataset(data)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=4)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = nn.MSELoss(reduction='mean')

    for i in range(n_epochs):
        total_loss = 0
        for bidx, [x] in enumerate(tqdm(train_loader, desc=f"Epoch {i+1}/{n_epochs}",
                                        disable=not sys.stdout.isatty())):
            x = x.cuda()
            x = pad_fn(x)
            t = torch.randint(0, num_time_steps, (batch_size,))
            e = torch.randn_like(x, requires_grad=False)
            a = scheduler.alpha[t].view(batch_size, 1, 1, 1).cuda()
            x = (torch.sqrt(a)*x) + (torch.sqrt(1-a)*e)
            output = model(x, t)
            optimizer.zero_grad()
            loss = criterion(output, e)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            ema.update(model)

            if cfg.debug:
                break  # Only run one batch in debug mode
        print(f'Epoch {i+1} | Loss {total_loss / len(train_loader):.5f}')
        if run is not None:
            run.log({"loss": total_loss})
        if i % 100 == 0:
            checkpoint = {
                'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ema': ema.state_dict()
            }
            torch.save(checkpoint, checkpoint_path)
        if cfg.debug:
            break  # Only run one epoch in debug mode
    # report_objective(loss.item(), 'loss')
    return total_loss


def inference(cfg,
              checkpoint_path: str = None,
              savepath: str = "images",
              meep_eval: bool = True,
              **kwargs,
              ):
    num_time_steps = cfg.model.time_steps
    ema_decay = cfg.train.ema_decay
    n_images = cfg.active_learning.get('n_to_generate_debug', 2) if cfg.debug else cfg.active_learning.n_to_generate
    batch_size = cfg.generation.batch_size
    image_shape = tuple(cfg.data.image_shape)
    padded_image_shape = tuple(cfg.data.padded_image_shape)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print("INFERENCE")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Handle both Lightning checkpoints (.ckpt) and custom checkpoints (.pt)
    if 'state_dict' in checkpoint:
        # Lightning checkpoint format
        state_dict = checkpoint['state_dict']
        
        # Filter out EMA keys (they're saved in the checkpoint separately)
        model_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('ema.')}
        
        # Try to extract EMA state if it exists
        ema_state = checkpoint.get('ema')
        if ema_state is None:
            # EMA might be embedded in state_dict with 'ema.' prefix
            ema_keys = {k.replace('ema.', ''): v for k, v in state_dict.items() if k.startswith('ema.')}
            ema_state = ema_keys if ema_keys else None
            
    elif 'weights' in checkpoint:
        # Custom checkpoint format
        model_state_dict = checkpoint['weights']
        ema_state = checkpoint.get('ema')
    else:
        # Direct state dict
        model_state_dict = checkpoint
        ema_state = None

    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)

    model.load_state_dict(model_state_dict)
    ema = ModelEmaV3(model, decay=ema_decay)
    if ema_state is not None:
        ema.load_state_dict(ema_state)
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps, device=device)
    times = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999]

    with torch.no_grad():
        all_samples = []
        model = ema.module.eval()
        
        # Calculate number of batches
        num_batches = (n_images + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Generating samples", disable=not sys.stdout.isatty()):
            # Determine actual batch size for this iteration
            current_batch_size = min(batch_size, n_images - batch_idx * batch_size)
            
            # Initialize batch of noise at padded size directly
            z = torch.randn((current_batch_size, 1,) + padded_image_shape, device=device)

            # Reverse diffusion process for entire batch
            for t in reversed(range(1, num_time_steps)):
                t_batch = [t] * current_batch_size
                temp = (scheduler.beta[t]/((torch.sqrt(1-scheduler.alpha[t]))
                                           * (torch.sqrt(1-scheduler.beta[t]))))
                z = (1/(torch.sqrt(1-scheduler.beta[t]))) * z - (temp * model(z, t_batch))
                
                # Add noise
                e = torch.randn_like(z, device=device)
                z = z + (e * torch.sqrt(scheduler.beta[t]))
            
            # Final denoising step (t=0)
            temp = scheduler.beta[0]/((torch.sqrt(1-scheduler.alpha[0]))
                                      * (torch.sqrt(1-scheduler.beta[0])))
            x = (1/(torch.sqrt(1-scheduler.beta[0]))) * z - (temp * model(z, [0] * current_batch_size))

            # Crop back to original image shape and collect samples
            x = x[..., :image_shape[0], :image_shape[1]]
            all_samples.append(x.cpu())
        
        # Concatenate all batches
        samples = torch.cat(all_samples, dim=0).squeeze(1)
        
        # Save a visualization of the first sample
        if savepath and len(samples) > 0:
            x_vis = samples[0].numpy()
            plt.figure(figsize=(3, 3))
            plt.imshow(x_vis, cmap='gray')
            plt.axis('off')
            plt.savefig(savepath / "generated_sample_0.png", bbox_inches='tight')
            plt.close()
        
        samples = samples.numpy()
        samples = (samples - samples.min()) / (samples.max() - samples.min())
        assert samples.shape == (n_images,) + image_shape, samples.shape
        np.save(savepath / "images.npy", samples)

    return samples
