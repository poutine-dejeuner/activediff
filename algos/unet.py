import torch
import torch.nn as nn
import pytorch_lightning as pl
from timm.utils.model_ema import ModelEmaV3
import hydra
from omegaconf import OmegaConf


class DDPM_Scheduler(torch.nn.Module):
    def __init__(self, num_time_steps: int=1000, device:torch.device=torch.device("cpu")):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False, device=device)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)

    def forward(self, t):
        return self.beta[t], self.alpha[t]


class UNet(pl.LightningModule):
    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # Model setup
        self.model = hydra.utils.instantiate(cfg.model)
        self.scheduler_ddpm = DDPM_Scheduler(
            num_time_steps=cfg.num_time_steps, 
            device=self.device
        )
        self.criterion = nn.MSELoss(reduction='mean')

        # EMA
        self.ema_decay = cfg.ema_decay
        self.ema = None  # Initialized in configure_optimizers

        # Training config
        self.lr = cfg.lr
        self.num_time_steps = cfg.num_time_steps

    def forward(self, x, t):
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        x = batch[0]

        # Generate random timesteps and noise
        t = torch.randint(0, self.num_time_steps, (x.size(0),), device=self.device)
        e = torch.randn_like(x, device=self.device)
        a = self.scheduler_ddpm.alpha[t].view(-1, 1, 1, 1).to(self.device)

        # Forward diffusion
        x_noisy = (torch.sqrt(a) * x) + (torch.sqrt(1 - a) * e)

        # Predict noise
        output = self(x_noisy, t)
        loss = self.criterion(output, e)

        # Update EMA
        if self.ema is not None:
            self.ema.update(self.model)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch[0]

        # Generate random timesteps and noise
        t = torch.randint(0, self.num_time_steps, (x.size(0),), device=self.device)
        e = torch.randn_like(x, device=self.device)
        a = self.scheduler_ddpm.alpha[t].view(-1, 1, 1, 1).to(self.device)

        # Forward diffusion
        x_noisy = (torch.sqrt(a) * x) + (torch.sqrt(1 - a) * e)

        # Predict noise with EMA model if available
        if self.ema is not None:
            with self.ema.average_parameters():
                output = self(x_noisy, t)
        else:
            output = self(x_noisy, t)

        loss = self.criterion(output, e)

        self.log('test_loss', loss, prog_bar=True)
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
            self.ema = ModelEmaV3(self.model, decay=self.ema_decay)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr * 2,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_save_checkpoint(self, checkpoint):
        # Save EMA state
        if self.ema is not None:
            checkpoint['ema'] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        # Load EMA state
        if 'ema' in checkpoint:
            if self.ema is None:
                self.ema = ModelEmaV3(self.model, decay=self.ema_decay)
            self.ema.load_state_dict(checkpoint['ema'])
