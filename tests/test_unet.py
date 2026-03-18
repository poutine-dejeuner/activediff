import pytest
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from activediff.models.unet import UNet
from activediff.models.unet_utils import compute_unet_channels


# Shared small-model kwargs to keep tests fast
SMALL_MODEL_KWARGS = dict(
    first_channels=8,
    num_layers=4,
    input_channels=1,
    output_channels=1,
    num_groups=8,
    dropout_prob=0.0,
    num_heads=4,
    time_steps=20,
    image_shape=[16, 16],
    lr=1e-3,
    ema_decay=0.999,
    ema_update_every=1,
    Attentions=[False, True, False, True],
    Upscales=[False, False, True, True],
    device="cpu",
)


def _make_dummy_loader(n=32, image_size=(16, 16), batch_size=8):
    data = torch.randn(n, 1, *image_size)
    return DataLoader(TensorDataset(data), batch_size=batch_size, drop_last=True)


# ── Unit tests ───────────────────────────────────────────────────────────────


class TestUNetForward:
    def test_output_shape(self):
        model = UNet(**SMALL_MODEL_KWARGS)
        x = torch.randn(2, 1, 16, 16)
        t = torch.randint(0, 20, (2,))
        out = model(x, t)
        assert out.shape == (2, 1, 16, 16)

    def test_output_shape_list_timesteps(self):
        """The model should accept a list of ints for t (used by callbacks)."""
        model = UNet(**SMALL_MODEL_KWARGS)
        x = torch.randn(2, 1, 16, 16)
        out = model(x, [5, 10])
        assert out.shape == (2, 1, 16, 16)

    def test_channels_computation(self):
        channels = compute_unet_channels(initial_n_channels=8, n_layers=4)
        assert len(channels) == 4
        assert channels[0] == 8


class TestUNetEMA:
    def test_ema_initialized_after_configure_optimizers(self):
        model = UNet(**SMALL_MODEL_KWARGS)
        assert model.ema is None
        trainer = pl.Trainer(max_epochs=1, fast_dev_run=True, logger=False,
                             enable_progress_bar=False, accelerator="cpu")
        trainer.fit(model, _make_dummy_loader())
        assert model.ema is not None

    def test_ema_saved_in_checkpoint(self, tmp_path):
        model = UNet(**SMALL_MODEL_KWARGS)
        trainer = pl.Trainer(max_epochs=1, fast_dev_run=True, logger=False,
                             enable_progress_bar=False, accelerator="cpu")
        trainer.fit(model, _make_dummy_loader())
        ckpt_path = tmp_path / "ema.ckpt"
        trainer.save_checkpoint(ckpt_path)

        ckpt = torch.load(ckpt_path, weights_only=False)
        assert "ema" in ckpt, "EMA state missing from checkpoint"

    def test_ema_restored_from_checkpoint(self, tmp_path):
        model = UNet(**SMALL_MODEL_KWARGS)
        trainer = pl.Trainer(max_epochs=1, fast_dev_run=True, logger=False,
                             enable_progress_bar=False, accelerator="cpu")
        trainer.fit(model, _make_dummy_loader())
        ckpt_path = tmp_path / "ema.ckpt"
        trainer.save_checkpoint(ckpt_path)

        loaded = UNet.load_from_checkpoint(ckpt_path)
        assert loaded.ema is not None, "EMA not restored on load_from_checkpoint"


# ── Scheduler tests ──────────────────────────────────────────────────────────


class TestScheduler:
    def test_configure_optimizers_returns_scheduler(self):
        """configure_optimizers must return an LR scheduler dict."""
        model = UNet(**SMALL_MODEL_KWARGS)
        loader = _make_dummy_loader()
        trainer = pl.Trainer(max_epochs=2, logger=False,
                             enable_progress_bar=False, accelerator="cpu")
        trainer.fit(model, loader)

        opt_cfg = model.configure_optimizers()
        assert isinstance(opt_cfg, dict), "configure_optimizers should return a dict"
        assert "lr_scheduler" in opt_cfg, "Missing lr_scheduler key"
        sched_cfg = opt_cfg["lr_scheduler"]
        assert sched_cfg["interval"] == "step"
        assert isinstance(sched_cfg["scheduler"],
                          torch.optim.lr_scheduler.OneCycleLR)

    def test_lr_changes_during_training(self):
        """LR should vary across training thanks to OneCycleLR."""
        model = UNet(**SMALL_MODEL_KWARGS)
        loader = _make_dummy_loader(n=32, batch_size=8)
        trainer = pl.Trainer(max_epochs=4, logger=False,
                             enable_progress_bar=False, accelerator="cpu")
        trainer.fit(model, loader)

        # After training, the OneCycleLR should have annealed the LR
        optimizer = trainer.optimizers[0]
        final_lr = optimizer.param_groups[0]["lr"]
        # OneCycleLR ends near 0; it should NOT still be at the initial lr
        assert final_lr < model.lr, (
            f"LR did not anneal: final_lr={final_lr}, initial={model.lr}"
        )


# ── Training integration tests ───────────────────────────────────────────────


class TestTraining:
    def test_training_step_reduces_loss(self):
        """Loss should decrease after a few epochs of training."""
        torch.manual_seed(42)
        model = UNet(**SMALL_MODEL_KWARGS)
        data = torch.randn(16, 1, 16, 16)
        loader = DataLoader(TensorDataset(data), batch_size=8, drop_last=True)

        trainer = pl.Trainer(max_epochs=200, logger=False,
                             enable_progress_bar=False, accelerator="cpu")
        trainer.fit(model, loader)

        final_loss = trainer.callback_metrics.get("train/loss")
        assert final_loss is not None, "train/loss not logged"
        assert final_loss < 0.9, f"Loss did not decrease enough: {final_loss:.4f}"

    def test_validation_step_logs_val_loss(self):
        model = UNet(**SMALL_MODEL_KWARGS)
        train_loader = _make_dummy_loader()
        val_loader = _make_dummy_loader(n=16)

        trainer = pl.Trainer(max_epochs=2, logger=False,
                             enable_progress_bar=False, accelerator="cpu",
                             check_val_every_n_epoch=1)
        trainer.fit(model, train_loader, val_loader)

        val_loss = trainer.callback_metrics.get("val/loss")
        assert val_loss is not None, "val/loss not logged"

    def test_checkpoint_save_and_resume(self, tmp_path):
        """Training, save, reload, and re-train should work without errors."""
        model = UNet(**SMALL_MODEL_KWARGS)
        loader = _make_dummy_loader()
        ckpt_path = tmp_path / "resume.ckpt"

        trainer1 = pl.Trainer(max_epochs=2, logger=False,
                              enable_progress_bar=False, accelerator="cpu")
        trainer1.fit(model, loader)
        trainer1.save_checkpoint(ckpt_path)

        model2 = UNet.load_from_checkpoint(ckpt_path)
        trainer2 = pl.Trainer(max_epochs=2, logger=False,
                              enable_progress_bar=False, accelerator="cpu")
        trainer2.fit(model2, loader)

