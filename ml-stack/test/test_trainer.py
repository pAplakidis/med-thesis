import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ML_STACK_DIR = Path(__file__).resolve().parents[1]
if str(ML_STACK_DIR) not in sys.path:
  sys.path.insert(0, str(ML_STACK_DIR))
os.chdir(ML_STACK_DIR)

from config import IMAGE_SIZE
from models.presets import PRESETS
from models.unet import UNet
from trainer import Trainer


class _DummyTqdm:
  def set_description(self, *_args, **_kwargs):
    return None


def _make_trainer(model, loader, tmp_path):
  return Trainer(
    device=torch.device("cpu"),
    model=model,
    model_path=str(tmp_path / "model.pt"),
    train_loader=loader,
    val_loader=loader,
    checkpoint_path=None,
    writer_path=str(tmp_path / "tb"),
    eval_epoch=False,
    save_checkpoints=False,
    early_stopping=False,
    dataset=None,
    finetune=False,
  )


def test_trainer_segmentation_only_step(tmp_path):
  model = UNet(PRESETS["unet_tiny"].to_config())

  images = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)
  masks = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.long)
  loader = DataLoader(TensorDataset(images, masks), batch_size=1)

  trainer = _make_trainer(model, loader, tmp_path)
  trainer.epoch_train_metrics = {key: [] for key in trainer.metric_names}
  trainer.epoch_val_metrics = {key: [] for key in trainer.metric_names}

  batch = next(iter(loader))
  trainer.train_step(_DummyTqdm(), 0, batch, trainer.optim)

  assert trainer.epoch_train_metrics["loss"]
  assert trainer.epoch_train_metrics["seg_loss"]
  assert "clf_loss" not in trainer.epoch_train_metrics


def test_trainer_multitask_step(tmp_path):
  model = UNet(PRESETS["unet_tiny"].to_config(), multitask=True)

  images = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)
  masks = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.long)
  clf_targets = torch.zeros(1, model.num_classes)
  clf_targets[:, 0] = 1.0
  clf_targets[:, 1] = 1.0
  loader = DataLoader(TensorDataset(images, masks, clf_targets), batch_size=1)

  trainer = _make_trainer(model, loader, tmp_path)
  trainer.epoch_train_metrics = {key: [] for key in trainer.metric_names}
  trainer.epoch_val_metrics = {key: [] for key in trainer.metric_names}

  batch = next(iter(loader))
  trainer.train_step(_DummyTqdm(), 0, batch, trainer.optim)
  trainer.eval_step(_DummyTqdm(), 0, batch)

  assert trainer.epoch_train_metrics["loss"]
  assert trainer.epoch_train_metrics["clf_loss"]
  assert trainer.epoch_train_metrics["clf_acc"]
  assert trainer.epoch_val_metrics["clf_loss"]


def test_trainer_finetune_loads_model_only(tmp_path):
  seg_model = UNet(PRESETS["unet_tiny"].to_config())
  optimizer = torch.optim.AdamW(seg_model.parameters(), lr=1e-3)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

  # create optimizer state so we can verify it is ignored during finetune loading
  x = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)
  y = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.long)
  loss = nn.CrossEntropyLoss()(seg_model(x), y)
  loss.backward()
  optimizer.step()
  scheduler.step(loss.item())

  checkpoint_path = tmp_path / "seg_checkpoint.pt"
  torch.save(
    {
      "epoch": 7,
      "step": 99,
      "vstep": 55,
      "min_loss": 0.123,
      "stop_cnt": 3,
      "model": seg_model.state_dict(),
      "writer": str(tmp_path / "old_runs"),
      "optimizer": optimizer.state_dict(),
      "scheduler": scheduler.state_dict(),
      "config": {},
    },
    checkpoint_path,
  )

  multitask_model = UNet(PRESETS["unet_tiny"].to_config(), multitask=True)
  loader = DataLoader(TensorDataset(x, y), batch_size=1)
  trainer = Trainer(
    device=torch.device("cpu"),
    model=multitask_model,
    model_path=str(tmp_path / "model.pt"),
    train_loader=loader,
    val_loader=loader,
    checkpoint_path=str(checkpoint_path),
    writer_path=str(tmp_path / "tb"),
    eval_epoch=False,
    save_checkpoints=False,
    early_stopping=False,
    dataset=None,
    finetune=True,
  )

  assert trainer.start_epoch == 0
  assert trainer.step == 0
  assert trainer.vstep == 0
  assert trainer.min_epoch_vloss == float("inf")
  assert trainer.stop_cnt == 0
  assert trainer.writer is not None
  assert trainer.optim.state_dict()["state"] == {}
  assert trainer.scheduler.state_dict()["last_epoch"] == 0


def test_trainer_finetune_skips_shape_mismatched_head(tmp_path):
  source_config = PRESETS["unet_tiny"].to_config()
  source_config.num_classes = 9
  seg_model = UNet(source_config)
  checkpoint_path = tmp_path / "seg_checkpoint_9class.pt"
  torch.save({"model": seg_model.state_dict()}, checkpoint_path)

  target_model = UNet(PRESETS["unet_tiny"].to_config(), multitask=True)
  loader = DataLoader(TensorDataset(torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE), torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.long)), batch_size=1)
  trainer = Trainer(
    device=torch.device("cpu"),
    model=target_model,
    model_path=str(tmp_path / "model.pt"),
    train_loader=loader,
    val_loader=loader,
    checkpoint_path=str(checkpoint_path),
    writer_path=str(tmp_path / "tb2"),
    eval_epoch=False,
    save_checkpoints=False,
    early_stopping=False,
    dataset=None,
    finetune=True,
  )

  assert trainer.start_epoch == 0
  assert trainer.step == 0
  assert trainer.vstep == 0
  assert trainer.min_epoch_vloss == float("inf")
  assert trainer.stop_cnt == 0
  assert trainer.writer is not None


def test_trainer_auto_writer_path_uses_experiment_name(tmp_path, monkeypatch):
  captured = {}

  class DummyWriter:
    def __init__(self, path, *args, **kwargs):
      captured["path"] = path

    def add_scalar(self, *args, **kwargs):
      return None

    def close(self):
      return None

  monkeypatch.setattr("trainer.SummaryWriter", DummyWriter)

  model = UNet(PRESETS["unet_tiny"].to_config())
  images = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)
  masks = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.long)
  loader = DataLoader(TensorDataset(images, masks), batch_size=1)

  trainer = Trainer(
    device=torch.device("cpu"),
    model=model,
    model_path="checkpoints/equal_hybrid_seg_loss/unet_base_best.pt",
    train_loader=loader,
    val_loader=loader,
    checkpoint_path=None,
    writer_path=None,
    eval_epoch=False,
    save_checkpoints=False,
    early_stopping=False,
    dataset=None,
    finetune=False,
  )

  assert trainer.writer is not None
  assert captured["path"].startswith("runs/equal_hybrid_seg_loss-unet_base_best-")
