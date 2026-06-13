import os
import sys
from pathlib import Path

import torch
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
