import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Optional, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import *
from utils import *

class Trainer:
  def __init__(
    self,
    device: torch.device,
    model,
    model_path: str,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    checkpoint_path: Optional[str] = None,
    writer_path: Optional[str] = None,
    eval_epoch = False,
    skip_training = False,
    save_checkpoints = False,
    early_stopping = True
  ):
    self.device = device
    self.model = model
    self.model_path = model_path
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.checkpoint_path = checkpoint_path
    self.writer_path = writer_path
    self.eval_epoch = eval_epoch
    self.skip_training = skip_training
    self.save_checkpoints = save_checkpoints
    self.scheduler = None
    self.ema_model = None
    self.early_stopping = early_stopping

    self.loss_func = nn.CrossEntropyLoss()  # TODO: weights depending on class balance
    self.optim = torch.optim.AdamW(self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE)
    if EMA:
      self.ema_model = torch.optim.swa_utils.AveragedModel(
        self.model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
      )

    self.start_epoch, self.step, self.vstep, self.min_epoch_vloss, self.stop_cnt = self.load_checkpoint(self.checkpoint_path)
    self.start_epoch += 1 # resume from the next epoch

    if not writer_path:
      today = str(datetime.now()).replace(" ", "_")
      auto_name = "-".join([model_path.split('/')[-1].split('.')[0], today, f"lr_{LR}", f"bs_{BATCH_SIZE}"])
      writer_path = str("runs/" + auto_name).replace(":", "_").replace(".", "_")
      self.writer = SummaryWriter(writer_path)
    else:
      self.writer = SummaryWriter(writer_path, purge_step=cast(int, None), max_queue=10, flush_secs=30)
    print("[*] Tensorboard output path:", writer_path)

    self.train_metrics = {
      "loss": [],
      "pixel_acc": [],
      "IoU": [],
      "Dice": [],
      "F1": [],
      "Hausdorff": [],
    }
    self.val_metrics = {
      "loss": [],
      "pixel_acc": [],
      "IoU": [],
      "Dice": [],
      "F1": [],
      "Hausdorff": [],
    }

  def save_checkpoint(self, epoch, step, vstep, min_loss, stop_cnt, best=False):
    chpt_path = self.model_path.split(".")[0] + f"_best.pt" if best else self.model_path.split(".")[0] + ".pt"
    checkpoint = {
      "epoch": epoch,
      "step": step,
      "vstep": vstep,
      "min_loss": min_loss,
      "stop_cnt": stop_cnt,
      "model": self.ema_model.module.state_dict() if EMA else self.model.state_dict(),
      "optimizer": self.optim.state_dict(),
      "scheduler": self.scheduler.state_dict() if self.scheduler else None,
    }
    torch.save(checkpoint, chpt_path)
    print(f"[+] Checkpoint saved at {chpt_path}. New min eval loss {min_loss}")

  def load_checkpoint(self, chpt_path):
    if chpt_path is None or not os.path.exists(chpt_path):
      print(f"[!] No checkpoint found at {chpt_path}")
      return -1, 0, 0, float("inf"), 0

    checkpoint = torch.load(chpt_path, map_location=self.device)

    # load model
    if EMA:
      if self.ema_model is None:
        self.ema_model = torch.optim.swa_utils.AveragedModel(
          self.model,
          multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
        )
      self.ema_model.module.load_state_dict(checkpoint["model"])
    self.model.load_state_dict(checkpoint["model"])

    # load optimizer and scheduler
    if "optimizer" in checkpoint:
      self.optim.load_state_dict(checkpoint["optimizer"])
    if self.scheduler and "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
      self.scheduler.load_state_dict(checkpoint["scheduler"])

    # load training state
    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)
    vstep = checkpoint.get("vstep", 0)
    min_loss = checkpoint.get("min_loss", float("inf"))
    stop_cnt = checkpoint.get("stop_cnt", 0)

    print(f"[+] Resumed from checkpoint {chpt_path} (epoch {epoch})")
    return epoch, step, vstep, min_loss, stop_cnt

  def log_scalars(
      self,
      tag_prefix: str,
      metrics: dict,
      step: int,
      accumulators: Optional[dict] = None
  ):
    for name, value in metrics.items():
      self.writer.add_scalar(f"{tag_prefix}/{name}", value, step)
      if accumulators is not None and name in accumulators:
        accumulators[name].append(value)

  def train_step(self, t, step, sample_batched, optim):
    image_batch, mask_batch = sample_batched
    X = image_batch.to(self.device)
    Y = mask_batch.to(self.device)

    out = self.model(X)
    optim.zero_grad()

    loss = self.loss_func(out, Y)
    metrics = compute_metrics(out.detach(), Y.detach())
    loss.backward()
    optim.step()
    # if self.scheduler: self.scheduler.step()

    if EMA: self.ema_model.update_parameters(self.model)

    current_metrics = {
      "loss": loss.item(),
      "pixel_acc": metrics["pixel_acc"],
      "IoU": metrics["IoU"],
      "Dice": metrics["Dice"],
      "F1": metrics["F1"],
      "Hausdorff": metrics["Hausdorff"],
    }
    self.log_scalars(
      "running train",
      current_metrics,
      step,
      self.epoch_train_metrics
    )
    t.set_description("[train] " + " | ".join(
      f"{name}: {value:.4f}" for name, value in current_metrics.items()
    ))

  def train(self):
    try:
      min_epoch_vloss = self.min_epoch_vloss
      step = self.step
      vstep = self.vstep
      stop_cnt = self.stop_cnt

      print("[*] Training...")
      for epoch in range(self.start_epoch, EPOCHS):
        self.epoch_train_metrics = {
          "loss": [],
          "pixel_acc": [],
          "IoU": [],
          "Dice": [],
          "F1": [],
          "Hausdorff": [],
        }
        self.epoch_val_metrics = {
          "loss": [],
          "pixel_acc": [],
          "IoU": [],
          "Dice": [],
          "F1": [],
          "Hausdorff": [],
        }

        self.model.train()
        print(f"\n[=>] Epoch {epoch+1}/{EPOCHS}")
        for i_batch, sample_batched in enumerate((t := tqdm(self.train_loader))):
          self.train_step(t, step, sample_batched, self.optim)
          step += 1

        avg_metrics = {name: np.mean(values) for name, values in self.epoch_train_metrics.items()}
        self.log_scalars("epoch training", avg_metrics, epoch, self.train_metrics)
        print("[->] Epoch average training metrics: " + " | ".join(
          f"{name}: {value:.4f}" for name, value in avg_metrics.items()
        ))

        avg_epoch_vloss = None
        if self.eval_epoch:
          vstep, avg_epoch_vloss = self.eval(vstep, epoch)

        if self.scheduler:
          self.scheduler.step(avg_epoch_vloss)
          # print(f"LR: {self.optim.param_groups[0]['lr']}")

        # save checkpoints and early stop
        self.save_checkpoint(epoch, step, vstep, min_epoch_vloss, stop_cnt)
        if self.save_checkpoints and avg_epoch_vloss is not None and avg_epoch_vloss < min_epoch_vloss:
          min_epoch_vloss = avg_epoch_vloss
          stop_cnt = 0
          self.save_checkpoint(epoch, step, vstep, min_epoch_vloss, stop_cnt, best=True)
        else:
          stop_cnt += 1
          if self.early_stopping and stop_cnt >= EARLY_STOP_EPOCHS:
            print(f"[!] Early stopping at epoch {epoch+1}/{EPOCHS}.")
            break
    except KeyboardInterrupt:
      print("[*] Training interrupted. Saving model...")

    print("[+] Training done")

  def eval_step(self, t, vstep, sample_batched):
    X = sample_batched[0].to(self.device)
    Y = sample_batched[1].to(self.device)
    out = self.ema_model(X) if EMA else self.model(X)

    loss = self.loss_func(out, Y).mean()
    metrics = compute_metrics(out.detach(), Y.detach())

    current_metrics = {
      "loss": loss.item(),
      "pixel_acc": metrics["pixel_acc"],
      "IoU": metrics["IoU"],
      "Dice": metrics["Dice"],
      "F1": metrics["F1"],
      "Hausdorff": metrics["Hausdorff"],
    }
    self.log_scalars(
      "running val",
      current_metrics,
      vstep,
      self.epoch_val_metrics
    )
    t.set_description("[val] " + " | ".join(
      f"{name}: {value:.4f}" for name, value in current_metrics.items()
    ))

  def eval(self, vstep, epoch):
    with torch.no_grad():
      self.model.eval()
      for i_batch, sample_batched in enumerate((t := tqdm(self.val_loader))):
        self.eval_step(t, vstep, sample_batched)
        vstep += 1

      avg_metrics = {name: np.mean(values) for name, values in self.epoch_val_metrics.items()}
      self.log_scalars("epoch validation", avg_metrics, epoch, self.val_metrics)
      print("[->] Epoch average validation metrics: " + " | ".join(
        f"{name}: {value:.4f}" for name, value in avg_metrics.items()
      ))

    return vstep, avg_metrics["loss"] if "loss" in avg_metrics else None
