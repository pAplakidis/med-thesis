import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Optional, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import generate_report
from config import *
from utils import *
from loss.hybrid_segmentation_loss import HybridSegmentationLoss

# TODO: loss is weighted sum, log dice + focal tversky + ce separately
SEG_METRICS = [
  "loss",
  "seg_loss",
  "pixel_acc",
  "IoU",
  "Dice",
  "F1",
  "Hausdorff",
]

CLF_METRICS = [
  "clf_loss",
  "clf_acc",
  "clf_exact_match",
  "clf_precision_micro",
  "clf_recall_micro",
  "clf_f1_micro",
  "clf_precision_macro",
  "clf_recall_macro",
  "clf_f1_macro",
]

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
    early_stopping = True,
    dataset = None,
    finetune = False
  ):
    self.device = device
    self.model = model
    self.model_path = model_path
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.checkpoint_path = checkpoint_path
    self.writer_path = writer_path
    self.writer = None
    self.eval_epoch = eval_epoch
    self.skip_training = skip_training
    self.save_checkpoints = save_checkpoints
    self.scheduler = None
    self.ema_model = None
    self.early_stopping = early_stopping
    self.finetune = finetune
    self.multitask = getattr(self.model, "multitask", False)
    self.lambda_clf = LAMBDA_CLF
    self.clf_loss_func = nn.BCEWithLogitsLoss().to(device)
    self.metric_names = SEG_METRICS + (CLF_METRICS if self.multitask else [])

    # compute class weights from per-pixel class balance
    weights = None
    # NOTE: applying class-balanced weights results in the model including all classes per prediction, but most predictions contain background + 1-2 classes
    # if dataset is not None:
    #   pixel_counts = dataset.get_class_balance_per_pixel()
    #   total_pixels = sum(pixel_counts.values())
    #   num_classes = len(pixel_counts)
    #   weights = torch.tensor([
    #     total_pixels / (num_classes * pixel_counts[cls]) if pixel_counts[cls] > 0 else 0.0
    #     for cls in range(num_classes)
    #   ], dtype=torch.float32).to(device)
    #   print("[*] Class weights:", weights.cpu().numpy())

    self.loss_func = HybridSegmentationLoss(ce_weights=weights, lambda_ft=LAMBDA_FT, lambda_gd=LAMBDA_GD, lambda_ce=LAMBDA_CE).to(device)
    self.class_weights = weights
    self.optim = torch.optim.AdamW(self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE)
    if EMA:
      self.ema_model = torch.optim.swa_utils.AveragedModel(
        self.model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
      )

    self.start_epoch, self.step, self.vstep, self.min_epoch_vloss, self.stop_cnt, self.writer = self.load_checkpoint(self.checkpoint_path)
    self.start_epoch += 1 # resume from the next epoch

    if not self.writer:
      if not writer_path:
        experiment_folder = os.path.basename(os.path.dirname(model_path)) or "default"
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        today = str(datetime.now()).replace(" ", "_")
        auto_name = "-".join([experiment_folder, model_name, today, f"lr_{LR}", f"bs_{BATCH_SIZE}"])
        writer_path = str("runs/" + auto_name).replace(":", "_").replace(".", "_")
        self.writer = SummaryWriter(writer_path)
      else:
        self.writer = SummaryWriter(writer_path, purge_step=cast(int, None), max_queue=10, flush_secs=30)
      self.writer_path = writer_path
    print("[*] Tensorboard output path:", writer_path)

    self.train_metrics = { key: [] for key in self.metric_names }
    self.val_metrics = { key: [] for key in self.metric_names }

  def _unpack_batch(self, sample_batched):
    if len(sample_batched) == 2:
      image_batch, mask_batch = sample_batched
      return image_batch, mask_batch, None
    if len(sample_batched) == 3:
      image_batch, mask_batch, clf_target = sample_batched
      return image_batch, mask_batch, clf_target
    raise ValueError(f"Unsupported batch structure with {len(sample_batched)} items")

  def _unpack_output(self, out):
    if isinstance(out, dict):
      return out["seg"], out.get("clf")
    return out, None

  def _mean_metrics(self, metrics_dict):
    return {name: float(np.mean(values)) if len(values) > 0 else 0.0 for name, values in metrics_dict.items()}

  def _scalar_metrics(self, metrics: dict):
    return {name: value for name, value in metrics.items() if not isinstance(value, dict)}

  def save_onnx(self, example_input: torch.Tensor):
    self.onnx_path = self.model_path.split(".")[0] + ".onnx"
    model = self.model
    output_names = ["mask"]

    if getattr(self.model, "multitask", False):
      class OnnxWrapper(nn.Module):
        def __init__(self, wrapped_model):
          super().__init__()
          self.wrapped_model = wrapped_model

        def forward(self, x):
          out = self.wrapped_model(x)
          return out["seg"], out["clf"]

      model = OnnxWrapper(self.model)
      output_names = ["seg", "clf"]

    torch.onnx.export(
      model,
      example_input.to(self.device),
      self.onnx_path,
      export_params=True,
      opset_version=18,
      do_constant_folding=True,
      input_names=["image"],
      output_names=output_names,
    )
    print(f"[+] ONNX model saved at {self.onnx_path}.")
    return self.onnx_path

  def save_checkpoint(self, epoch, step, vstep, min_loss, stop_cnt, best=False):
    chpt_path = self.model_path.split(".")[0] + f"_best.pt" if best else self.model_path.split(".")[0] + ".pt"
    checkpoint = {
      "epoch": epoch,
      "step": step,
      "vstep": vstep,
      "min_loss": min_loss,
      "stop_cnt": stop_cnt,
      "model": self.ema_model.module.state_dict() if EMA else self.model.state_dict(),
      "writer": self.writer_path,
      "optimizer": self.optim.state_dict(),
      "scheduler": self.scheduler.state_dict() if self.scheduler else None,
      "config": json.load(open("configs/main.json", "r")),
    }
    torch.save(checkpoint, chpt_path)
    print(f"[+] Checkpoint saved at {chpt_path}.")
    if best:
      print(f"[+] New min eval loss {min_loss}")

  def load_checkpoint(self, chpt_path):
    if chpt_path is None or not os.path.exists(chpt_path):
      print(f"[!] No checkpoint found at {chpt_path}")
      return -1, 0, 0, float("inf"), 0, None

    checkpoint = torch.load(chpt_path, map_location=self.device, weights_only=False)

    if self.finetune:
      if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
      elif isinstance(checkpoint, dict):
        state_dict = checkpoint
      else:
        raise TypeError("Checkpoint must be a dict or a full trainer checkpoint")

      model_state = self.model.state_dict()
      filtered_state = {}
      skipped_shape_mismatch = []
      skipped_missing = []
      for key, value in state_dict.items():
        if key not in model_state:
          skipped_missing.append(key)
          continue
        if model_state[key].shape != value.shape:
          skipped_shape_mismatch.append((key, tuple(value.shape), tuple(model_state[key].shape)))
          continue
        filtered_state[key] = value

      load_result = self.model.load_state_dict(filtered_state, strict=False)
      print(f"[+] Loaded finetune checkpoint {chpt_path} (model weights only)")
      print(f"[*] Loaded {len(filtered_state)}/{len(state_dict)} compatible tensors")
      if skipped_shape_mismatch:
        print(f"[*] Skipped shape-mismatched keys: {skipped_shape_mismatch}")
      if skipped_missing:
        print(f"[*] Skipped missing keys: {skipped_missing}")
      if load_result.missing_keys:
        print(f"[*] Missing keys: {load_result.missing_keys}")
      if load_result.unexpected_keys:
        print(f"[*] Unexpected keys: {load_result.unexpected_keys}")
      return -1, 0, 0, float("inf"), 0, None

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
    writer_path = checkpoint.get("writer", None)
    writer = SummaryWriter(writer_path, purge_step=cast(int, None), max_queue=10, flush_secs=30) if writer_path else None
    self.writer_path = writer_path

    optim_state = checkpoint.get("optimizer", None)
    self.optim.load_state_dict(optim_state) if optim_state else None
    scheduler_state = checkpoint.get("scheduler", None)
    self.scheduler.load_state_dict(scheduler_state) if self.scheduler and scheduler_state else None

    print(f"[+] Resumed from checkpoint {chpt_path} (epoch {epoch})")
    return epoch, step, vstep, min_loss, stop_cnt, writer

  def log_scalars(
    self,
    tag_prefix: str,
    metrics: dict,
    step: int,
    accumulators: Optional[dict] = None
  ):
    for name, value in self._scalar_metrics(metrics).items():
      self.writer.add_scalar(f"{tag_prefix}/{name}", value, step)
      if accumulators is not None and name in accumulators:
        accumulators[name].append(value)

  def train_step(self, t, step, sample_batched, optim):
    image_batch, mask_batch, clf_target = self._unpack_batch(sample_batched)
    X = image_batch.to(self.device)
    Y = mask_batch.to(self.device)
    clf_target = clf_target.to(self.device).float() if clf_target is not None else None

    optim.zero_grad()

    out = self.model(X)
    seg_logits, clf_logits = self._unpack_output(out)

    seg_loss = self.loss_func(seg_logits, Y)
    loss = seg_loss
    metrics = compute_metrics(seg_logits.detach(), Y.detach(), weights=self.class_weights)

    current_metrics = {
      "loss": loss.item(),
      "seg_loss": seg_loss.item(),
      **metrics,
    }

    if clf_logits is not None and clf_target is not None:
      clf_loss = self.clf_loss_func(clf_logits, clf_target)
      loss = seg_loss + self.lambda_clf * clf_loss
      clf_metrics = compute_multilabel_metrics(clf_logits.detach(), clf_target.detach())
      current_metrics["loss"] = loss.item()
      current_metrics["clf_loss"] = clf_loss.item()
      current_metrics.update(clf_metrics)

    loss.backward()
    optim.step()
    # if self.scheduler: self.scheduler.step()
    if EMA: self.ema_model.update_parameters(self.model)

    self.log_scalars(
      "running train",
      current_metrics,
      step,
      self.epoch_train_metrics
    )
    scalar_metrics = self._scalar_metrics(current_metrics)
    t.set_description("[train] " + " | ".join(f"{name}: {value:.4f}" for name, value in scalar_metrics.items()))

  def train(self):
    try:
      min_epoch_vloss = self.min_epoch_vloss
      step = self.step
      vstep = self.vstep
      stop_cnt = self.stop_cnt

      print("[*] Training...")
      for epoch in range(self.start_epoch, EPOCHS):
        self.epoch_train_metrics = {key: [] for key in self.metric_names}
        self.epoch_val_metrics = {key: [] for key in self.metric_names}

        self.model.train()
        print(f"\n[=>] Epoch {epoch+1}/{EPOCHS}")
        for i_batch, sample_batched in enumerate((t := tqdm(self.train_loader))):
          self.train_step(t, step, sample_batched, self.optim)
          step += 1

        avg_metrics = self._mean_metrics(self.epoch_train_metrics)
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
          self.save_onnx(example_input=sample_batched[0][:1])
        else:
          stop_cnt += 1
          if self.early_stopping and stop_cnt >= EARLY_STOP_EPOCHS:
            print(f"[!] Early stopping at epoch {epoch+1}/{EPOCHS}.")
            break
    except KeyboardInterrupt:
      print("[*] Training interrupted. Saving model...")

    print("[+] Training done")
    print("[*] Generating report ...")
    report_name = self.model_path.split('.')[0].replace("checkpoints/", '')
    generate_report.main(
      self.onnx_path,
      f"reports/{report_name}.json",
      dataset=self.train_loader.dataset.dataset if isinstance(self.train_loader.dataset, torch.utils.data.Subset) else self.train_loader.dataset
    )

  def eval_step(self, t, vstep, sample_batched):
    image_batch, mask_batch, clf_target = self._unpack_batch(sample_batched)
    X = image_batch.to(self.device)
    Y = mask_batch.to(self.device)
    clf_target = clf_target.to(self.device).float() if clf_target is not None else None
    out = self.ema_model(X) if EMA else self.model(X)
    seg_logits, clf_logits = self._unpack_output(out)

    seg_loss = self.loss_func(seg_logits, Y)
    loss = seg_loss
    metrics = compute_metrics(seg_logits.detach(), Y.detach(), weights=self.class_weights)

    current_metrics = {
      "loss": loss.item(),
      "seg_loss": seg_loss.item(),
      **metrics,
    }

    if clf_logits is not None and clf_target is not None:
      clf_loss = self.clf_loss_func(clf_logits, clf_target)
      loss = seg_loss + self.lambda_clf * clf_loss
      clf_metrics = compute_multilabel_metrics(clf_logits.detach(), clf_target.detach())
      current_metrics["loss"] = loss.item()
      current_metrics["clf_loss"] = clf_loss.item()
      current_metrics.update(clf_metrics)

    self.log_scalars(
      "running val",
      current_metrics,
      vstep,
      self.epoch_val_metrics
    )
    scalar_metrics = self._scalar_metrics(current_metrics)
    t.set_description("[val] " + " | ".join(f"{name}: {value:.4f}" for name, value in scalar_metrics.items()))

  def eval(self, vstep, epoch):
    with torch.no_grad():
      self.model.eval()
      for i_batch, sample_batched in enumerate((t := tqdm(self.val_loader))):
        self.eval_step(t, vstep, sample_batched)
        vstep += 1

      avg_metrics = self._mean_metrics(self.epoch_val_metrics)
      self.log_scalars("epoch validation", avg_metrics, epoch, self.val_metrics)
      print("[->] Epoch average validation metrics: " + " | ".join(
        f"{name}: {value:.4f}" for name, value in avg_metrics.items()
      ))

    return vstep, avg_metrics["loss"] if "loss" in avg_metrics else None
