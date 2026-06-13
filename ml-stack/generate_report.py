#!/usr/bin/env python3
import os
import sys
import json
import time
import numpy as np
from tqdm import tqdm

import torch
import onnxruntime as ort
import onnx

from config import *
from utils import compute_metrics, compute_multilabel_metrics
from dataset import CTScanDataset

# EXAMPLE USAGE:
# ONNX_USE_CUDA=1 python generate_report.py checkpoints/<experiment>/<onnx_model> reports/<experiment>/<report_name>.json


def get_model_size(onnx_path):
  """Get ONNX model size in bytes and parameter count."""
  size_bytes = os.path.getsize(onnx_path)
  model = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

  param_count = 0
  for initializer in model.get_inputs():
    pass

  try:
    import onnx
    onnx_model = onnx.load(onnx_path)
    for init in onnx_model.graph.initializer:
      param_count += int(np.prod(init.dims))
  except Exception:
    param_count = None

  return size_bytes, param_count


def get_gflops(session, input_shape):
  """Estimate GFLOPs for a single forward pass. Returns None if unavailable."""
  try:
    from thop import profile
    import torch
    print("[!] Warning: GFLOPs estimation requires 'thop' package, skipping.")
    return None
  except ImportError:
    return None


def make_session(onnx_path, use_cuda=False):
  providers = ["CPUExecutionProvider"]
  if use_cuda:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

  session = ort.InferenceSession(onnx_path, providers=providers)
  return session


def predict_single(session, image_tensor, onnx_path=None, use_cuda=False):
  """Run inference on a single image tensor [1, C, H, W], return timing and raw outputs."""
  input_np = image_tensor.numpy().astype(np.float32)
  input_name = session.get_inputs()[0].name

  start = time.perf_counter()
  try:
    outputs = session.run(None, {input_name: input_np})
  except Exception as e:
    if use_cuda and onnx_path is not None:
      print(f"[!] CUDA inference failed, falling back to CPU: {e}")
      session = make_session(onnx_path, use_cuda=False)
      start = time.perf_counter()
      outputs = session.run(None, {input_name: input_np})
    else:
      raise
  elapsed = time.perf_counter() - start

  return elapsed, outputs


def _resolve_outputs(session, outputs):
  output_meta = session.get_outputs()
  output_map = {meta.name: out for meta, out in zip(output_meta, outputs)}

  seg_logits = output_map.get("seg")
  if seg_logits is None:
    seg_logits = output_map.get("mask")
  clf_logits = output_map.get("clf")

  if seg_logits is None:
    for out in outputs:
      if out.ndim == 4:
        seg_logits = out
        break

  if clf_logits is None:
    for out in outputs:
      if out.ndim == 2:
        clf_logits = out
        break

  return seg_logits, clf_logits


def main(onnx_path, report_path, dataset=None):
  if not os.path.exists(onnx_path):
    print(f"[!] ONNX model not found: {onnx_path}")
    sys.exit(1)

  os.makedirs(os.path.dirname(report_path) if os.path.dirname(report_path) else ".", exist_ok=True)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  use_cuda = os.getenv("ONNX_USE_CUDA", "0").lower() in ("1", "true", "yes")
  print(f"[+] Using device: {device}")

  print(f"[*] Loading ONNX model from {onnx_path}")
  session = make_session(onnx_path, use_cuda=use_cuda and torch.cuda.is_available())
  print(f"[*] ONNX providers: {session.get_providers()}")
  output_names = [meta.name for meta in session.get_outputs()]
  is_multitask = "clf" in output_names or len(output_names) > 1

  model_size_bytes, param_count = get_model_size(onnx_path)
  print(f"[*] Model size: {model_size_bytes / (1024*1024):.2f} MB")
  print(f"[*] Parameter count: {param_count:,}" if param_count else "[*] Parameter count: N/A")

  if dataset is None:
    print(f"[*] Loading dataset from {BASE_DATA_DIR}")
    dataset = CTScanDataset(BASE_DATA_DIR, clf=is_multitask)
  elif is_multitask and not getattr(dataset, "clf", False):
    print("[*] Rebuilding dataset with classification targets for multitask report")
    dataset = CTScanDataset(dataset.base_dir, clf=True)

  train_size = int(TRAIN_SIZE * len(dataset))
  val_size = len(dataset) - train_size
  train_indices = set(range(train_size))
  val_indices = set(range(train_size, len(dataset)))

  dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=0
  )

  predictions = []
  all_metrics = {
    "loss": [], "pixel_acc": [], "IoU": [], "Dice": [], "F1": [],
    "Hausdorff": [],
  }
  if is_multitask:
    all_metrics.update({
      "clf_loss": [],
      "clf_acc": [],
      "clf_exact_match": [],
      "clf_precision_micro": [],
      "clf_recall_micro": [],
      "clf_f1_micro": [],
      "clf_precision_macro": [],
      "clf_recall_macro": [],
      "clf_f1_macro": [],
    })

  loss_func = torch.nn.CrossEntropyLoss()
  clf_loss_func = torch.nn.BCEWithLogitsLoss()

  print("[*] Running predictions on full dataset...")
  for idx, sample_batched in enumerate(tqdm(dataloader, desc="[*] Predicting")):
    if len(sample_batched) == 2:
      image, mask = sample_batched
    elif len(sample_batched) == 3:
      image, mask, _ = sample_batched
    else:
      raise ValueError(f"Unsupported batch structure with {len(sample_batched)} items")
    image_path = dataset.images[idx]
    mask_path = dataset.masks[idx]

    elapsed, outputs = predict_single(session, image, onnx_path=onnx_path, use_cuda=use_cuda and torch.cuda.is_available())
    seg_logits_np, clf_logits_np = _resolve_outputs(session, outputs)

    logits = torch.from_numpy(seg_logits_np)
    mask_long = mask.long()

    metrics = compute_metrics(logits, mask_long, num_classes=dataset.num_classes)

    loss = loss_func(logits, mask_long).item()
    metrics["loss"] = loss

    if is_multitask and clf_logits_np is not None and len(sample_batched) == 3:
      clf_logits = torch.from_numpy(clf_logits_np)
      clf_target = sample_batched[2].float()
      if clf_target.ndim == 1:
        clf_target = clf_target.unsqueeze(0)
      clf_loss = clf_loss_func(clf_logits, clf_target).item()
      clf_metrics = compute_multilabel_metrics(clf_logits, clf_target)
      metrics["clf_loss"] = clf_loss
      metrics.update(clf_metrics)

    for k, v in metrics.items():
      if k in all_metrics:
        all_metrics[k].append(v)

    predictions.append({
      "index": idx,
      "image_path": image_path,
      "mask_path": mask_path,
      "split": "train" if idx in train_indices else "val",
      "inference_time_sec": elapsed,
      "metrics": metrics,
    })

  avg_metrics = {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in all_metrics.items()}

  report = {
    "model": {
      "onnx_path": onnx_path,
      "size_bytes": model_size_bytes,
      "param_count": param_count,
      "gflops": None,
      "multitask": is_multitask,
      "output_names": output_names,
    },
    "config": {
      "image_size": IMAGE_SIZE,
      "norm_mean": NORM_MEAN,
      "norm_std": NORM_STD,
      "train_size": TRAIN_SIZE,
      "val_size": VAL_SIZE,
    },
    "summary": {
      "total_samples": len(predictions),
      "train_samples": sum(1 for p in predictions if p["split"] == "train"),
      "val_samples": sum(1 for p in predictions if p["split"] == "val"),
      "avg_inference_time_sec": float(np.mean([p["inference_time_sec"] for p in predictions])),
      "avg_metrics": avg_metrics,
    },
    "predictions": predictions,
  }

  with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

  print(f"\n[+] Report saved to {report_path}")
  print("\n[*] Average metrics:")
  for k, v in avg_metrics.items():
    print(f"  {k}: {v:.4f}")
  print(f"[*] Avg inference time: {report['summary']['avg_inference_time_sec']*1000:.2f} ms")


if __name__ == "__main__":
  if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} <onnx_path> <output_report_path>")
    print(f"Example: {sys.argv[0]} checkpoints/unet_best.onnx reports/unet_report.json")
    sys.exit(1)

  onnx_path = sys.argv[1]
  report_path = sys.argv[2]
  main(onnx_path, report_path)
