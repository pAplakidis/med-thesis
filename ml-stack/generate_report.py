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
from utils import compute_metrics, RGB_COLORS
from dataset import CTScanDataset


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


def predict_single(session, image_tensor):
  """Run inference on a single image tensor [1, C, H, W], return prediction and timing."""
  input_np = image_tensor.numpy().astype(np.float32)
  input_name = session.get_inputs()[0].name

  start = time.perf_counter()
  outputs = session.run(None, {input_name: input_np})
  elapsed = time.perf_counter() - start

  pred = np.argmax(outputs[0], axis=1).squeeze(0)
  return pred, elapsed, outputs[0]


def main(onnx_path, report_path, dataset=None):
  if not os.path.exists(onnx_path):
    print(f"[!] ONNX model not found: {onnx_path}")
    sys.exit(1)

  os.makedirs(os.path.dirname(report_path) if os.path.dirname(report_path) else ".", exist_ok=True)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
  print(f"[+] Using device: {device}")

  print(f"[*] Loading ONNX model from {onnx_path}")
  session = ort.InferenceSession(onnx_path, providers=providers)

  model_size_bytes, param_count = get_model_size(onnx_path)
  print(f"[*] Model size: {model_size_bytes / (1024*1024):.2f} MB")
  print(f"[*] Parameter count: {param_count:,}" if param_count else "[*] Parameter count: N/A")

  if dataset is None:
    print(f"[*] Loading dataset from {BASE_DATA_DIR}")
    dataset = CTScanDataset(BASE_DATA_DIR)

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
    "Hausdorff": [], "w_IoU": [], "w_Dice": [], "w_F1": [], "w_Hausdorff": [],
  }

  loss_func = torch.nn.CrossEntropyLoss()

  print("[*] Running predictions on full dataset...")
  for idx, (image, mask) in enumerate(tqdm(dataloader, desc="[*] Predicting")):
    image_path = dataset.images[idx]
    mask_path = dataset.masks[idx]

    pred, elapsed, logits_np = predict_single(session, image)

    logits = torch.from_numpy(logits_np)
    mask_long = mask.long()

    metrics = compute_metrics(logits, mask_long, num_classes=dataset.num_classes)

    loss = loss_func(logits, mask_long).item()
    metrics["loss"] = loss

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
