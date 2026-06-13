import torch
import numpy as np
from medpy.metric import binary
import matplotlib.pyplot as plt

# epidural hematoma (EDH)
# subdural hematoma (SDH)
# intraventricular hemorrhage (IVH)
# subarachnoid hemorrhage (SAH)
# intraparenchymal hemorrhage (IPH)
CLASS_NAMES = [
  "BACKGROUND",
  "BONE",
  "CONTUSSION",
  "EDH",
  "IVH",
  "SAH",
  "SDH",
]

RGB_COLORS = {
    0: (0, 0, 0),          # background
    1: (128, 0, 0),        # category_id 0
    2: (0, 128, 0),        # category_id 1
    3: (128, 128, 0),      # category_id 2
    4: (0, 0, 128),        # category_id 3
    5: (255, 0, 255),      # category_id 4
    6: (250, 136, 5)       # category_id 5
}

def compute_metrics(pred, target, num_classes=len(RGB_COLORS), weights=None):
  """
  pred: (N, C, H, W) logits
  target: (N, H, W) with class indices
  weights: optional (C,) tensor for weighted aggregation
  """

  metrics = {}
  pred_classes = pred.argmax(dim=1)  # (N, H, W)

  # Pixel accuracy
  correct = (pred_classes == target).float().sum()
  total = torch.numel(target)
  metrics["pixel_acc"] = (correct / total).item()

  per_class = {"IoU": [], "Dice": [], "F1": [], "Hausdorff": [], "counts": []}
  for cls in range(num_classes):
    pred_c = (pred_classes == cls).cpu().numpy()
    target_c = (target == cls).cpu().numpy()

    target_count = target_c.sum()
    pred_count = pred_c.sum()

    # skip absent classes
    if target_count == 0 and pred_count == 0:
      continue

    # IoU
    intersection = (pred_c & target_c).sum()
    union = (pred_c | target_c).sum()
    iou = intersection / union if union > 0 else 0

    # Dice (DSC)
    dice = 2 * intersection / (pred_count + target_count + 1e-8)

    # Hausdorff Distance (use medpy)
    hd = None
    try:
      hd = binary.hd(pred_c, target_c)
    except Exception:
      pass

    per_class["IoU"].append(iou)
    per_class["Dice"].append(dice)
    per_class["F1"].append(dice)
    per_class["Hausdorff"].append(hd if hd is not None else 0.0)
    per_class["counts"].append(target_count)

  counts = torch.tensor(per_class["counts"], dtype=torch.float32)
  if weights is not None:
    w = weights[:len(counts)].float()
  else:
    w = counts

  w_sum = w.sum()
  if w_sum > 0:
    w_norm = w / w_sum
  else:
    w_norm = torch.ones(len(counts)) / len(counts) if len(counts) > 0 else torch.tensor([1.0])

  metrics["IoU"] = sum(per_class["IoU"]) / len(per_class["IoU"]) if per_class["IoU"] else 0
  metrics["Dice"] = sum(per_class["Dice"]) / len(per_class["Dice"]) if per_class["Dice"] else 0
  metrics["F1"] = sum(per_class["F1"]) / len(per_class["F1"]) if per_class["F1"] else 0
  metrics["Hausdorff"] = sum(per_class["Hausdorff"]) / len(per_class["Hausdorff"]) if per_class["Hausdorff"] else 0

  if len(per_class["IoU"]) > 0:
    metrics["w_IoU"] = sum(w_norm[i].item() * v for i, v in enumerate(per_class["IoU"]))
    metrics["w_Dice"] = sum(w_norm[i].item() * v for i, v in enumerate(per_class["Dice"]))
    metrics["w_F1"] = sum(w_norm[i].item() * v for i, v in enumerate(per_class["F1"]))
    metrics["w_Hausdorff"] = sum(w_norm[i].item() * v for i, v in enumerate(per_class["Hausdorff"]))
  else:
    metrics["w_IoU"] = metrics["w_Dice"] = metrics["w_F1"] = metrics["w_Hausdorff"] = 0

  metrics["per_class"] = {}
  for cls in range(num_classes):
    cls_metrics = {}
    if cls < len(per_class["IoU"]):
      cls_metrics["IoU"] = float(per_class["IoU"][cls])
      cls_metrics["Dice"] = float(per_class["Dice"][cls])
      cls_metrics["F1"] = float(per_class["F1"][cls])
      cls_metrics["Hausdorff"] = float(per_class["Hausdorff"][cls])
      cls_metrics["count"] = int(per_class["counts"][cls])
    else:
      cls_metrics["IoU"] = 0.0
      cls_metrics["Dice"] = 0.0
      cls_metrics["F1"] = 0.0
      cls_metrics["Hausdorff"] = 0.0
      cls_metrics["count"] = 0
    metrics["per_class"][str(cls)] = cls_metrics

  return metrics


def compute_multilabel_metrics(logits, target, threshold=0.5, ignore_background=True):
  """
  logits: (N, C) raw classification logits
  target: (N, C) binary multi-hot labels
  """
  pred = (torch.sigmoid(logits) >= threshold).to(torch.int64)
  target = target.to(torch.int64)

  if ignore_background and pred.ndim == 2 and pred.shape[1] > 0:
    pred = pred[:, 1:]
    target = target[:, 1:]

  pred_np = pred.detach().cpu().numpy()
  target_np = target.detach().cpu().numpy()

  metrics = {}
  if pred_np.size == 0:
    return {
      "clf_acc": 0.0,
      "clf_exact_match": 0.0,
      "clf_precision_micro": 0.0,
      "clf_recall_micro": 0.0,
      "clf_f1_micro": 0.0,
      "clf_precision_macro": 0.0,
      "clf_recall_macro": 0.0,
      "clf_f1_macro": 0.0,
    }

  exact_match = float(np.all(pred_np == target_np, axis=1).mean()) if len(pred_np) > 0 else 0.0
  acc = float((pred_np == target_np).mean())

  tp = np.logical_and(pred_np == 1, target_np == 1).sum(axis=0).astype(np.float32)
  fp = np.logical_and(pred_np == 1, target_np == 0).sum(axis=0).astype(np.float32)
  fn = np.logical_and(pred_np == 0, target_np == 1).sum(axis=0).astype(np.float32)

  eps = 1e-8
  precision_per_class = tp / (tp + fp + eps)
  recall_per_class = tp / (tp + fn + eps)
  f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + eps)

  tp_sum = float(tp.sum())
  fp_sum = float(fp.sum())
  fn_sum = float(fn.sum())
  precision_micro = tp_sum / (tp_sum + fp_sum + eps)
  recall_micro = tp_sum / (tp_sum + fn_sum + eps)
  f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro + eps)

  metrics["clf_acc"] = acc
  metrics["clf_exact_match"] = exact_match
  metrics["clf_precision_micro"] = float(precision_micro)
  metrics["clf_recall_micro"] = float(recall_micro)
  metrics["clf_f1_micro"] = float(f1_micro)
  metrics["clf_precision_macro"] = float(precision_per_class.mean()) if len(precision_per_class) else 0.0
  metrics["clf_recall_macro"] = float(recall_per_class.mean()) if len(recall_per_class) else 0.0
  metrics["clf_f1_macro"] = float(f1_per_class.mean()) if len(f1_per_class) else 0.0
  return metrics


def visualize_ct_with_mask(image, mask, alpha=0.5):
  """
  image: torch.Tensor [1,H,W] or numpy [H,W], grayscale CT scan
  mask: torch.Tensor [H,W], integer class IDs
  alpha: float, transparency of mask overlay
  """
  # convert tensors to numpy
  if hasattr(image, "detach"):
      image = image.detach().cpu().numpy()
  if hasattr(mask, "detach"):
      mask = mask.detach().cpu().numpy()

  # squeeze channel if needed
  if image.ndim == 3 and image.shape[0] == 1:
      image = image[0]

  # normalize grayscale image for display
  img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)

  # create RGB overlay for mask
  h, w = mask.shape
  mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
  for cat_id, color in RGB_COLORS.items():
      mask_rgb[mask == cat_id] = color

  # blend image and mask
  img_rgb = np.stack([img_norm]*3, axis=-1) * 255
  overlay = ((1 - alpha) * img_rgb + alpha * mask_rgb).astype(np.uint8)

  # plot
  plt.figure(figsize=(6, 6))
  plt.imshow(overlay)
  plt.axis("off")
  plt.show()
