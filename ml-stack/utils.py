import torch
import numpy as np
from medpy.metric import binary
import matplotlib.pyplot as plt

RGB_COLORS = {
    0: (0, 0, 0),          # background
    1: (128, 0, 0),        # category_id 0
    2: (0, 128, 0),        # category_id 1
    3: (128, 128, 0),      # category_id 2
    4: (0, 0, 128),        # category_id 3
    5: (255, 0, 255),      # category_id 4
    6: (255, 0, 0),        # category_id 5
    7: (0, 255, 0),        # category_id 6
    8: (0, 0, 255),        # category_id 7
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
