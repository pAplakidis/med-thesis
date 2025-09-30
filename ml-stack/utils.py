import torch
import numpy as np
from medpy.metric import binary
import matplotlib.pyplot as plt

rgb_colors = {
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

def compute_metrics(pred, target, num_classes=len(rgb_colors)):
  """
  pred: (N, C, H, W) logits
  target: (N, H, W) with class indices
  """

  metrics = {}
  pred_classes = pred.argmax(dim=1)  # (N, H, W)

  # Pixel accuracy
  correct = (pred_classes == target).float().sum()
  total = torch.numel(target)
  metrics["pixel_acc"] = (correct / total).item()

  ious, dices, f1s, hausdorffs = [], [], [], []
  for cls in range(num_classes):
    pred_c = (pred_classes == cls).cpu().numpy()
    target_c = (target == cls).cpu().numpy()

    # skip absent classes
    if target_c.sum() == 0 and pred_c.sum() == 0:
      continue

    # IoU
    intersection = (pred_c & target_c).sum()
    union = (pred_c | target_c).sum()
    iou = intersection / union if union > 0 else 0
    ious.append(iou)

    # Dice (DSC)
    dice = 2 * intersection / (pred_c.sum() + target_c.sum() + 1e-8)
    dices.append(dice)

    # F1 (same as Dice for segmentation, but often separated in reports)
    f1s.append(dice)

    # Hausdorff Distance (use medpy)
    try:
      hd = binary.hd(pred_c, target_c)
      hausdorffs.append(hd)
    except Exception:
      # Hausdorff can fail if one set is empty
        pass

  # Aggregate (mean per class)
  metrics["IoU"] = sum(ious) / len(ious) if ious else 0
  metrics["Dice"] = sum(dices) / len(dices) if dices else 0
  metrics["F1"] = sum(f1s) / len(f1s) if f1s else 0
  metrics["Hausdorff"] = sum(hausdorffs) / len(hausdorffs) if hausdorffs else 0
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
  for cat_id, color in rgb_colors.items():
      mask_rgb[mask == cat_id] = color

  # blend image and mask
  img_rgb = np.stack([img_norm]*3, axis=-1) * 255
  overlay = ((1 - alpha) * img_rgb + alpha * mask_rgb).astype(np.uint8)

  # plot
  plt.figure(figsize=(6, 6))
  plt.imshow(overlay)
  plt.axis("off")
  plt.show()
