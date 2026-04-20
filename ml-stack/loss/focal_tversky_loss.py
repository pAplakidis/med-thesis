import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
  def __init__(self, alpha=0.3, beta=0.7, gamma=1.5, smooth=1e-6, ignore_background=True):
    super().__init__()
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.smooth = smooth
    self.ignore_background = ignore_background

  def forward(self, logits, targets):
    """
    logits: [B, C, H, W]
    targets: [B, H, W]
    """
    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)
    targets_one_hot = F.one_hot(targets, num_classes=num_classes)
    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

    if self.ignore_background:
      probs = probs[:, 1:]
      targets_one_hot = targets_one_hot[:, 1:]

    dims = (0, 2, 3)
    tp = (probs * targets_one_hot).sum(dim=dims)
    fp = (probs * (1 - targets_one_hot)).sum(dim=dims)
    fn = ((1 - probs) * targets_one_hot).sum(dim=dims)

    tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
    focal_tversky = (1 - tversky) ** self.gamma
    valid_classes = (targets_one_hot.sum(dim=dims) > 0).float()

    return (focal_tversky * valid_classes).sum() / (valid_classes.sum() + self.smooth)

