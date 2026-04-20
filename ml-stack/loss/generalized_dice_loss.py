import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_background=True):
      super().__init__()
      self.smooth = smooth
      self.ignore_background = ignore_background

    def forward(self, logits, targets):
      num_classes = logits.shape[1]

      probs = torch.softmax(logits, dim=1)
      targets_one_hot = F.one_hot(targets, num_classes=num_classes)
      targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

      if self.ignore_background:
        probs = probs[:, 1:]
        targets_one_hot = targets_one_hot[:, 1:]

      dims = (0, 2, 3)
      class_volumes = targets_one_hot.sum(dim=dims)
      weights = 1.0 / (class_volumes ** 2 + self.smooth)

      intersection = (probs * targets_one_hot).sum(dim=dims)
      union = (probs + targets_one_hot).sum(dim=dims)
      dice = (2 * (weights * intersection).sum() + self.smooth) / ((weights * union).sum() + self.smooth)
      return 1 - dice
