import torch.nn as nn
from .focal_tversky_loss import FocalTverskyLoss
from .generalized_dice_loss import GeneralizedDiceLoss

class HybridSegmentationLoss(nn.Module):
  def __init__(
    self,
    ce_weights=None,
    lambda_ft=0.5,
    lambda_gd=0.3,
    lambda_ce=0.2,
  ):
    super().__init__()

    self.ft_loss = FocalTverskyLoss()
    self.gd_loss = GeneralizedDiceLoss()
    self.ce_loss = nn.CrossEntropyLoss(weight=ce_weights)

    self.lambda_ft = lambda_ft
    self.lambda_gd = lambda_gd
    self.lambda_ce = lambda_ce

  def forward(self, logits, targets):
    ft = self.ft_loss(logits, targets)
    gd = self.gd_loss(logits, targets)
    ce = self.ce_loss(logits, targets)

    return (
      self.lambda_ft * ft
      + self.lambda_gd * gd
      + self.lambda_ce * ce
    )
