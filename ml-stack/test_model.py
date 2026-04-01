import torch

from config import *
from utils import RGB_COLORS
from models.unet import UNet

if __name__ == "__main__":
  model = UNet(in_channels=1, num_classes=len(RGB_COLORS))
  x = torch.randn(2, 1, *IMAGE_SIZE)
  out = model(x)
  print(f"Input: {x.shape} -> Output: {out.shape}")
