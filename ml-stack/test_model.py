import torch

from config import *
from models.unet import UNet, PRESETS

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("[+] Using device:", device)

  # model = UNet(in_channels=1, num_classes=len(RGB_COLORS))
  model = UNet(PRESETS["unet_xl"].to_config()).to(device)
  model.eval()

  x = torch.randn(1, 1, * (IMAGE_SIZE, IMAGE_SIZE), device=device)
  with torch.no_grad():
    out = model(x)

  print(f"Input: {x.shape} -> Output: {out.shape}")
