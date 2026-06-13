import torch

from config import *
from models.unet import UNet, PRESETS


def test_unet_segmentation_only_backward_compatible():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = UNet(PRESETS["unet_base"].to_config()).to(device)
  model.eval()

  x = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)
  with torch.no_grad():
    out = model(x)

  assert isinstance(out, torch.Tensor)
  assert out.shape == (1, model.num_classes, IMAGE_SIZE, IMAGE_SIZE)
