#!/usr/bin/env python3
import os
import sys
import torch

from models.unet import UNet
from config import IMAGE_SIZE
from utils import RGB_COLORS


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <model_path>")
    sys.exit(1)

  model_path = sys.argv[1]
  onnx_path = os.path.splitext(model_path)[0] + ".onnx"

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"[*] Loading model from {model_path}")

  checkpoint = torch.load(model_path, map_location=device, weights_only=False)
  state_dict = checkpoint["model"] if isinstance(checkpoint, dict) else checkpoint

  # TODO: use model abstraction layer to select which model (or do it by hand)
  # model = UNet(in_channels=1, num_classes=len(RGB_COLORS))
  model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                          in_channels=1, out_channels=len(RGB_COLORS), init_features=32, pretrained=False)

  model.load_state_dict(state_dict)
  model.to(device)
  model.eval()

  h, w = IMAGE_SIZE
  dummy_input = torch.randn(1, 1, h, w).to(device)

  print(f"[*] Exporting to ONNX with input size {IMAGE_SIZE}")
  torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=["image"],
    output_names=["mask"],
    dynamic_axes={
      "image": {0: "batch_size"},
      "mask": {0: "batch_size"},
    },
  )
  print(f"[+] ONNX model saved at {onnx_path}.")
