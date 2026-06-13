import os
import sys
from pathlib import Path

import torch


ML_STACK_DIR = Path(__file__).resolve().parents[1]
if str(ML_STACK_DIR) not in sys.path:
  sys.path.insert(0, str(ML_STACK_DIR))
os.chdir(ML_STACK_DIR)

from config import IMAGE_SIZE
from models.unet import UNet
from models.presets import PRESETS


def test_unet_segmentation_only_backward_compatible():
  model = UNet(PRESETS["unet_base"].to_config())
  x = torch.randn(2, 1, IMAGE_SIZE, IMAGE_SIZE)

  out = model(x)

  assert isinstance(out, torch.Tensor)
  assert out.shape == (2, model.num_classes, IMAGE_SIZE, IMAGE_SIZE)


def test_unet_multitask_output_and_freezing():
  model = UNet(
    PRESETS["unet_base"].to_config(),
    multitask=True,
    freeze_decoder=True,
  )
  x = torch.randn(2, 1, IMAGE_SIZE, IMAGE_SIZE)

  out = model(x)

  assert isinstance(out, dict)
  assert set(out.keys()) == {"seg", "clf"}
  assert out["seg"].shape == (2, model.num_classes, IMAGE_SIZE, IMAGE_SIZE)
  assert out["clf"].shape == (2, model.num_labels)

  assert all(not p.requires_grad for p in model.up.parameters())
  assert all(not p.requires_grad for p in model.final.parameters())
  if model.attention is not None:
    assert all(not p.requires_grad for p in model.attention.parameters())

  seg_model = UNet(PRESETS["unet_base"].to_config())
  load_result = model.load_unet_weights(seg_model.state_dict())
  assert all(key.startswith("clf_head") for key in load_result.missing_keys)
