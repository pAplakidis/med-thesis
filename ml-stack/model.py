#!/usr/bin/env python3
import torch
import torch.nn as nn
from enum import Enum
from dataclasses import dataclass

from config import *
from utils import *

class UNet(nn.Module):
  pass


if __name__ == "__main__":
  model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                          in_channels=3, out_channels=1, init_features=32, pretrained=True)
  print(model)
