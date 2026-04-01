#!/usr/bin/env python3
import torch
import torch.nn as nn
from enum import Enum
from dataclasses import dataclass

from config import *
from utils import *

# TODO: cleanup - add model config dataclass for easy scaling and parameter tuning + use variables instead of magic numbers


class DoubleConv(nn.Module):
  def __init__(self, in_ch, out_ch):
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(in_ch, out_ch, 3, padding=1),
      nn.BatchNorm2d(out_ch),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_ch, out_ch, 3, padding=1),
      nn.BatchNorm2d(out_ch),
      nn.ReLU(inplace=True),
    )

  def forward(self, x):
    return self.block(x)


class UNet(nn.Module):
  def __init__(self, in_channels=1, num_classes=len(RGB_COLORS), features=64):
    super().__init__()
    self.down = nn.ModuleList()
    self.pool = nn.MaxPool2d(2)

    ch = features
    for _ in range(3):
      self.down.append(DoubleConv(in_channels, ch))
      in_channels = ch
      ch *= 2

    self.bottleneck = DoubleConv(in_channels, ch)

    self.up = nn.ModuleList()
    for _ in range(3):
      ch //= 2
      self.up.append(nn.ConvTranspose2d(ch * 2, ch, 2, stride=2))
      self.up.append(DoubleConv(ch * 2, ch))

    self.final = nn.Conv2d(features, num_classes, 1)

  def forward(self, x):
    skips = []
    for block in self.down:
      x = block(x)
      skips.append(x)
      x = self.pool(x)

    x = self.bottleneck(x)

    for i in range(0, len(self.up), 2):
      x = self.up[i](x)
      skip = skips.pop()
      x = torch.cat([x, skip], dim=1)
      x = self.up[i + 1](x)

    return self.final(x)
