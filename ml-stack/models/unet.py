#!/usr/bin/env python3
import os
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn

from config import *
from utils import *
from models.presets import PRESETS, Preset

@dataclass
class UnetConfig:
  in_channels: int = 1
  num_classes: int = len(RGB_COLORS)
  base_features: int = 64

  depth: int = 4
  convs_per_block: int = 2

  use_residual: bool = False
  use_attention: bool = False
  dropout: float = 0.0
  use_batch_norm: bool = True
  activation: str = "relu"


class ConvBlock(nn.Module):
  def __init__(
    self,
    in_ch,
    out_ch,
    convs=2,
    use_residual=False,
    use_batch_norm=True,
    activation="relu",
    dropout=0.0,
  ):
    super().__init__()
    layers = []
    for i in range(convs):
      layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1))
      if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_ch))
      if activation == "relu":
        layers.append(nn.ReLU(inplace=True))
      elif activation == "leaky_relu":
        layers.append(nn.LeakyReLU(0.2, inplace=True))
      elif activation == "gelu":
        layers.append(nn.GELU())
    if use_residual:
      self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
      self.act = nn.ReLU(inplace=True) if activation == "relu" else nn.Identity()
    if dropout > 0:
      layers.append(nn.Dropout2d(dropout))
    self.block = nn.Sequential(*layers)

  def forward(self, x):
    return self.block(x)


class ResidualBlock(nn.Module):
  def __init__(self, channels, use_batch_norm=True, activation="relu"):
    super().__init__()
    self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
    self.bn1 = nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()
    self.act1 = nn.ReLU(inplace=True) if activation == "relu" else nn.GELU()
    self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
    self.bn2 = nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()

  def forward(self, x):
    residual = x
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = x + residual
    return self.act1(x)


class AttentionGate(nn.Module):
  def __init__(self, gate_ch, skip_ch, int_ch):
    super().__init__()
    self.W_g = nn.Sequential(
      nn.Conv2d(gate_ch, int_ch, 1),
      nn.BatchNorm2d(int_ch),
    )
    self.W_x = nn.Sequential(
      nn.Conv2d(skip_ch, int_ch, 1),
      nn.BatchNorm2d(int_ch),
    )
    self.psi = nn.Sequential(
      nn.Conv2d(int_ch, 1, 1),
      nn.BatchNorm2d(1),
      nn.Sigmoid(),
    )
    self.relu = nn.ReLU(inplace=True)

  def forward(self, g, x):
    skip = x
    g = self.W_g(g)
    x = self.W_x(x)
    psi = self.relu(g + x)
    psi = self.psi(psi)
    return skip * psi


class DoubleConv(nn.Module):
  def __init__(
    self,
    in_ch,
    out_ch,
    use_residual=False,
    use_batch_norm=True,
    activation="relu",
    dropout=0.0,
  ):
    super().__init__()
    if use_residual:
      self.block = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch) if use_batch_norm else nn.Identity(),
        nn.ReLU(inplace=True),
        ResidualBlock(out_ch, use_batch_norm, activation),
      )
    else:
      self.block = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch) if use_batch_norm else nn.Identity(),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch) if use_batch_norm else nn.Identity(),
        nn.ReLU(inplace=True),
      )
    if dropout > 0:
      self.block = nn.Sequential(
        self.block,
        nn.Dropout2d(dropout),
      )

  def forward(self, x):
    return self.block(x)


class UNet(nn.Module):
  def __init__(
    self,
    config: Optional[UnetConfig] = None,
    multitask: bool = False,
    num_labels: Optional[int] = None,
    clf_dropout: float = 0.0,
    freeze_encoder: bool = False,
    freeze_decoder: bool = False,
    **kwargs,
  ):
    super().__init__()
    if config is None:
      config = UnetConfig(**kwargs)

    self.config = config
    self.multitask = multitask
    self.in_channels = config.in_channels
    self.num_classes = config.num_classes
    self.num_labels = num_labels if num_labels is not None else self.num_classes
    self.base_features = config.base_features
    self.depth = config.depth
    self.convs_per_block = config.convs_per_block

    features = self.base_features
    self.down = nn.ModuleList()
    self.pool = nn.MaxPool2d(2)
    self.bottleneck_channels = None

    in_ch = self.in_channels
    for _ in range(self.depth):
      self.down.append(
        DoubleConv(
          in_ch,
          features,
          use_residual=config.use_residual,
          use_batch_norm=config.use_batch_norm,
          activation=config.activation,
          dropout=config.dropout,
        )
      )
      in_ch = features
      features *= 2

    self.bottleneck_channels = features
    self.bottleneck = DoubleConv(
      in_ch,
      features,
      use_residual=config.use_residual,
      use_batch_norm=config.use_batch_norm,
      activation=config.activation,
      dropout=config.dropout,
    )

    self.up = nn.ModuleList()
    self.attention = nn.ModuleList() if config.use_attention else None
    for _ in range(self.depth):
      features //= 2
      self.up.append(nn.ConvTranspose2d(features * 2, features, 2, stride=2))
      if self.attention is not None:
        self.attention.append(AttentionGate(features, features, features // 2))
      self.up.append(
        DoubleConv(
          features * 2,
          features,
          use_residual=config.use_residual,
          use_batch_norm=config.use_batch_norm,
          activation=config.activation,
          dropout=config.dropout,
        )
      )

    self.final = nn.Conv2d(self.base_features, self.num_classes, 1)
    self.clf_head = None
    if self.multitask:
      clf_layers = [
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
      ]
      if clf_dropout > 0:
        clf_layers.append(nn.Dropout(clf_dropout))
      clf_layers.append(nn.Linear(self.bottleneck_channels, self.num_labels))
      self.clf_head = nn.Sequential(*clf_layers)

    if freeze_encoder:
      self.set_encoder_trainable(False)
    if freeze_decoder:
      self.set_decoder_trainable(False)

  def set_encoder_trainable(self, trainable: bool):
    for module in [self.down, self.bottleneck]:
      for param in module.parameters():
        param.requires_grad = trainable

  def set_decoder_trainable(self, trainable: bool):
    modules = [self.up, self.final]
    if self.attention is not None:
      modules.append(self.attention)
    for module in modules:
      for param in module.parameters():
        param.requires_grad = trainable

  def load_unet_weights(self, checkpoint: Union[str, os.PathLike, dict], strict: bool = False, map_location: str = "cpu"):
    if isinstance(checkpoint, (str, os.PathLike)):
      checkpoint = torch.load(checkpoint, map_location=map_location)
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
      state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict):
      state_dict = checkpoint
    else:
      raise TypeError("checkpoint must be a path or state dict")

    return self.load_state_dict(state_dict, strict=strict)

  def encode(self, x):
    skips = []
    for block in self.down:
      x = block(x)
      skips.append(x)
      x = self.pool(x)

    x = self.bottleneck(x)
    return x, skips

  def decode(self, x, skips):
    for i in range(0, len(self.up), 2):
      x = self.up[i](x)
      skip = skips.pop()
      if self.attention is not None:
        attn_idx = i // 2
        skip = self.attention[attn_idx](x, skip)
      x = torch.cat([x, skip], dim=1)
      x = self.up[i + 1](x)

    return self.final(x)

  def forward(self, x):
    x, skips = self.encode(x)
    seg = self.decode(x, skips)
    if not self.multitask:
      return seg
    clf = self.clf_head(x)
    return {
      "seg": seg,
      "clf": clf,
    }


def count_parameters(model: nn.Module) -> int:
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
  for name, preset in PRESETS.items():
    config = preset.to_config()
    model = UNet(config)
    params = count_parameters(model)
    print(f"{name}: {params:,} params, {params/1e6:.2f}M")
