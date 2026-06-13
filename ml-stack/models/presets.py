from dataclasses import dataclass

from utils import RGB_COLORS


@dataclass
class Preset:
  name: str
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

  def to_config(self):
    from models.unet import UnetConfig

    return UnetConfig(
      in_channels=self.in_channels,
      num_classes=self.num_classes,
      base_features=self.base_features,
      depth=self.depth,
      convs_per_block=self.convs_per_block,
      use_residual=self.use_residual,
      use_attention=self.use_attention,
      dropout=self.dropout,
      use_batch_norm=self.use_batch_norm,
      activation=self.activation,
    )


PRESETS = {
  "unet_tiny": Preset(
    name="unet_tiny",
    base_features=16,
    depth=3,
    convs_per_block=2,
  ),
  "unet_small": Preset(
    name="unet_small",
    base_features=32,
    depth=4,
    convs_per_block=2,
  ),
  "unet_base": Preset(
    name="unet_base",
    base_features=64,
    depth=4,
    convs_per_block=2,
  ),
  "unet_large": Preset(
    name="unet_large",
    base_features=128,
    depth=5,
    convs_per_block=3,
  ),
  "unet_large_plus": Preset(
    name="unet_large_plus",
    base_features=144,
    depth=5,
    convs_per_block=3,
  ),
  "unet_xl": Preset(
    name="unet_xl_attn_res",
    base_features=256,
    depth=5,
    convs_per_block=3,
    use_residual=True,
    use_attention=True,
    dropout=0.1,
  ),
}
