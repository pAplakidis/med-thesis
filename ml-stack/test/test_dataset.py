import os
import sys
from pathlib import Path

import torch


ML_STACK_DIR = Path(__file__).resolve().parents[1]
if str(ML_STACK_DIR) not in sys.path:
  sys.path.insert(0, str(ML_STACK_DIR))
os.chdir(ML_STACK_DIR)

from config import BASE_DATA_DIR, IMAGE_SIZE
from dataset import CTScanDataset
from utils import RGB_COLORS


def test_ct_scan_dataset_sanity():
  dataset = CTScanDataset(BASE_DATA_DIR)

  assert len(dataset) > 0
  assert len(dataset.images) == len(dataset.masks) == len(dataset)
  assert dataset.num_classes == len(RGB_COLORS)
  assert dataset.get_classes() == list(RGB_COLORS.keys())

  image, mask = dataset[0]
  assert isinstance(image, torch.Tensor)
  assert isinstance(mask, torch.Tensor)
  assert image.shape == (1, IMAGE_SIZE, IMAGE_SIZE)
  assert mask.shape == (IMAGE_SIZE, IMAGE_SIZE)
  assert mask.dtype == torch.long

  class_balance = dataset.get_class_balance()
  assert set(class_balance.keys()) == set(range(len(RGB_COLORS)))
  assert all(count >= 0 for count in class_balance.values())
  assert sum(class_balance.values()) >= len(dataset)


def test_ct_scan_dataset_multilabel_targets():
  dataset = CTScanDataset(BASE_DATA_DIR, clf=True)

  image, mask, clf_target = dataset[0]
  assert isinstance(image, torch.Tensor)
  assert isinstance(mask, torch.Tensor)
  assert isinstance(clf_target, torch.Tensor)
  assert clf_target.shape == (len(RGB_COLORS),)
  assert clf_target.dtype == torch.float32
  assert clf_target[0].item() == 1.0
  assert torch.all((clf_target == 0) | (clf_target == 1))
