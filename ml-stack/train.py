#!/usr/bin/env python3
import os
import psutil
import torch
from torch.utils.data import DataLoader, random_split
# from sklearn.model_selection import StratifiedShuffleSplit

from config import *
from dataset import *
from model import *
from trainer import Trainer

# EXAMPLE USAGE: MODEL_PATH=checkpoints/CILv3D.pt CHECKPOINT=checkpoints/CILv3D_best.py ./train.py

MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/unet.pt")
CHECKPOINT = os.getenv("CHECKPOINT", None)
WRITER_PATH = os.getenv("WRITER_PATH", None)

N_WORKERS = psutil.cpu_count(logical=False)
PREFETCH_FACTOR = psutil.cpu_count(logical=False) // 2
PIN_MEMORY = not EMA

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
torch.set_warn_always(False)


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("[+] Using device:", device)

  os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
  print("\n[*] Configuration:")
  print(f"Model path: {MODEL_PATH}")
  print(f"Checkpoint path: {CHECKPOINT}")
  print(f"Epochs: {EPOCHS} - Batch size: {BATCH_SIZE} - Learning rate: {LR} - Weight decay: {WEIGHT_DECAY}")
  print(f"Number of workers: {N_WORKERS} - Prefetch factor: {PREFETCH_FACTOR}")
  print(f"EMA: {EMA} - Pin memory: {PIN_MEMORY}")
  print()

  dataset = CTScanDataset(BASE_DATA_DIR)
  train_size = int(TRAIN_SIZE * len(dataset))
  val_size = len(dataset) - train_size
  # sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=42)
  # train_idx, val_idx = next(sss.split(np.zeros(dataset.num_classes), dataset.classes))
  train_set, val_set = random_split(dataset, [train_size, val_size])  # FIXME: no class balance

  train_loader =  DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    prefetch_factor=PREFETCH_FACTOR,
    num_workers=N_WORKERS,
    pin_memory=PIN_MEMORY
  )
  val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    prefetch_factor=PREFETCH_FACTOR,
    num_workers=N_WORKERS,
    pin_memory=PIN_MEMORY
  )

  model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                          in_channels=1, out_channels=dataset.num_classes, init_features=32, pretrained=False)
  model.to(device)

  trainer = Trainer(
    device, model, MODEL_PATH, train_loader, val_loader,
    checkpoint_path=CHECKPOINT, writer_path=WRITER_PATH, eval_epoch=True,
    save_checkpoints=True, early_stopping=True
  )
  trainer.train()
