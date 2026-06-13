#!/usr/bin/env python3
import os
import psutil
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from config import *
from dataset import *
from trainer import Trainer
from models.unet import UNet, PRESETS

# EXAMPLE USAGE:
# MODEL_PATH=checkpoints/model.pt CHECKPOINT=checkpoints/model.pt ./train.py
# for finetuning a multitask model:
# MULTITASK=1 FINETUNE=1 CHECKPOINT=checkpoints/equal_hybrid_seg_loss/unet_base_best.pt MODEL_PATH=checkpoints/multitask/unet_base_mt.pt ./train.py 

MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/unet.pt")
CHECKPOINT = os.getenv("CHECKPOINT", None)
WRITER_PATH = os.getenv("WRITER_PATH", None)
MULTITASK = os.getenv("MULTITASK", "0").lower() in ("1", "true", "yes")
FINETUNE = os.getenv("FINETUNE", "0").lower() in ("1", "true", "yes")

N_WORKERS = psutil.cpu_count(logical=False)
PREFETCH_FACTOR = psutil.cpu_count(logical=False) // 2
PIN_MEMORY = not EMA
EARLY_STOP = False

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
torch.set_warn_always(False)


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("[+] Using device:", device)

  os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
  print("\n[*] Train configuration:")
  print(f"Model path: {MODEL_PATH}")
  print(f"Multitask: {MULTITASK}")
  print(f"Checkpoint path: {CHECKPOINT}")
  print(f"Finetune: {FINETUNE}")
  print(f"Number of workers: {N_WORKERS} - Prefetch factor: {PREFETCH_FACTOR}")
  print(f"EMA: {EMA} - Pin memory: {PIN_MEMORY}")
  print()

  dataset = CTScanDataset(BASE_DATA_DIR, clf=MULTITASK)
  labels = dataset.clf_targets if MULTITASK and dataset.clf_targets is not None else dataset.get_multilabel_targets()
  msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=VAL_SIZE,random_state=42)
  train_idx, val_idx = next(msss.split(np.zeros(len(labels)), labels))
  train_set = Subset(dataset, train_idx)
  val_set = Subset(dataset, val_idx)
  train_size = int(TRAIN_SIZE * len(dataset))
  val_size = len(dataset) - train_size
  # train_set, val_set = random_split(dataset, [train_size, val_size])

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

  model = UNet(
    PRESETS["unet_base"].to_config(),
    multitask=MULTITASK,
    num_labels=dataset.num_classes,
  )
  # model.set_decoder_trainable(False)
  # print("[*] Decoder frozen for multitask finetuning")
  model.to(device)

  trainer = Trainer(
    device, model, MODEL_PATH, train_loader, val_loader,
    checkpoint_path=CHECKPOINT, writer_path=WRITER_PATH, eval_epoch=True,
    save_checkpoints=True, early_stopping=EARLY_STOP, dataset=dataset,
    finetune=FINETUNE
  )
  trainer.train()
