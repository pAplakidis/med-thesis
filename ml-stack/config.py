import json

def print_config(config):
  print("\n[*] Configuration:")
  print(f"EMA: {config['ema']}")
  print(f"Image size: {config['image_size']} - Norm mean: {config['norm_mean']} - Norm std: {config['norm_std']}")
  print(f"Train size: {config['train_size']} - Epochs: {config['epochs']} - Batch size: {config['batch_size']}")
  print(f"Learning rate: {config['lr']} - LR factor: {config['lr_factor']} - Weight decay: {config['weight_decay']}")
  print(f"LR patience: {config['lr_patience']} - Early stop epochs: {config['early_stop_epochs']}")
  print()

with open("configs/main.json", "r") as f:
  config = json.load(f)
  BASE_DATA_DIR = "../data"
  # BASE_DATA_DIR = "../anonymizer/anonymized"

  EMA = config["ema"]
  IMAGE_SIZE = config["image_size"]
  NORM_MEAN = config["norm_mean"]
  NORM_STD = config["norm_std"]
  TRAIN_SIZE = config["train_size"]
  VAL_SIZE = 1 - TRAIN_SIZE
  EPOCHS = config["epochs"]
  BATCH_SIZE = config["batch_size"]
  LR = config["lr"]
  LR_FACTOR = config["lr_factor"]
  WEIGHT_DECAY = config["weight_decay"]
  LR_PATIENCE = config["lr_patience"]
  EARLY_STOP_EPOCHS = config["early_stop_epochs"]

  print_config(config)
