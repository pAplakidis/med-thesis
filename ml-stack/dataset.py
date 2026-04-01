#!/usr/bin/env python3
import os
import numpy as np
from tqdm import tqdm, trange
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms

from config import *
from utils import *


class CTScanDataset(Dataset):
  def __init__(self, base_dir):
    self.base_dir = base_dir
    self.num_classes = len(RGB_COLORS)
    self.classes = list(RGB_COLORS.keys())

    # find annotated studies
    self.result_files = []
    for dirpath, _, filenames in tqdm(os.walk(self.base_dir), desc="[*] Parsing dataset directory"):
      for filename in filenames:
        if filename.lower().startswith("result") and filename.lower().endswith(".json"):
          self.result_files.append(os.path.join(dirpath, filename))

    # get image and mask folders
    self.image_folders = set()
    self.mask_folders = set()
    for res in tqdm(self.result_files, desc="[*] Collecting image and mask folders"):
      self.image_folders.add(os.path.join(os.path.dirname(res), "images"))
      self.mask_folders.add(os.path.join(os.path.dirname(res), "masks"))
    self.image_folders = sorted(list(self.image_folders))
    self.mask_folders = sorted(list(self.mask_folders))

    self.images, self.masks = [], []
    for img_folder in (t := tqdm(self.image_folders)):
      t.set_description(f"[*] Processing image folder: {os.path.basename(img_folder)}")
      mask_folder = img_folder.replace("images", "masks")

      # FIXME: check excel, if patient operated + folder name > 1H CT => exclude from dataset (since we only want pre-op scans)
      # if operated_on and "1h" not in img_folder.lower():
      #   continue
      if not os.path.exists(img_folder.replace("images", "masks")):
        print("[!] Warning: No corresponding mask folder found for image folder: {}".format(img_folder))
        continue

      images = os.listdir(img_folder)
      masks = os.listdir(mask_folder)
      mask_map = {mask.split('-')[-1]: mask for mask in masks}  # mask file format: id-<original_image_name>
      for img_name in images:
        if img_name not in mask_map:
          continue
        self.images.append(os.path.join(img_folder, img_name))
        self.masks.append(os.path.join(mask_folder, mask_map[img_name]))

    self.images = sorted(self.images)
    self.masks = sorted(self.masks)
    print(f"[*] Found {len(self.images)} images and {len(self.masks)} masks.")
    assert len(self.images) == len(self.masks), f"Number of images and masks must be the same: {len(self.images)} != {len(self.masks)}"

    # pair images and masks
    # make sure each image has a corresponding mask
    print("[*] Pairing images and masks ...")
    mask_dict = {
      os.path.basename(mask).split('-', 1)[-1]: mask  # drop mask prefix before first '-'
      for mask in self.masks
    }
    paired = [(img, mask_dict[os.path.basename(img)]) for img in self.images] # keep only images that have a mask, and pair them properly
    self.images, self.masks = zip(*paired)
    self.images, self.masks = list(self.images), list(self.masks)
    print("[+] Found {} image-mask pairs".format(len(self.images)))

    # TODO: include some empty masks for better class balance
    # filter out masks with only background (class 0)
    print("[*] Filtering out empty masks...")
    filtered_images, filtered_masks = [], []
    for img, mask in zip(self.images, self.masks):
      m = Image.open(mask).convert("L")
      m = np.array(m)
      class_ids = (m // 10).astype(np.int64)
      if np.any(class_ids > 0):
        filtered_images.append(img)
        filtered_masks.append(mask)
    self.images, self.masks = filtered_images, filtered_masks
    print(f"[+] Kept {len(self.images)} pairs with at least one foreground class")

    print("[+] Dataset initialized.")

  def __len__(self):
    return len(self.masks)

  def get_classes(self):
    return self.classes

  def __getitem__(self, idx):
    # load grayscale image and mask
    image = Image.open(self.images[idx]).convert("L")   # CT scan image (grayscale)
    mask = Image.open(self.masks[idx]).convert("L")     # segmentation mask (grayscale)

    # transforms for CT image
    image_transform = transforms.Compose([
      transforms.Resize(IMAGE_SIZE),    # resize to fixed size
      transforms.ToTensor(),                      # -> [1,H,W], values in [0,1]
      transforms.Normalize(mean=[0.5], std=[0.5]) # normalize grayscale
    ])

    # transforms for mask
    mask_transform = transforms.Compose([
      transforms.Resize(IMAGE_SIZE, interpolation=Image.NEAREST),  # preserve class IDs
      transforms.ToTensor(),                                                 # -> [1,H,W], values in [0,1]
      transforms.Lambda(lambda x: (x * 255).long() // 10)                    # divide by 10 rescale to category IDs
    ])

    image = image_transform(image)
    mask = mask_transform(mask).squeeze(0)  # drop channel -> [H,W]
    return image, mask

  def get_class_balance(self):
    """Per image class balance"""
    class_counts = {cls: 0 for cls in range(self.num_classes)}
    for mask_path in tqdm(self.masks, desc="[*] Calculating class balance"):
      mask = Image.open(mask_path).convert("L")
      mask_tensor = transforms.ToTensor()(mask).squeeze(0)  # [H,W], values in [0,1]
      class_ids = (mask_tensor * 255).long() // 10           # rescale to category IDs
      for cls in torch.unique(class_ids):
        cls = cls.item()
        if cls < self.num_classes:
          class_counts[cls] += 1
    return class_counts

  def get_class_balance_per_pixel(self):
    """Per pixel class balance across the dataset (total pixels belonging to each class)"""
    class_counts = {cls: 0 for cls in range(self.num_classes)}
    for mask_path in tqdm(self.masks, desc="[*] Calculating class balance"):
      mask = Image.open(mask_path).convert("L")
      mask_tensor = transforms.ToTensor()(mask).squeeze(0)  # [H,W], values in [0,1]
      class_ids = (mask_tensor * 255).long() // 10           # rescale to category IDs
      for cls in range(self.num_classes):
        class_counts[cls] += (class_ids == cls).sum().item()
    return class_counts


if __name__ == "__main__":
  dataset = CTScanDataset("../data")
  class_counts = dataset.get_class_balance()
