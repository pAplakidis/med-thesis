#!/usr/bin/env python3
import os
from tqdm import tqdm, trange
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from config import *
from utils import *


class CTScanDataset(Dataset):
  def __init__(self, base_dir):
    self.base_dir = base_dir
    self.num_classes = len(rgb_colors)
    self.classes = list(rgb_colors.keys())

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
    self.image_folders = list(self.image_folders)
    self.mask_folders = list(self.mask_folders)

    self.images = [
      os.path.join(img_folder, img_name)
      for img_folder in self.image_folders
      for img_name in os.listdir(img_folder)
    ]
    self.masks = [
      os.path.join(mask_folder, mask_name)
      for mask_folder in self.mask_folders
      for mask_name in os.listdir(mask_folder)
    ]

    # make sure each image has a corresponding mask
    mask_names = [os.path.basename(mask).split('-')[-1] for mask in self.masks]
    for i in (t := trange(len(self.images) - 1, -1, -1, desc="[*] Filtering out images without masks")):
      if os.path.basename(self.images[i]) not in mask_names:
        del self.images[i]
    self.images = sorted(self.images)
    self.masks = sorted(self.masks)
    assert len(self.images) == len(self.masks), "Number of images and masks must be the same"

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
    print("[+] Dataset initialized.")

  def __len__(self):
    return len(self.masks)

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


if __name__ == "__main__":
  dataset = CTScanDataset("../anonymizer/anonymized")
  for data in dataset:
    image, mask = data
    print(image.shape, mask.shape, torch.unique(mask))
    # if torch.unique(mask).shape[0] > 1:
    #   visualize_ct_with_mask(image, mask)
