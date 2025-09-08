#!/usr/bin/env python3
import os
import sys
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

colors = {
    0: (0, 0, 0),          # background
    1: (128, 0, 0),        # category 1
    2: (0, 128, 0),        # category 2
    3: (128, 128, 0),      # category 3
    4: (0, 0, 128),        # category 4
    5: (128, 128, 128),    # category 5
    6: (255, 0, 0),        # category 6
    7: (0, 255, 0),        # category 7
    8: (0, 0, 255),        # category 8
}


class MaskExtractor:
  def __init__(self, root_dir):
    self.root_dir = root_dir

    self.result_files = []
    for dirpath, _, filenames in tqdm(os.walk(self.root_dir), desc="[*] Checking for annotation files ..."):
      for filename in filenames:
        if filename.lower() == "result.json":
          self.result_files.append(os.path.join(dirpath, filename))

  def run(self):
    print("[*] Extracting masks from results.json annotation files ...")
    for path in (t := tqdm(self.result_files)):
      t.set_description(path)

      img_dir = os.path.join(os.path.dirname(path), "images")
      out_dir = os.path.join(os.path.dirname(path), "masks")
      os.makedirs(out_dir, exist_ok=True)

      try:
        self.extract_mask(path, out_dir)
      except Exception as e:
        print(f"Error processing {path}: {e}")

  def extract_mask(self, path, out_dir):
    with open(path, "r") as f:
      coco = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]} # category_id -> name mapping
    images = {img["id"]: img for img in coco["images"]} # TODO: might need to match image by name, not id
    total = len(coco["images"])
    for i, img_info in enumerate(coco["images"]):
      img_id = img_info["id"]
      file_name = img_info["file_name"]
      image_name = os.path.basename(file_name)
      width, height = img_info["width"], img_info["height"]
      mask = np.zeros((height, width), dtype=np.uint8)

      anns = [ann for ann in coco["annotations"] if ann["image_id"] == img_id]
      for ann in anns:
        cat_id = ann["category_id"]
        for seg in ann["segmentation"]:
          poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
          cv2.fillPoly(mask, [poly], color=colors[cat_id + 1])

      mask_path = os.path.join(out_dir, image_name)
      Image.fromarray(mask).save(mask_path)
      print(f"Saved mask: {mask_path}")


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python mask_extractor.py <path_to_data_directory>")
    sys.exit(1)

  extractor = MaskExtractor(sys.argv[1])
  extractor.run()
