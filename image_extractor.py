#!/usr/bin/env python3
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from pydicom import dcmread

class ImageExtractor:
  def __init__(self, root_dir):
    self.root_dir = root_dir

    self.dicom_files = []
    for dirpath, _, filenames in tqdm(os.walk(self.root_dir), desc="[*] Checking for dicom files ..."):
      for filename in filenames:
        if filename.lower().endswith('.dcm'):
          self.dicom_files.append(os.path.join(dirpath, filename))

  def run(self):
    for path in (t := tqdm(self.dicom_files)):
      t.set_description(path)
      out_dir = os.path.join(os.path.dirname(path), "images")
      os.makedirs(out_dir, exist_ok=True)

      try:
        self.extract_pixel_data(path, out_dir)
      except Exception as e:
        print(f"Error processing {path}: {e}")

  def extract_pixel_data(self, path, out_dir):
    ds = dcmread(path)
    if 'PixelData' not in ds:
      print(f"Skipping file {path} (no pixel data)")
      return
    pixel_array = ds.pixel_array

    out_path = os.path.join(out_dir, (os.path.splitext(os.path.basename(path))[0]))
    # os.makedirs(out_path, exist_ok=True)
    out_path += ".png"  # TODO: if multiframe, make folder of out_path and save each frame with index

    if pixel_array.ndim > 2:
      print("Multi-frame DICOM detected, skipping for now.")
      return
    plt.imsave(out_path, pixel_array, cmap='gray')


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python image_extractor.py <path_to_dicom_directory>")
    sys.exit(1)

  extractor = ImageExtractor(sys.argv[1])
  extractor.run()
