#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import pydicom
from pydicom import dcmread

class ImageExtractor:
  def __init__(self, root_dir):
    self.root_dir = root_dir

    self.dicom_files = []
    for dirpath, _, filenames in tqdm(os.walk(self.root_dir), desc="[*] Checking for dicom files ..."):
      for filename in filenames:
        if filename.lower().endswith('.dcm'):
          self.dicom_files.append(os.path.join(dirpath, filename))

  @staticmethod
  def preprocess_monochrome2(ds, pixel_array):
    pixel_array = pixel_array.astype(float)
    if "RescaleSlope" in ds and "RescaleIntercept" in ds:
      pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    pixel_array = ImageExtractor.apply_window(ds, pixel_array)
    return pixel_array

  @staticmethod
  def apply_window(ds, pixel_array):
    if "WindowCenter" in ds and "WindowWidth" in ds:
      center = ds.WindowCenter
      width = ds.WindowWidth
      # Handle if multiple values exist
      if isinstance(center, pydicom.multival.MultiValue):
        center = center[0]
      if isinstance(width, pydicom.multival.MultiValue):
        width = width[0]
      center, width = float(center), float(width)

      lower = center - width/2
      upper = center + width/2
      pixel_array = np.clip(pixel_array, lower, upper)
      pixel_array = (pixel_array - lower) / (upper - lower)  # normalize to 0-1
      pixel_array = (pixel_array * 255).astype(np.uint8)
    else:
      # Fallback: just normalize min–max
      pixel_array = pixel_array - np.min(pixel_array)
      pixel_array = pixel_array / np.max(pixel_array)
      pixel_array = (pixel_array * 255).astype(np.uint8)

    return pixel_array

  def run(self):
    print("[*] Extracting pixel data from DICOM files ...")
    tqdm_string = ""
    for path in (t := tqdm(self.dicom_files)):
      tqdm_string = path
      t.set_description(path)
      out_dir = os.path.join(os.path.dirname(path), "images")
      os.makedirs(out_dir, exist_ok=True)

      try:
        self.extract_pixel_data(path, out_dir, tqdm_string)
      except Exception as e:
        print(f"Error processing {path}: {e}")

  def extract_pixel_data(self, path, out_dir, tqdm_string):
    ds = dcmread(path)
    if 'PixelData' not in ds:
      print(f"Skipping file {path} (no pixel data)")
      return
    pixel_array = ds.pixel_array
    out_path = os.path.join(out_dir, (os.path.splitext(os.path.basename(path))[0]))

    pi = ds.PhotometricInterpretation # usually MONOCHROME2 or RGB
    if pi == "MONOCHROME2":
      pixel_array = self.preprocess_monochrome2(ds, pixel_array)

    if pixel_array.ndim > 2:
      # NOTE: all images with pixel array > 2D are RGB reports (skip for now)
      # os.makedirs(out_path, exist_ok=True)
      # for i, frame in enumerate(pixel_array):
      #   tqdm_string += f" - Multiframe - processing frame {i}/{pixel_array.shape[0] - 1}"
      #   frame_out_path = os.path.join(out_path, f"frame_{i}.png")
      #   plt.imsave(frame_out_path, frame, cmap='gray')
      print("Multi-frame DICOM detected (probably report image), skipping for now.")
      return

    out_path += ".png"
    plt.imsave(out_path, pixel_array, cmap='gray')


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python image_extractor.py <path_to_dicom_directory>")
    sys.exit(1)

  extractor = ImageExtractor(sys.argv[1])
  extractor.run()
