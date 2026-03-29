#!/usr/bin/env python3
import os
import numpy as np
import streamlit as st
from PIL import Image
import torch

from dataset import CTScanDataset

# run:  streamlit run arena.py
# or:   streamlit run arena.py --server.address 0.0.0.0 --server.port 8501

# TODO:
# too slow
# wrong class indices, colors etc
# crashes:
# Traceback (most recent call last):
#   File "/home/pavlos/venv/lib/python3.12/site-packages/streamlit/web/server/media_file_handler.py", line 95, in validate_absolute_path
#     self._storage.get_file(absolute_path)
#   File "/home/pavlos/venv/lib/python3.12/site-packages/streamlit/runtime/memory_media_file_storage.py", line 154, in get_file
#     raise MediaFileStorageError(
# streamlit.runtime.media_file_storage.MediaFileStorageError: Bad filename 'c3a08efdfbbea926aa0443a855f9a4f08d25a233bb31c4b5153e020b.jpg'. (No media file with id 'c3a08efdfbbea926aa0443a855f9a4f08d25a233bb31c4b5153e020b')
# add overlay view (?)

CLASS_NAMES = [
  "BONE",
  "EDH",
  "SDH",
  "IVH",
  "SAH",
  "CONTUSSION",
  "CLASS_6",
  "CLASS_7",
  "CLASS_8"
]

CLASS_COLORS = {
  0: (0, 0, 255),       # blue
  1: (255, 165, 0),     # orange
  2: (255, 255, 0),     # yellow
  3: (0, 255, 0),       # green
  4: (255, 192, 203),   # pink
  5: (255, 0, 0),       # red
  6: (128, 128, 128),   # gray
  7: (0, 255, 255),     # cyan
  8: (255, 0, 255)      # magenta
}

# utils
@st.cache_data
def load_dataset(base_dir):
  return CTScanDataset(base_dir)

@st.cache_data
def compute_class_balance(_dataset):
  return _dataset.get_class_balance()

@st.cache_data
def compute_mask_class_presence(_dataset):
  """
  Returns:
    dict: {image_path: set(classes_present)}
  """
  presence = {}

  for img_path, mask_path in zip(_dataset.images, _dataset.masks):
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)

    class_ids = (mask // 10).astype(np.int64)
    unique_classes = set(np.unique(class_ids).tolist())

    presence[img_path] = unique_classes

  return presence

def colorize_mask(mask):
  """
  Convert mask [H,W] with class ids -> RGB image
  """
  h, w = mask.shape
  colored = np.zeros((h, w, 3), dtype=np.uint8)

  for cls, color in CLASS_COLORS.items():
    colored[mask == cls] = color

  return colored

def load_image_and_mask(dataset, idx):
  image, mask = dataset[idx]

  # denormalize image
  image = image * 0.5 + 0.5
  image = image.squeeze().numpy()

  mask = mask.numpy()

  return image, mask


# for modele selector
def discover_models(model_dir):
  if not model_dir or not os.path.exists(model_dir):
    return []

  models = []
  for f in os.listdir(model_dir):
    if f.endswith(".pt") or f.endswith(".pth"):
      models.append(os.path.join(model_dir, f))
  return models

def load_model(path):
  # placeholder for future
  return torch.load(path, map_location="cpu")

# UI components
def render_class_balance(balance):
  st.subheader("Class Balance")
  total = sum(balance.values())
  for cls, count in balance.items():
    pct = (count / total) * 100 if total > 0 else 0
    st.write(f"{CLASS_NAMES[cls]}: {count} ({pct:.2f}%)")

def render_legend():
  st.subheader("Legend")

  for cls, name in enumerate(CLASS_NAMES):
    color = CLASS_COLORS.get(cls, (255, 255, 255))
    st.markdown(
      f"""
      <div style="display:flex; align-items:center;">
        <div style="
          width:20px;
          height:20px;
          background-color:rgb{color};
          margin-right:10px;
        "></div>
        <span>{name} (Class {cls})</span>
      </div>
      """,
      unsafe_allow_html=True
    )

def render_image_view(image, mask):
  col1, col2 = st.columns(2)

  with col1:
    st.subheader("CT Image")
    st.image(image, clamp=True)

  with col2:
    st.subheader("Mask")
    colored = colorize_mask(mask)
    st.image(colored)


# sidebar
def sidebar_image_selector(dataset, filtered_indices, selected_idx_cache=None):
  """
  Scrollable vertical file list like VSCode sidebar.
  """
  selected_idx = selected_idx_cache if selected_idx_cache is not None else filtered_indices[0] if filtered_indices else None

  container = st.sidebar.container()
  container.markdown("<div style='max-height:500px; overflow-y:auto; padding:0;'>", unsafe_allow_html=True)

  for idx in filtered_indices:
    filename = os.path.basename(dataset.images[idx])
    display_name = filename if len(filename) <= 30 else filename[:27] + "..."
    is_selected = idx == selected_idx

    style = f"""
      width: 100%;
      text-align: left;
      padding: 4px 8px;
      margin: 0;
      border-radius: 4px;
      background-color: {'#0a84ff' if is_selected else 'transparent'};
      color: {'white' if is_selected else 'black'};
      font-family: monospace;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    """

    if container.button(display_name, key=idx, help=filename):
      selected_idx = idx

    container.markdown(f"<style>button[title='{filename}']{{{style}}}</style>", unsafe_allow_html=True)

  container.markdown("</div>", unsafe_allow_html=True)
  return selected_idx

def sidebar_controls(dataset, presence_map):
  st.sidebar.title("Controls")

  # model section
  model_dir = st.sidebar.text_input("Model Directory (optional)")
  model_paths = discover_models(model_dir)

  selected_models = st.sidebar.multiselect(
    "Select Models",
    options=model_paths
  )

  st.sidebar.markdown("---")  # separator

  # class filter
  selected_classes = st.sidebar.multiselect(
    "Filter by classes",
    options=list(range(len(CLASS_NAMES))),
    format_func=lambda x: CLASS_NAMES[x]
  )

  # filter images
  filtered_indices = []
  for idx, img_path in enumerate(dataset.images):
    if not selected_classes:
      filtered_indices.append(idx)
      continue

    present = presence_map[img_path]
    if any(cls in present for cls in selected_classes):
      filtered_indices.append(idx)

  # file list
  selected_idx = sidebar_image_selector(dataset, filtered_indices)

  return selected_idx, selected_models

def main():
  st.title("CT Scan Arena")

  base_dir = st.text_input("Dataset Directory", "../data")

  if not os.path.exists(base_dir):
    st.warning("Dataset path does not exist")
    return

  dataset = load_dataset(base_dir)
  presence_map = compute_mask_class_presence(dataset)

  # sidebar
  selected_idx, selected_models = sidebar_controls(dataset, presence_map)

  # main panels
  balance = compute_class_balance(dataset)
  render_class_balance(balance)

  st.divider()

  image, mask = load_image_and_mask(dataset, selected_idx)
  render_image_view(image, mask)
  render_legend()


  # future: model predictions
  if selected_models:
    st.subheader("Model Predictions (coming soon)")
    for model_path in selected_models:
      st.write(f"Loaded: {os.path.basename(model_path)}")


if __name__ == "__main__":
  main()