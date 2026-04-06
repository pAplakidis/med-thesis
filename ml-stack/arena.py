#!/usr/bin/env python3
import os
import pickle
import numpy as np
import streamlit as st
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from dataset import CTScanDataset
from utils import RGB_COLORS

# run:  streamlit run arena.py
# or:   streamlit run arena.py --server.address 0.0.0.0 --server.port 8501

# epidural hematoma (EDH)
# subdural hematoma (SDH)
# intraventricular hemorrhage (IVH)
# subarachnoid hemorrhage (SAH)
# intraparenchymal hemorrhage (IPH)
CLASS_NAMES = [
  "BACKGROUND",
  "BONE",
  "EDH",
  "SDH",
  "IVH",
  "SAH",
  "CONTUSSION",
  "CLASS_6",
  "CLASS_7",
]

# utils
@st.cache_data
def load_dataset(base_dir):
  return CTScanDataset(base_dir)

@st.cache_data
def compute_class_balance(_dataset):
  return _dataset.get_class_balance()

@st.cache_data
def compute_mask_class_presence(_dataset, cache_path="presence.pkl"):
  """
  Returns:
    dict: {image_path: set(classes_present)}
  """
  if os.path.exists(cache_path):
    with open(cache_path, "rb") as f:
      return pickle.load(f)

  presence = {}
  for img_path, mask_path in tqdm(zip(_dataset.images, _dataset.masks), desc="[*] Computing mask class presence", total=len(_dataset.images)):
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)
    print(mask)
    class_ids = (mask // 10).astype(np.int64)
    unique_classes = set(np.unique(class_ids).tolist())
    presence[img_path] = unique_classes

  with open(cache_path, "wb") as f:
    pickle.dump(presence, f)

  return presence

def colorize_mask(mask):
  """
  Convert mask [H,W] with class ids -> RGB image
  """
  h, w = mask.shape
  colored = np.zeros((h, w, 3), dtype=np.uint8)

  for cls, color in RGB_COLORS.items():
    colored[mask == cls] = color

  return colored

@st.cache_data
def load_image_and_mask(_dataset, idx):
  image, mask = _dataset[idx]

  # denormalize image
  disp_image = image * 0.5 + 0.5
  disp_image = image.squeeze().numpy()

  disp_mask = mask.numpy()

  return disp_image, disp_mask, image, mask

def discover_models(checkpoints_dir):
  if not checkpoints_dir or not os.path.exists(checkpoints_dir):
    return {}

  hierarchy = {}
  for experiment in sorted(os.listdir(checkpoints_dir)):
    exp_path = os.path.join(checkpoints_dir, experiment)
    if not os.path.isdir(exp_path):
      continue
    versions = []
    for f in sorted(os.listdir(exp_path)):
      if f.endswith(".pt") or f.endswith(".pth"):
        onnx_path = os.path.splitext(f)[0] + ".onnx"
        if os.path.exists(os.path.join(exp_path, onnx_path)):
          versions.append(f)
    if versions:
      hierarchy[experiment] = versions
  return hierarchy

@st.cache_resource
def load_model(checkpoints_dir, _dataset, experiment, version):
  onnx_path = os.path.join(checkpoints_dir, experiment, os.path.splitext(version)[0] + ".onnx")

  if not os.path.exists(onnx_path):
    st.error(f"ONNX model not found: {onnx_path}")
    return None

  try:
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return session
  except Exception as e:
    st.error(f"Failed to load ONNX model: {e}")
    return None

def discover_runs(runs_dir="runs"):
  if not runs_dir or not os.path.exists(runs_dir):
    return {}

  runs = {}
  for run_name in sorted(os.listdir(runs_dir)):
    run_path = os.path.join(runs_dir, run_name)
    if not os.path.isdir(run_path):
      continue
    model_name = run_name.split("-")[0]
    if model_name not in runs:
      runs[model_name] = []
    runs[model_name].append(run_name)
  return runs

def get_final_val_loss(runs_dir, run_name):
  try:
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing import event_accumulator
    run_path = os.path.join(runs_dir, run_name)
    ea = event_accumulator.EventAccumulator(run_path)
    ea.Reload()
    val_scalars = ea.Scalars("epoch validation/loss")
    if val_scalars:
      return val_scalars[-1].value
    train_scalars = ea.Scalars("epoch training/loss")
    if train_scalars:
      return train_scalars[-1].value
  except Exception:
    pass
  return None

def get_model_final_loss(checkpoints_dir, experiment, version):
  model_name = version.split(".")[0]
  runs = discover_runs()
  if model_name not in runs:
    return None
  latest_run = runs[model_name][-1]
  return get_final_val_loss("runs", latest_run)

def predict_with_model(model, image):
  if model is None:
    return None
  input_np = image.unsqueeze(0).numpy().astype(np.float32)
  input_name = model.get_inputs()[0].name
  out = model.run(None, {input_name: input_np})[0]
  pred = np.argmax(out, axis=1).squeeze(0)
  return pred

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
    color = RGB_COLORS.get(cls, (255, 255, 255))
    st.markdown(
      f"""
      <div style="display:flex; align-items:center;">
        <div style="
          width:20px;
          height:20px;
          background-color:rgb{color};
          margin-right:10px;
        "></div>
        <span>{name} (Class {cls-1})</span>
      </div>
      """,
      unsafe_allow_html=True
    )

def to_pil(img):
    return Image.fromarray((img * 255).astype(np.uint8)) if img.ndim == 2 else Image.fromarray(img)

def overlay_mask(image, mask, alpha=0.4):
    """
    Overlay segmentation mask on grayscale image
    """
    image_rgb = np.stack([image]*3, axis=-1)
    image_rgb = (image_rgb * 255).astype(np.uint8)

    colored_mask = colorize_mask(mask)

    overlay = (1 - alpha) * image_rgb + alpha * colored_mask
    return overlay.astype(np.uint8)

def render_image_view(image, mask):
  col1, col2, col3 = st.columns(3)

  with col1:
    st.subheader("CT Image")
    st.image(to_pil(image), width="stretch")

  with col2:
    st.subheader("Ground Truth")
    st.image(to_pil(colorize_mask(mask)), width="stretch")

  with col3:
    st.subheader("Overlay")
    st.image(to_pil(overlay_mask(image, mask)), width="stretch")

def build_hierarchy(dataset):
  """
  Build patient -> study -> images hierarchy from dataset image paths.
  Assumes path structure: .../patient_name/study_name/images/image_name
  """
  hierarchy = {}
  for img_path in dataset.images:
    parts = Path(img_path).parts
    # Find 'images' folder index; patient is 2 levels before, study is 1 level before
    try:
      images_idx = parts.index("images")
    except ValueError:
      continue
    patient = parts[images_idx - 2]
    study = parts[images_idx - 1]
    if patient not in hierarchy:
      hierarchy[patient] = {}
    if study not in hierarchy[patient]:
      hierarchy[patient][study] = []
    hierarchy[patient][study].append(img_path)
  return hierarchy

def sidebar_image_selector(dataset, filtered_indices):
  hierarchy = build_hierarchy(dataset)
  filtered_set = set(dataset.images[i] for i in filtered_indices)

  # Build filtered hierarchy
  filtered_hierarchy = {}
  for patient, studies in hierarchy.items():
    for study, images in studies.items():
      matching = [img for img in images if img in filtered_set]
      if matching:
        if patient not in filtered_hierarchy:
          filtered_hierarchy[patient] = {}
        filtered_hierarchy[patient][study] = matching

  patients = sorted(filtered_hierarchy.keys())
  selected_patient = st.sidebar.selectbox("Patient", options=["(All)"] + patients)

  studies = []
  if selected_patient and selected_patient != "(All)":
    studies = sorted(filtered_hierarchy[selected_patient].keys())
  selected_study = st.sidebar.selectbox("Study", options=["(All)"] + studies)

  # Collect candidate images
  candidate_images = []
  if selected_patient == "(All)":
    for patient, studies_dict in filtered_hierarchy.items():
      for study, images in studies_dict.items():
        candidate_images.extend(images)
  elif selected_study == "(All)":
    for study, images in filtered_hierarchy[selected_patient].items():
      candidate_images.extend(images)
  else:
    candidate_images = filtered_hierarchy[selected_patient][selected_study]

  candidate_images = sorted(candidate_images)

  if not candidate_images:
    return None

  display_names = candidate_images
  selected_path = st.sidebar.selectbox("Select Image", options=display_names)

  return dataset.images.index(selected_path)

def filter_indices(dataset, presence_map, selected_classes):
  if not selected_classes:
    return list(range(len(dataset.images)))
  selected_classes = set(selected_classes)
  return [i for i, img_path in enumerate(dataset.images) if selected_classes.issubset(presence_map[img_path])]

def sidebar_controls(dataset, presence_map):
  st.sidebar.title("Controls")

  # model section
  checkpoints_dir = st.sidebar.text_input("Checkpoints Directory (optional)", "checkpoints")
  model_hierarchy = discover_models(checkpoints_dir)

  selected_experiments = st.sidebar.multiselect(
    "Select Experiments",
    options=list(model_hierarchy.keys())
  )

  selected_models = []
  model_map = {}
  for experiment in selected_experiments:
    selected_versions = st.sidebar.multiselect(
      f"Versions: {experiment}",
      options=model_hierarchy[experiment]
    )
    for version in selected_versions:
      label = f"{experiment}/{version}"
      selected_models.append(label)
      model_map[label] = (experiment, version)

  st.sidebar.markdown("---")  # separator

  # class filter
  selected_classes = st.sidebar.multiselect(
    "Filter by classes",
    options=list(range(len(CLASS_NAMES))),
    format_func=lambda x: CLASS_NAMES[x]
  )

  # filter images
  filtered_indices = filter_indices(dataset, presence_map, selected_classes)

  # file list
  selected_idx = sidebar_image_selector(dataset, filtered_indices)

  return selected_idx, checkpoints_dir, selected_models, model_map

def main():
  st.title("CT Scan Arena")
  st.set_page_config(layout="wide")

  base_dir = st.sidebar.text_input("Dataset Directory", "../data")
  if not os.path.exists(base_dir):
    st.warning("Dataset path does not exist")
    return

  dataset = load_dataset(base_dir)
  presence_map = compute_mask_class_presence(dataset)

  # sidebar (left)
  selected_idx, checkpoints_dir, selected_models, model_map = sidebar_controls(dataset, presence_map)
  if selected_idx is None:
    st.warning("No image selected")
    return

  # 👉 main layout with right panel
  main_col, right_col = st.columns([3, 1])
  with main_col:
    image, mask, t_image, t_mask  = load_image_and_mask(dataset, selected_idx)
    render_image_view(image, mask)

    st.divider()

    # model predictions
    if selected_models:
      st.subheader("Model Predictions")
      for model_label in selected_models:
        experiment, version = model_map[model_label]
        model = load_model(checkpoints_dir, dataset, experiment, version)
        if model is not None:
          pred = predict_with_model(model, t_image)
          if pred is not None:
            final_loss = get_model_final_loss(checkpoints_dir, experiment, version)
            loss_suffix = f" (val loss: {final_loss:.4f})" if final_loss is not None else ""
            col1, col2 = st.columns(2)
            with col1:
              st.subheader(f"{model_label}{loss_suffix}")
              st.image(to_pil(colorize_mask(pred)), width="stretch")
            with col2:
              st.subheader(f"{model_label} (Overlay)")
              st.image(to_pil(overlay_mask(image, pred)), width="stretch")

  # RIGHT: stats + legend
  with right_col:
    st.subheader("Classes in Image")
    present = presence_map[dataset.images[selected_idx]]
    for cls in sorted(present):
      st.write(f"{CLASS_NAMES[cls]}")
    st.divider()
    render_legend()
    balance = compute_class_balance(dataset)
    render_class_balance(balance)


if __name__ == "__main__":
  main()