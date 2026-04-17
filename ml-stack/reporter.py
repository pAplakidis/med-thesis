#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

from config import *
from utils import RGB_COLORS

# run: streamlit run reporter.py --server.address 0.0.0.0 --server.port 8501

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

METRIC_NAMES = {
  "loss": "Loss",
  "pixel_acc": "Pixel Accuracy",
  "IoU": "IoU",
  "Dice": "Dice",
  "F1": "F1",
  "Hausdorff": "Hausdorff Distance",
  "w_IoU": "Weighted IoU",
  "w_Dice": "Weighted Dice",
  "w_F1": "Weighted F1",
  "w_Hausdorff": "Weighted Hausdorff Distance",
}

DEFAULT_COLORS = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


@st.cache_data
def load_report(report_path):
  with open(report_path, "r") as f:
    return json.load(f)


@st.cache_data
def load_tensorboard_scalars(run_path):
  if not os.path.exists(run_path):
    return {}

  ea = event_accumulator.EventAccumulator(run_path)
  ea.Reload()

  scalars = {}
  for tag in ea.Tags().get("scalars", []):
    events = ea.Scalars(tag)
    scalars[tag] = {
      "step": [e.step for e in events],
      "value": [e.value for e in events],
      "wall_time": [e.wall_time for e in events],
    }
  return scalars


def discover_checkpoints(checkpoints_dir, reports_dir):
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
        report_path = os.path.join(reports_dir, experiment, os.path.splitext(f)[0] + ".json")
        if os.path.exists(os.path.join(exp_path, onnx_path)) and os.path.exists(report_path):
          versions.append(f)
    if versions:
      hierarchy[experiment] = versions
  return hierarchy


def find_report_for_checkpoint(reports_dir, checkpoints_dir, experiment, version):
  onnx_name = os.path.splitext(version)[0] + ".json"
  report_path = os.path.join(reports_dir, experiment, onnx_name)
  if os.path.exists(report_path):
    return report_path
  return None


def get_writer_path(checkpoints_dir, experiment, version):
  checkpoint_path = os.path.join(checkpoints_dir, experiment, version)
  if not os.path.exists(checkpoint_path):
    return None

  try:
    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
      if checkpoint.get("writer"):
        return checkpoint["writer"]
  except Exception:
    pass

  runs_dir = os.path.join(os.path.dirname(checkpoints_dir), "runs")
  if not os.path.exists(runs_dir):
    return None

  version_base = os.path.splitext(version)[0].replace("_best", "")
  run_prefix = version_base + "-"
  
  for run_name in sorted(os.listdir(runs_dir), reverse=True):
    run_path = os.path.join(runs_dir, run_name)
    if not os.path.isdir(run_path):
      continue
   
    if run_name.startswith(run_prefix):
      return run_path
  
  return None


def build_scalars_plot(scalars_dict, model_labels, metric_key, color_map, y_log=False):
  fig = go.Figure()

  for label, scalars in zip(model_labels, scalars_dict):
    if metric_key not in scalars or scalars[metric_key] is None:
      continue

    data = scalars[metric_key]
    if not isinstance(data, dict) or "step" not in data or "value" not in data:
      continue

    steps = data["step"]
    values = data["value"]
    color = color_map.get(label, DEFAULT_COLORS[len(color_map) % len(DEFAULT_COLORS)])

    fig.add_trace(go.Scatter(
      x=steps,
      y=values,
      mode="lines+markers",
      name=label,
      line=dict(color=color, width=2),
      marker=dict(size=4),
      hovertemplate=f"<b>{label}</b><br>Step: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
    ))

  is_loss = "loss" in metric_key.lower()
  fig.update_layout(
    title=METRIC_NAMES.get(metric_key, metric_key),
    xaxis_title="Step",
    yaxis_title=METRIC_NAMES.get(metric_key, metric_key),
    yaxis_type="log" if y_log else "linear",
    hovermode="x unified",
    legend=dict(
      orientation="h",
      yanchor="bottom",
      y=1.02,
      xanchor="right",
      x=1,
    ),
    margin=dict(l=40, r=40, t=60, b=40),
    height=400,
  )

  fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
  fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")

  return fig


def build_comparison_table(reports, model_labels):
  data = []
  for label, report in zip(model_labels, reports):
    summary = report.get("summary", {})
    avg_metrics = summary.get("avg_metrics", {})
    model_info = report.get("model", {})

    row = {
      "Model": label,
      "Size (MB)": f"{model_info.get('size_bytes', 0) / (1024*1024):.2f}",
      "Params": f"{model_info.get('param_count', 0):,}" if model_info.get('param_count') else "N/A",
      "Avg Inference (ms)": f"{summary.get('avg_inference_time_sec', 0) * 1000:.2f}",
      "Train Samples": summary.get("train_samples", 0),
      "Val Samples": summary.get("val_samples", 0),
    }

    for metric_key in ["pixel_acc", "IoU", "Dice", "F1", "Hausdorff", "w_IoU", "w_Dice", "w_F1", "w_Hausdorff"]:
      row[METRIC_NAMES.get(metric_key, metric_key)] = f"{avg_metrics.get(metric_key, 0):.4f}"

    data.append(row)

  return pd.DataFrame(data)


def build_per_class_table(reports, model_labels):
  rows = []
  for label, report in zip(model_labels, reports):
    predictions = report.get("predictions", [])
    if not predictions:
      continue

    class_metrics = {}
    for pred in predictions:
      metrics = pred.get("metrics", {})
      for cls in range(len(RGB_COLORS)):
        if cls not in class_metrics:
          class_metrics[cls] = {"IoU": [], "Dice": [], "F1": []}

        if "per_class" in metrics:
          for metric_name in ["IoU", "Dice", "F1"]:
            if metric_name in metrics["per_class"].get(str(cls), {}):
              class_metrics[cls][metric_name].append(metrics["per_class"][str(cls)][metric_name])

    for cls in range(len(RGB_COLORS)):
      if cls in class_metrics and class_metrics[cls]["IoU"]:
        rows.append({
          "Model": label,
          "Class": CLASS_NAMES[cls],
          "Avg IoU": f"{np.mean(class_metrics[cls]['IoU']):.4f}",
          "Avg Dice": f"{np.mean(class_metrics[cls]['Dice']):.4f}",
          "Avg F1": f"{np.mean(class_metrics[cls]['F1']):.4f}",
        })

  if not rows:
    return None
  return pd.DataFrame(rows)


def build_top_worst_table(report, model_label, metric="IoU", n=10, ascending=True):
  predictions = report.get("predictions", [])
  if not predictions:
    return None

  sorted_preds = sorted(predictions, key=lambda p: p.get("metrics", {}).get(metric, 0), reverse=not ascending)
  top_preds = sorted_preds[:n]

  rows = []
  for pred in top_preds:
    rows.append({
      "Model": model_label,
      "Image": os.path.basename(pred.get("image_path", "")),
      "Split": pred.get("split", ""),
      metric: f"{pred.get('metrics', {}).get(metric, 0):.4f}",
      "Pixel Acc": f"{pred.get('metrics', {}).get('pixel_acc', 0):.4f}",
      "Dice": f"{pred.get('metrics', {}).get('Dice', 0):.4f}",
    })

  return pd.DataFrame(rows)


def sidebar_model_selector(checkpoints_dir, reports_dir):
  st.sidebar.title("Model Selection")

  checkpoint_hierarchy = discover_checkpoints(checkpoints_dir, reports_dir)

  st.sidebar.subheader("Training Runs")
  selected_checkpoint_exps = st.sidebar.multiselect(
    "Checkpoint Experiments",
    options=list(checkpoint_hierarchy.keys()),
    key="ckpt_exp"
  )

  selected_checkpoints = []
  checkpoint_map = {}
  for experiment in selected_checkpoint_exps:
    selected_versions = st.sidebar.multiselect(
      f"Checkpoints: {experiment}",
      options=checkpoint_hierarchy[experiment],
      key=f"ckpt_ver_{experiment}"
    )
    for version in selected_versions:
      label = f"{experiment}/{version}"
      selected_checkpoints.append(label)
      checkpoint_map[label] = (experiment, version)

  return selected_checkpoints, checkpoint_map


def main():
  st.set_page_config(page_title="Model Reporter", layout="wide")
  st.title("Model Reporter")

  checkpoints_dir = st.sidebar.text_input("Checkpoints Directory", "checkpoints")
  reports_dir = st.sidebar.text_input("Reports Directory", "reports")

  checkpoints_dir = os.path.abspath(checkpoints_dir)
  reports_dir = os.path.abspath(reports_dir)

  st.sidebar.caption(f"Checkpoints: `{checkpoints_dir}`\n\nReports: `{reports_dir}`")

  selected_checkpoints, checkpoint_map = sidebar_model_selector(checkpoints_dir, reports_dir)

  if not selected_checkpoints:
    st.info("Select checkpoints from the sidebar to begin. Only checkpoints with both a .onnx and a corresponding .json report are shown.")
    return

  all_labels = []
  reports = []
  report_labels = []
  checkpoint_scalars = {}
  checkpoint_labels = []

  for label in selected_checkpoints:
    exp, version = checkpoint_map[label]
    all_labels.append(label)

    report_path = find_report_for_checkpoint(reports_dir, checkpoints_dir, exp, version)
    if report_path and os.path.exists(report_path):
      report = load_report(report_path)
      reports.append(report)
      report_labels.append(label)

    writer_path_val = get_writer_path(checkpoints_dir, exp, version)
    if writer_path_val:
      scalars = load_tensorboard_scalars(writer_path_val)
      if scalars:
        checkpoint_scalars[label] = scalars
        checkpoint_labels.append(label)

  if not all_labels:
    st.warning("No valid checkpoints selected.")
    return

  st.subheader("Model Comparison")
  if reports:
    comparison_df = build_comparison_table(reports, report_labels)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

  st.divider()

  st.subheader("Training Curves")
  all_scalars = {}
  for label in checkpoint_labels:
    if label in checkpoint_scalars:
      all_scalars[label] = checkpoint_scalars[label]

  available_tags = set()
  for scalars in all_scalars.values():
    available_tags.update(scalars.keys())
  available_tags = sorted(available_tags)

  if available_tags:
    col1, col2 = st.columns([3, 1])
    with col1:
      selected_metrics = st.multiselect(
        "Select Metrics",
        options=available_tags,
        default=[t for t in available_tags if "loss" in t.lower()][:2],
      )
    with col2:
      y_log = st.checkbox("Log Scale", value=False)

    if selected_metrics:
      n_metrics = len(selected_metrics)
      if n_metrics == 1:
        metric = selected_metrics[0]
        scalars_list = []
        labels_list = []
        for label in all_labels:
          if label in all_scalars and metric in all_scalars[label]:
            scalars_list.append(all_scalars[label])
            labels_list.append(label)

        if not labels_list:
          st.info(f"No data for metric: {METRIC_NAMES.get(metric, metric)}")
        else:
          color_map = {}
          for i, label in enumerate(labels_list):
            color_map[label] = st.color_picker(f"Color: {label}", value=DEFAULT_COLORS[i % len(DEFAULT_COLORS)], key=f"color_{label}_{metric}")

          fig = build_scalars_plot(scalars_list, labels_list, metric, color_map, y_log=y_log)
          st.plotly_chart(fig, use_container_width=True)
      else:
        n_cols = min(n_metrics, 2)

        rows = []
        for i in range(0, n_metrics, n_cols):
          rows.append(st.columns(n_cols))

        for idx, metric in enumerate(selected_metrics):
          row_idx = idx // n_cols
          col_idx = idx % n_cols
          with rows[row_idx][col_idx]:
            scalars_list = []
            labels_list = []
            for label in all_labels:
              if label in all_scalars and metric in all_scalars[label]:
                scalars_list.append(all_scalars[label])
                labels_list.append(label)

            if not labels_list:
              st.info(f"No data for {METRIC_NAMES.get(metric, metric)}")
              continue

            color_map = {}
            for i, label in enumerate(labels_list):
              color_map[label] = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]

            fig = build_scalars_plot(scalars_list, labels_list, metric, color_map, y_log=y_log)
            st.plotly_chart(fig, use_container_width=True)
  else:
    st.info("No TensorBoard scalars found for selected checkpoints.")

  st.divider()

  if reports:
    st.subheader("Per-Class Metrics")
    per_class_df = build_per_class_table(reports, report_labels)
    if per_class_df is not None:
      st.dataframe(per_class_df, use_container_width=True, hide_index=True)
    else:
      st.info("Per-class metrics not available in reports.")

    st.divider()

    st.subheader("Top / Worst Predictions")
    col1, col2, col3 = st.columns(3)
    with col1:
      selected_model = st.selectbox("Model", options=report_labels, key="top_worst_model")
    with col2:
      selected_metric = st.selectbox("Metric", options=["IoU", "Dice", "F1", "pixel_acc", "Hausdorff"], key="top_worst_metric")
    with col3:
      n_results = st.number_input("Number of results", min_value=1, max_value=50, value=10, key="top_worst_n")

    selected_report = None
    for label, report in zip(report_labels, reports):
      if label == selected_model:
        selected_report = report
        break

    if selected_report:
      tab_top, tab_worst = st.tabs(["Top Predictions", "Worst Predictions"])
      with tab_top:
        top_df = build_top_worst_table(selected_report, selected_model, metric=selected_metric, n=n_results, ascending=False)
        if top_df is not None:
          st.dataframe(top_df, use_container_width=True, hide_index=True)
      with tab_worst:
        worst_df = build_top_worst_table(selected_report, selected_model, metric=selected_metric, n=n_results, ascending=True)
        if worst_df is not None:
          st.dataframe(worst_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
  main()
