import os
import time
import json
import joblib

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split, ParameterGrid, StratifiedKFold
from sklearn.metrics import (
  accuracy_score,
  precision_score,
  recall_score,
  f1_score,
  roc_auc_score,
  confusion_matrix,
  roc_curve
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.tensorboard import SummaryWriter

# FIXME: cleanup incosistent cells manually

INF = float("inf")

# Hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
KFOLD = 5
MAX_TREE_DEPTH = 5

# experiment tracking
RUN_NAME = time.strftime("%Y%m%d-%H%M%S")
RUN_DIR = f"runs/prognosis-classifiers/{RUN_NAME}"
PLOT_DIR = f"{RUN_DIR}/plots"
CKPT_DIR = "checkpoints/prognosis-classifiers"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
LOG_FILE = f"{RUN_DIR}/log.txt"
writer = SummaryWriter(RUN_DIR)
print("[*] Writer initialized at:", RUN_DIR)

# Data constants
INPUT_COLUMNS = [
  "Age",
  "MECHANISM OF INJURY",
  "ISOLATED VS MULTI TRAUMA",
  "HEMODYNAMICAL STATUS (STABLE VS UNSTABLE)",
  "GCS AT EMERGENCY DEPARTMENT",
  "PUPIL STATUS ",
  "TYPE(S) OF INJURY (CT SCAN)",
  "INTRAPARECHYMAL",
  "INTRAPARECHYMAL SIZE (MM)",
  "INTRAVENTRICULAR",
  "SUBDURAL ",
  "SUBDURAL (SIZE MM)",
  "SAH ",
  "EPIDURAL (SIZE)",
  "BRAIN STEM",
  "CEREBELLAR",
  "CRANIAL BONE INJURY ",
  "CERVICAL SPINE INJURY",
  "PATIENT INTUBATED  EMERGENT (YES/NO)",
  "PATIENT OPERATED (YES/NO)",
  "SIGNIFICANT MEDICAL HISTORY (DM, MI, STROKE, CANCER, HF, CKD)",
  "Glc 1",
  "NLR",
  "PLR",
  "SII",
  "ICU CARE",
  "DAYS IN ICU",  # TODO: // vs - ? what to replace with?
  # input and/or outputs
  # "CRANIOTOMY",
  # "CRANIECTOMY",
]

INPUT_CATEGORICAL_COLUMNS = [
  "MECHANISM OF INJURY",
  "ISOLATED VS MULTI TRAUMA",
  "PUPIL STATUS ",
  "SIGNIFICANT MEDICAL HISTORY (DM, MI, STROKE, CANCER, HF, CKD)",
]

INPUT_BOOLEAN_COLUMNS = [
  "INTRAPARECHYMAL",
  "INTRAVENTRICULAR",
  "HEMODYNAMICAL STATUS (STABLE VS UNSTABLE)",
  "SUBDURAL ",
  "SAH ",
  "BRAIN STEM",
  "CEREBELLAR",
  "CRANIAL BONE INJURY ",
  "CERVICAL SPINE INJURY",
  "PATIENT INTUBATED  EMERGENT (YES/NO)",
  "PATIENT OPERATED (YES/NO)",
  "ICU CARE",
]

BOOLEAN_MAPPING = {
  "Yes": 1,
  "No": 0,
  "ISOLATED": 0,
  "MULTI": 1,
  "STABLE": 0,
  "UNSTABLE": 1,
}

# df["ventilated"] = df["ventilated"].map({"No":0, "Yes":1})

# TODO: '//' if INF (outcome is irrelevant)
OUTPUT_COLUMNS = [
  "GOS 15 DAYS ",
  "GOS 1 MONTH",
  "GOS 2 MONTHS",
  "GOS 6 MONTS",
  "OUTCOME AT DISCHARGE ",
]

# classification tree ground truth
OUTPUT_CATEGORICAL_COLUMNS = [
  "OUTCOME AT DISCHARGE ",
]

# regression ground truth
OUTPUT_REGRESSION_COLUMNS = [col for col in OUTPUT_COLUMNS if col not in OUTPUT_CATEGORICAL_COLUMNS]

CLASSIFIERS = {
  "DecisionTree": (
    DecisionTreeClassifier,
    {
      "max_depth": [3,5,7,10]
    }
  ),
  "RandomForest": (
    RandomForestClassifier,
    {
      "n_estimators": [100,200],
      "max_depth": [5,10,None]
    }
  ),
  "KNN": (
    KNeighborsClassifier,
    {
      "n_neighbors": [3,5,7]
    }
  ),
  "SVM": (
    SVC,
    {
      "C": [0.1,1,10],
      "kernel": ["linear","rbf"],
      "probability": [True]
    }
  ),
  "AdaBoost": (
    AdaBoostClassifier,
    {
      "n_estimators": [50,100]
    }
  ),
  "NaiveBayes": (
    GaussianNB,
    {}
  ),

  "MLP": (
    MLPClassifier,
    {
      "hidden_layer_sizes": [(32,),(64,)],
      "max_iter": [500]
    }
  )
}

def split_labels(x):
  if pd.isna(x):
    return []
  x = str(x).replace('+', ',')
  return [i.strip() for i in x.split(',')]

def preprocess_data(df):
  # cut out unecessary rows
  df = df.iloc[:169]
  df = df.dropna(subset=OUTPUT_COLUMNS)

  # clean excel artifacts
  df = df.replace({
    r"^\s*-\s*$": -1,
    r"^\s*//\s*$": -1,
    "#DIV/0!": np.nan,
    "#VALUE!": np.nan
  }, regex=True)

  # handle boolean
  for col in INPUT_BOOLEAN_COLUMNS:
    df[col] = df[col].astype(str).str.strip().map(BOOLEAN_MAPPING)

  # replace '-' and '//' values
  df = df.replace({
    '-': -1,
    '//': -1,
    '--': -1
  })

  # handle multi-category
  labels = df["TYPE(S) OF INJURY (CT SCAN)"].apply(split_labels)
  mlb = MultiLabelBinarizer()
  injury_matrix = mlb.fit_transform(labels)
  injury_df = pd.DataFrame(
    injury_matrix,
    columns=[f"INJURY_{c}" for c in mlb.classes_],
    index=df.index
  )
  df = df.drop(columns=["TYPE(S) OF INJURY (CT SCAN)"])
  df = pd.concat([df, injury_df], axis=1)
  INPUT_COLUMNS.remove("TYPE(S) OF INJURY (CT SCAN)")
  INPUT_COLUMNS.extend(injury_df.columns.tolist())

  X = df[INPUT_COLUMNS].copy()
  y = df[OUTPUT_COLUMNS].copy()

  # handle categorical (one-hot encode)
  X = pd.get_dummies(X, columns=INPUT_CATEGORICAL_COLUMNS)
  y = pd.get_dummies(y, columns=OUTPUT_CATEGORICAL_COLUMNS)

  # handle numeric
  # FIXME: some empty nan values should be -1 (check which columns)
  # X = X.fillna(X.median())  # TODO: tempfix - fill numeric NaNs
  X = X.fillna(-1)  # TODO: tempfix - fill numeric NaNs

  # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
  #   print(X)
  #   print(y)
  # print(X)
  # print(y)

  y_class = df[OUTPUT_CATEGORICAL_COLUMNS].copy()
  y_class = y_class.apply(lambda col: col.map({"ALIVE": 1, "DEAD": 0}))
  y_reg = df[OUTPUT_REGRESSION_COLUMNS].copy()

  return X, y_class, y_reg

def log(msg):
  print(msg)
  with open(LOG_FILE,"a") as f:
    f.write(msg+"\n")

def compute_metrics(y_true, y_pred, y_prob):
  acc = accuracy_score(y_true, y_pred)
  prec = precision_score(y_true, y_pred, zero_division=0)
  rec = recall_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred)

  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  specificity = tn / (tn + fp + 1e-8)

  auc = roc_auc_score(y_true,y_prob)

  return {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "specificity": specificity,
    "f1": f1,
    "auc": auc
  }

# model plots for explainability
def plot_decision_tree(model, feature_names, name):
  try:
    plt.figure(figsize=(20,10))
    plot_tree(
      model,
      feature_names=feature_names,
      class_names=["DEAD", "ALIVE"],
      filled=True,
      rounded=True,
      fontsize=6
    )
    plt.title(f"{name} Tree")
    plt.savefig(f"{PLOT_DIR}/{name}_tree.png", bbox_inches="tight")
    plt.close()
  except Exception as e:
    log(f"[!] Tree plot failed for {name}: {e}")

def plot_random_forest(model, feature_names, name, n_trees=3):
  try:
    for i, tree in enumerate(model.estimators_[:n_trees]):
      plt.figure(figsize=(20,10))
      plot_tree(
        tree,
        feature_names=feature_names,
        class_names=["DEAD", "ALIVE"],
        filled=True,
        rounded=True,
        fontsize=6
      )
      plt.title(f"{name} Tree {i}")
      plt.savefig(f"{PLOT_DIR}/{name}_tree_{i}.png", bbox_inches="tight")
      plt.close()
  except Exception as e:
    log(f"[!] RF plot failed for {name}: {e}")

def plot_feature_importance(model, X, name):
  try:
    if hasattr(model, "feature_importances_"):
      importances = model.feature_importances_
    elif hasattr(model, "coef_"):
      importances = np.abs(model.coef_).flatten()
    else:
      return

    idx = np.argsort(importances)[-20:]  # top 20

    plt.figure(figsize=(10,6))
    plt.barh(range(len(idx)), importances[idx])
    plt.yticks(range(len(idx)), X.columns[idx])
    plt.title(f"{name} Feature Importance")
    plt.savefig(f"{PLOT_DIR}/{name}_importance.png")
    plt.close()
  except Exception as e:
    log(f"[!] Feature importance failed for {name}: {e}")

def explain_knn(model, X, sample, name):
  try:
    distances, indices = model.kneighbors(sample)
    log(f"[KNN] Neighbors indices: {indices}")
    log(f"[KNN] Distances: {distances}")
  except Exception as e:
    log(f"[!] KNN explain failed: {e}")

# evaluation metrics
def confidence_interval(scores):
  scores = np.array(scores)
  mean = scores.mean()
  std = scores.std()
  ci = 1.96*std/np.sqrt(len(scores))
  return mean, ci

def plot_confusion(cm, name):
  plt.figure()
  plt.imshow(cm)
  plt.title(name)
  plt.colorbar()
  plt.xlabel("Predicted")
  plt.ylabel("True")
  plt.savefig(f"{PLOT_DIR}/{name}_confusion.png")
  plt.close()

def plot_roc(y_true, y_prob, name):
  fpr,tpr,_ = roc_curve(y_true, y_prob)

  plt.figure()
  plt.plot(fpr, tpr, label=name)
  plt.plot([0,1],[0,1],"--")
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("ROC Curve")
  plt.legend()

  plt.savefig(f"{PLOT_DIR}/{name}_roc.png")
  plt.close()

def compute_shap(model, X, name):
  try:
    explainer = shap.Explainer(model,X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(f"{PLOT_DIR}/{name}_shap.png")
    plt.close()
  except Exception as e:
    log(f"[!] SHAP failed for {name}: {e}")

def benchmark_classifiers(X, y):
  results = []
  class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y.values.ravel())
  weight_dict = {c: w for c, w in zip(np.unique(y), class_weights)}

  for name, (Model, grid) in CLASSIFIERS.items():
    log(f"\n[clf] {name}")
    params_list = list(ParameterGrid(grid)) if grid else [{}]
    best_model = None
    best_auc = -1
    for params in tqdm(params_list):
      if "class_weight" in Model().get_params().keys():
        params["class_weight"] = weight_dict

      model = Model(**params)
      skf = StratifiedKFold(KFOLD,shuffle=True,random_state=RANDOM_STATE)
      auc_scores = []
      for train_idx, val_idx in skf.split(X, y):
        X_train,X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train,y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train.values.ravel())
        probs = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, probs)
        auc_scores.append(auc)

      mean_auc, _ = confidence_interval(auc_scores)
      if mean_auc > best_auc:
        best_auc = mean_auc
        best_model = Model(**params)

    log(f"[+] best auc {best_auc:.4f}")

    best_model.fit(X, y.values.ravel())
    # Explainability before calibration
    if name == "DecisionTree":
      plot_decision_tree(best_model, X.columns, name)
    elif name == "RandomForest":
      plot_random_forest(best_model, X.columns, name)
    plot_feature_importance(best_model, X, name)  # General importance

    # Calibration
    calibrated = CalibratedClassifierCV(best_model, method="sigmoid")
    calibrated.fit(X, y.values.ravel())
    joblib.dump(calibrated, f"{CKPT_DIR}/{name}.joblib")
    log("[+] model saved")

    preds = calibrated.predict(X)
    probs = calibrated.predict_proba(X)[:, 1]

    metrics = compute_metrics(y, preds, probs)
    cm = confusion_matrix(y, preds)
    plot_confusion(cm, name)
    plot_roc(y, probs, name)
    compute_shap(calibrated, X, name)

    for k, v in metrics.items():
      writer.add_scalar(f"{name}/{k}", v)

    results.append((name,metrics))
  return results

def train_regressor(X_train,y_train,X_test,y_test):
  reg = DecisionTreeRegressor(max_depth=MAX_TREE_DEPTH, random_state=RANDOM_STATE)
  reg.fit(X_train, y_train)
  score = reg.score(X_test, y_test)
  log(f"[reg] R_2 {score:.4f}")
  joblib.dump(reg,f"{CKPT_DIR}/regressor.joblib")
  return reg


if __name__=="__main__":
  log("[*] loading dataset")
  df = pd.read_excel("../data/TBI DAY ONE ONLY LABS.xlsx", sheet_name="Φύλλο1")
  log(f"[+] shape {df.shape}")

  X, y_class, y_reg = preprocess_data(df)
  X_train,X_test,y_train,y_test = train_test_split(X, y_class, test_size=TEST_SIZE, random_state=RANDOM_STATE)

  log("[*] running classifier benchmark")
  results = benchmark_classifiers(X_train, y_train)
  log("\n[+] benchmark summary")
  for name, metrics in results:
    log(f"{name} {json.dumps(metrics, indent=2)}")

  reg = train_regressor(X_train, y_reg.iloc[X_train.index], X_test, y_reg.iloc[X_test.index])

  log("\n[*] example prognosis")
  sample = X_test.iloc[[0]]
  for name, _ in results:
    model = joblib.load(f"{CKPT_DIR}/{name}.joblib")
    prob = model.predict_proba(sample)[0][1]
    pred = model.predict(sample)[0]
    log(f"[clf] {name} -> {pred} ({prob*100:.2f}% survival probability)")

  writer.close()
  log("[+] DONE")
