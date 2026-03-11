import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

# FIXME: cleanup incosistent cells manually

INF = float("inf")

# Hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_TREE_DEPTH = 5

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
  y_reg = df[OUTPUT_REGRESSION_COLUMNS].copy()

  return X, y_class, y_reg

# TODO: train model ensembles
# names = [
#     "Nearest Neighbors",
#     "Linear SVM",
#     "RBF SVM",
#     "Gaussian Process",
#     "Decision Tree",
#     "Random Forest",
#     "Neural Net",
#     "AdaBoost",
#     "Naive Bayes",
#     "QDA",
# ]
# TODO: class inbalances => cost based learning
def train_classifier(X_train, y_train, X_test, y_test):
  clf = DecisionTreeClassifier(max_depth=MAX_TREE_DEPTH, random_state=RANDOM_STATE)
  clf.fit(X_train, y_train)
  print("[+] Classification accuracy: {:.2f}%".format(clf.score(X_test, y_test) * 100))
  return clf

def train_regressor(X_train, y_train, X_test, y_test):
  reg = DecisionTreeRegressor(max_depth=MAX_TREE_DEPTH, random_state=RANDOM_STATE)
  reg.fit(X_train, y_train)
  print("[+] Regression R^2 score: {:.2f}".format(reg.score(X_test, y_test)))
  return reg


if __name__ == "__main__":
  # get data
  df = pd.read_excel("../data/TBI DAY ONE ONLY LABS.xlsx", sheet_name="Φύλλο1")
  print("[+] Loaded dataset with shape:", df.shape)

  # preprocess
  X, y_class, y_reg = preprocess_data(df)
  print("[+] Data preprocessed")
  X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=TEST_SIZE, random_state=RANDOM_STATE)
  _, _, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE)
  print("[+] Train/test split extracted")

  print("[*] Training models ...")
  clf = train_classifier(X_train, y_class_train, X_test, y_class_test)
  reg = train_regressor(X_train, y_reg_train, X_test, y_reg_test)
  print(clf.classes_)
  
  # visualize
  sample = X_test.iloc[[0]]

  print("Classification prediction:")
  print(clf.predict(sample))
  print("Regression prediction:")
  print(np.round(reg.predict(sample)).astype(int))

  plt.figure(figsize=(20,10))
  plot_tree(clf, feature_names=X.columns, filled=True, class_names=clf.classes_)
  plt.show()

  print("[+] DONE")