import os
import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMClassifier


# SUBJECTS = ["Maths", "Science", "English", "Social", "Computer"]
SUBJECTS = ["Math", "Science", "English", "Social_Studies", "Computer_Science"]
FEATURES = ["Class"] + SUBJECTS + ["Average"]


def ensure_models_dir(models_dir: str):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)


def load_dataset(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    return df


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, float]]:
    # Drop unnecessary columns if present
    drop_cols = [c for c in ["Student_ID", "Name"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Coerce types
    for col in ["Class"] + SUBJECTS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute Average
    df["Average"] = df[SUBJECTS].mean(axis=1)

    # Derive Result: Pass (>=40 in all subjects & Average >= 40), else Fail
    pass_all = (df[SUBJECTS] >= 40).all(axis=1)
    pass_avg = df["Average"] >= 40
    df["Result"] = np.where(pass_all & pass_avg, "Pass", "Fail")

    # Handle missing values (numeric columns): fill with mean
    feature_df = df[["Class"] + SUBJECTS + ["Average"]].copy()
    feature_means = {col: float(feature_df[col].mean()) for col in feature_df.columns}
    for col in feature_df.columns:
        feature_df[col] = feature_df[col].fillna(feature_means[col])

    # Target
    y = df["Result"].copy()

    return feature_df, y, feature_means


def train_test_split_custom(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
):
    """
    Simple custom stratified split without scikit-learn.
    """
    rng = np.random.default_rng(seed=random_state)

    # Stratify by y
    classes = y.unique().tolist()
    train_idx = []
    test_idx = []

    for cls in classes:
        cls_idx = np.where(y.values == cls)[0]
        rng.shuffle(cls_idx)
        n_test = max(1, int(len(cls_idx) * test_size))
        test_idx.extend(cls_idx[:n_test].tolist())
        train_idx.extend(cls_idx[n_test:].tolist())

    # In case rounding caused issues, ensure unique and cover all
    test_idx = list(sorted(set(test_idx)))
    train_idx = list(sorted(set([i for i in range(len(y)) if i not in test_idx])))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute accuracy, precision, recall, f1, confusion matrix without scikit-learn.
    Binary labels expected: 'Pass'/'Fail' or 1/0. We'll map to 1=Pass, 0=Fail for metrics.
    """
    # Map strings to 0/1
    def to01(arr):
        if arr.dtype.kind in {"U", "S", "O"}:
            return np.array([1 if v == "Pass" else 0 for v in arr], dtype=int)
        return arr.astype(int)

    yt = to01(np.array(y_true))
    yp = to01(np.array(y_pred))

    tp = int(np.sum((yp == 1) & (yt == 1)))
    tn = int(np.sum((yp == 0) & (yt == 0)))
    fp = int(np.sum((yp == 1) & (yt == 0)))
    fn = int(np.sum((yp == 0) & (yt == 1)))

    total = max(1, len(yt))
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def train_and_save_model(
    data_path: str,
    model_out_path: str,
    metrics_out_path: str = None
) -> Dict:
    df = load_dataset(data_path)
    X, y, feature_means = preprocess(df)

    # Map target to integers for LightGBM
    label_to_int = {"Fail": 0, "Pass": 1}
    int_to_label = {0: "Fail", 1: "Pass"}
    y_int = y.map(label_to_int).astype(int)

    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)
    y_train_int = y_train.map(label_to_int).astype(int)
    y_test_int = y_test.map(label_to_int).astype(int)

    # LightGBM parameters per spec
    model = LGBMClassifier(
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.05,
        n_estimators=200,
        random_state=42,
    )

    model.fit(X_train.values, y_train_int.values)
    y_pred_int = model.predict(X_test.values)

    # Convert back to labels for human-readable metrics
    y_pred = pd.Series([int_to_label[int(v)] for v in y_pred_int], index=y_test.index)
    metrics = compute_metrics(y_test.values, y_pred.values)

    # Persist
    bundle = {
        "model": model,
        "feature_names": FEATURES,
        "feature_means": feature_means,
        "label_map": {0: "Fail", 1: "Pass"},
    }
    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    joblib.dump(bundle, model_out_path)

    if metrics_out_path:
        with open(metrics_out_path, "w") as f:
            json.dump(metrics, f, indent=2)

    # Print metrics for script visibility
    print("[TRAIN] Metrics:", json.dumps(metrics, indent=2))
    print(f"[TRAIN] Model saved to: {model_out_path}")

    return bundle


if __name__ == "__main__":
    # Example local training
    APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(APP_DIR, "data", "student_performance_classwise.csv")
    models_dir = os.path.join(APP_DIR, "models")
    ensure_models_dir(models_dir)
    model_out = os.path.join(models_dir, "student_model.pkl")
    metrics_out = os.path.join(models_dir, "metrics.json")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Add 'student_performance_classwise.csv' first."
        )

    train_and_save_model(data_path, model_out, metrics_out)
