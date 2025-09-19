import os
import io
import json
import base64
from typing import Dict, List, Tuple

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(APP_DIR, "data", "student_performance_classwise.csv")
MODELS_DIR = os.path.join(APP_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "student_model.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

SUBJECTS = ["Math", "Science", "English", "Social_Studies", "Computer_Science"]
FEATURES = ["Class"] + SUBJECTS + ["Average"]

# Form field mapping: form input names -> canonical subject names
SUBJECT_FORM_NAMES = {
    "Math": "maths",
    "Science": "science",
    "English": "english",
    "Social_Studies": "social",
    "Computer_Science": "computer",
}

# Optional: pretty display labels for charts/UI
DISPLAY_LABELS = {
    "Math": "Math",
    "Science": "Science",
    "English": "English",
    "Social_Studies": "Social Studies",
    "Computer_Science": "Computer Science",
}

# Synonyms for subjects to handle varying column names in CSV
SUBJECT_SYNONYMS = {
    "Math": ["Math", "Maths", "Mathematics"],
    "Science": ["Science"],
    "English": ["English"],
    "Social_Studies": ["Social_Studies", "Social", "Social Studies"],
    "Computer_Science": ["Computer_Science", "Computer", "Computer Science", "CS"],
}

app = Flask(__name__)

# Lazy imports from our training module to avoid circularity
try:
    from scripts.train_model import train_and_save_model, ensure_models_dir, load_dataset
except Exception as e:
    # If for some reason local import fails (shouldn't), provide readable error at runtime
    train_and_save_model = None
    ensure_models_dir = None
    load_dataset = None

UPLOADS_DIR = os.path.join(APP_DIR, "uploads")
CURRENT_CSV_PATH = None
CURRENT_CSV_DF: pd.DataFrame | None = None

def ensure_uploads_dir():
    os.makedirs(UPLOADS_DIR, exist_ok=True)

def get_active_dataset():
    # Prefer uploaded CSV for CSV flows; fallback to training dataset DF_DATA
    return CURRENT_CSV_DF if CURRENT_CSV_DF is not None else DF_DATA

def plot_subject_distributions(marks: Dict[str, float], df: pd.DataFrame, class_value: int | float | None) -> str:
    # Build a 2x3 grid of subject distributions for the student's class; last cell empty
    if df is None or df.empty:
        # Return a blank image indicating unavailable distributions
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No distributions available", ha="center", va="center", fontsize=12)
        return figure_to_base64(fig)

    df_local = df.copy()
    # Coerce numerics
    for c in ["Class"] + SUBJECTS:
        if c in df_local.columns:
            df_local[c] = pd.to_numeric(df_local[c], errors="coerce")

    if class_value is not None:
        df_local = df_local[df_local["Class"] == float(class_value)]
    # If class filter yields empty, keep full dataset to avoid empty plots
    if df_local.empty:
        df_local = df.copy()
        for c in SUBJECTS:
            if c in df_local.columns:
                df_local[c] = pd.to_numeric(df_local[c], errors="coerce")

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()
    for idx, sub in enumerate(SUBJECTS):
        ax = axes[idx]
        if sub in df_local.columns:
            vals = pd.to_numeric(df_local[sub], errors="coerce").dropna()
            ax.hist(vals, bins=10, color="#6c757d", edgecolor="white")
            # student mark line
            try:
                v = float(marks.get(sub, np.nan))
                if not pd.isna(v):
                    ax.axvline(v, color="#0d6efd", linestyle="--", linewidth=2)
            except Exception:
                pass
            ax.set_title(DISPLAY_LABELS.get(sub, sub), fontsize=10)
            ax.set_xlim(0, 100)
        else:
            ax.axis("off")
    # hide last unused subplot if subjects < 6
    if len(SUBJECTS) < len(axes):
        axes[-1].axis("off")
    plt.tight_layout()
    return figure_to_base64(fig)


def load_model() -> Tuple[Dict, pd.DataFrame]:
    """
    Loads the trained model dict and the dataset used for class averages/visualizations.
    If the model file is missing, attempts to train it automatically using the local dataset.
    """
    # Ensure model directory exists
    if ensure_models_dir:
        ensure_models_dir(MODELS_DIR)

    # Auto-train if not found and training module is available
    if not os.path.exists(MODEL_PATH):
        if train_and_save_model is None:
            raise RuntimeError("Model not found and training module unavailable.")
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(
                f"Dataset not found at {DATA_PATH}. Please add 'student_performance_classwise.csv'."
            )
        train_and_save_model(
            data_path=DATA_PATH,
            model_out_path=MODEL_PATH,
            metrics_out_path=METRICS_PATH
        )

    model_bundle = joblib.load(MODEL_PATH)
    # Load dataset for visualization and class averages
    if os.path.exists(DATA_PATH) and load_dataset:
        df = load_dataset(DATA_PATH)
    else:
        # Fallback empty dataframe with required columns
        df = pd.DataFrame(columns=["Class"] + SUBJECTS)

    return model_bundle, df


MODEL_BUNDLE, DF_DATA = load_model()


def subjects_below_threshold(marks: Dict[str, float], threshold: float = 40.0) -> List[str]:
    return [sub for sub, val in marks.items() if val is not None and float(val) < threshold]


def figure_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def normalize_csv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize incoming CSV to ensure canonical SUBJECTS and 'Class' exist with numeric values.
    Does not drop original columns; adds canonical ones when needed.
    """
    if df is None or df.empty:
        return df

    # Build a case-insensitive column lookup
    col_lookup = {str(c).strip().lower(): c for c in df.columns}

    # Ensure Class exists and is numeric if present
    if "class" in col_lookup:
        c = col_lookup["class"]
        df["Class"] = pd.to_numeric(df[c], errors="coerce")

    # Create/standardize each canonical subject column
    for canon, options in SUBJECT_SYNONYMS.items():
        found = None
        for opt in options:
            key = str(opt).strip().lower()
            if key in col_lookup:
                found = col_lookup[key]
                break
        if found is not None:
            df[canon] = pd.to_numeric(df[found], errors="coerce")
        else:
            # Create empty numeric column if missing
            df[canon] = np.nan

    return df


def class_subject_averages(df: pd.DataFrame, class_value: int) -> Dict[str, float]:
    if df.empty:
        return {s: 0.0 for s in SUBJECTS}
    # Robust filter: Class might be string or numeric
    df_local = df.copy()
    try:
        df_local["Class"] = pd.to_numeric(df_local["Class"], errors="coerce")
    except Exception:
        pass
    class_df = df_local[df_local["Class"] == float(class_value)]
    if class_df.empty:
        return {s: 0.0 for s in SUBJECTS}
    # Guard against missing columns (after normalization they should exist, but stay safe)
    out: Dict[str, float] = {}
    for s in SUBJECTS:
        if s in class_df.columns:
            vals = pd.to_numeric(class_df[s], errors="coerce")
            out[s] = float(np.nanmean(vals)) if len(vals) else 0.0
        else:
            out[s] = 0.0
    return out


def plot_student_subjects(marks: Dict[str, float]) -> str:
    subjects = list(marks.keys())
    values = [float(marks[s]) for s in subjects]

    labels = [DISPLAY_LABELS.get(s, s) for s in subjects]

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color="#0d6efd")
    ax.set_ylim(0, 100)
    ax.set_title("Subject-wise Marks", fontsize=12)
    ax.set_ylabel("Marks")
    ax.bar_label(bars, fmt="%.0f", padding=3)
    return figure_to_base64(fig)


def plot_student_vs_class_avg(marks: Dict[str, float], class_avgs: Dict[str, float]) -> str:
    subjects = list(marks.keys())
    student_vals = [float(marks[s]) for s in subjects]
    class_vals = [float(class_avgs[s]) for s in subjects]

    labels = [DISPLAY_LABELS.get(s, s) for s in subjects]

    x = np.arange(len(subjects))
    width = 0.35

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, student_vals, width, label="Student", color="#0d6efd")
    ax.bar(x + width / 2, class_vals, width, label="Class Avg", color="#6c757d")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_title("Student vs Class Average", fontsize=12)
    ax.set_ylabel("Marks")
    ax.legend()
    return figure_to_base64(fig)


@app.route("/", methods=["GET"])
def home():
    return render_template("hero.html")

@app.route("/hero", methods=["GET"])
def hero():
    return render_template("hero.html")

@app.route("/csv", methods=["GET", "POST"])
def csv_page():
    global CURRENT_CSV_PATH, CURRENT_CSV_DF
    ensure_uploads_dir()
    message = None

    if request.method == "POST":
        file = request.files.get("csv_file")
        if not file or file.filename == "":
            message = "Please select a CSV file."
        else:
            if not file.filename.lower().endswith(".csv"):
                message = "Only .csv files are supported."
            else:
                save_path = os.path.join(UPLOADS_DIR, "uploaded.csv")
                file.save(save_path)
                try:
                    df = pd.read_csv(save_path, encoding="utf-8")
                    # Normalize uploaded CSV to canonical subject columns to prevent KeyError
                    df = normalize_csv_columns(df)
                    CURRENT_CSV_PATH = save_path
                    CURRENT_CSV_DF = df
                    message = "CSV uploaded successfully."
                except Exception as e:
                    message = f"Failed to read CSV: {e}"

    # Prepare context
    has_dataset = CURRENT_CSV_DF is not None and not CURRENT_CSV_DF.empty
    students = []
    if has_dataset:
        df = CURRENT_CSV_DF
        # Try to coerce Class to string for display
        cls = df.get("Class")
        for _, row in df.iterrows():
            sid = str(row.get("Student_ID", ""))
            name = str(row.get("Name", ""))
            sclass = str(row.get("Class", ""))
            students.append({"id": sid, "name": name, "cls": sclass})

    return render_template(
        "csv.html",
        message=message,
        has_dataset=has_dataset,
        students=students,
        filename=os.path.basename(CURRENT_CSV_PATH) if CURRENT_CSV_PATH else None
    )

@app.route("/csv/clear", methods=["POST"])
def csv_clear():
    global CURRENT_CSV_PATH, CURRENT_CSV_DF
    if CURRENT_CSV_PATH and os.path.exists(CURRENT_CSV_PATH):
        try:
            os.remove(CURRENT_CSV_PATH)
        except Exception:
            pass
    CURRENT_CSV_PATH = None
    CURRENT_CSV_DF = None
    return render_template(
        "csv.html",
        message="Cleared uploaded CSV.",
        has_dataset=False,
        students=[],
        filename=None
    )

@app.route("/csv/student/<student_id>", methods=["GET"])
def csv_student_analytics(student_id):
    global CURRENT_CSV_DF
    if CURRENT_CSV_DF is None or CURRENT_CSV_DF.empty:
        # No CSV loaded; redirect to CSV page
        return render_template(
            "csv.html",
            message="Please upload a CSV first.",
            has_dataset=False,
            students=[],
            filename=None
        )

    df = CURRENT_CSV_DF
    # Find the row by Student_ID (string compare)
    row = df[df["Student_ID"].astype(str) == str(student_id)]
    if row.empty:
        return render_template(
            "csv.html",
            message=f"Student with ID {student_id} not found in uploaded CSV.",
            has_dataset=True,
            students=[
                {"id": str(r.get("Student_ID", "")), "name": str(r.get("Name", "")), "cls": str(r.get("Class", ""))}
                for _, r in df.iterrows()
            ],
            filename=os.path.basename(CURRENT_CSV_PATH) if CURRENT_CSV_PATH else None
        )
    r = row.iloc[0]

    # Extract class and marks according to canonical SUBJECTS
    try:
        class_level = int(pd.to_numeric(r.get("Class"), errors="coerce"))
    except Exception:
        class_level = 0

    marks = {}
    for s in SUBJECTS:
        try:
            marks[s] = float(pd.to_numeric(r.get(s), errors="coerce"))
        except Exception:
            marks[s] = np.nan

    # Compute Average
    avg = float(np.nanmean(list(marks.values()))) if len(marks) > 0 else 0.0

    # Align features with the trained model
    means_dict = MODEL_BUNDLE.get("feature_means", {})
    feature_names = MODEL_BUNDLE.get("feature_names", FEATURES)
    feats_aligned = []
    for fname in feature_names:
        if fname == "Class":
            feats_aligned.append(class_level if not pd.isna(class_level) else means_dict.get(fname, 0.0))
        elif fname == "Average":
            feats_aligned.append(avg if not pd.isna(avg) else means_dict.get(fname, 0.0))
        else:
            val = marks.get(fname, np.nan)
            feats_aligned.append(means_dict.get(fname, 0.0) if pd.isna(val) else val)

    X = np.array(feats_aligned, dtype=float).reshape(1, -1)
    model = MODEL_BUNDLE["model"]
    label_map = MODEL_BUNDLE["label_map"]
    y_pred = model.predict(X)
    result_label = label_map.get(int(y_pred[0]), "Unknown")

    weak = subjects_below_threshold(marks, threshold=40.0)

    # Use uploaded CSV for class averages and distributions
    class_avgs = class_subject_averages(CURRENT_CSV_DF, class_level)
    subjects_plot = plot_student_subjects(marks)
    compare_plot = plot_student_vs_class_avg(marks, class_avgs)
    insights = generate_insights(marks, class_avgs, avg, result_label)

    return render_template(
        "result.html",
        result=result_label,
        average=f"{avg:.2f}",
        weak_subjects=weak,
        subjects=SUBJECTS,
        marks=marks,
        class_level=class_level,
        subjects_plot=subjects_plot,
        compare_plot=compare_plot,
        insights=insights,
    )

@app.route("/predict", methods=["POST"])
def predict():
    # Collect form inputs
    try:
        class_level = int(request.form.get("class", "").strip())
    except Exception:
        class_level = 0

    marks = {}
    for s in SUBJECTS:
        field = SUBJECT_FORM_NAMES.get(s, s.lower())
        val_raw = request.form.get(field, "")
        val = val_raw.strip() if isinstance(val_raw, str) else val_raw
        try:
            marks[s] = float(val)
        except Exception:
            marks[s] = np.nan

    # Compute Average
    avg = float(np.nanmean(list(marks.values()))) if len(marks) > 0 else 0.0

    # Prepare feature vector using canonical order; align to model's feature_names
    means_dict = MODEL_BUNDLE.get("feature_means", {})
    feature_names = MODEL_BUNDLE.get("feature_names", FEATURES)

    feats_aligned = []
    for fname in feature_names:
        if fname == "Class":
            feats_aligned.append(class_level if not pd.isna(class_level) else means_dict.get(fname, 0.0))
        elif fname == "Average":
            feats_aligned.append(avg if not pd.isna(avg) else means_dict.get(fname, 0.0))
        else:
            val = marks.get(fname, np.nan)
            feats_aligned.append(means_dict.get(fname, 0.0) if pd.isna(val) else val)

    X = np.array(feats_aligned, dtype=float).reshape(1, -1)
    model = MODEL_BUNDLE["model"]
    label_map = MODEL_BUNDLE["label_map"]

    # Predict
    y_pred = model.predict(X)
    result_label = label_map.get(int(y_pred[0]), "Unknown")

    # Weak subjects (below 40)
    weak = subjects_below_threshold(marks, threshold=40.0)

    # Visualizations (class averages from training data)
    class_avgs = class_subject_averages(DF_DATA, class_level)
    subjects_plot = plot_student_subjects(marks)
    compare_plot = plot_student_vs_class_avg(marks, class_avgs)
    insights = generate_insights(marks, class_avgs, avg, result_label)

    return render_template(
        "result.html",
        result=result_label,
        average=f"{avg:.2f}",
        weak_subjects=weak,
        subjects=SUBJECTS,
        marks=marks,
        class_level=class_level,
        subjects_plot=subjects_plot,
        compare_plot=compare_plot,
        insights=insights,
    )

def generate_insights(
    marks: Dict[str, float],
    class_avgs: Dict[str, float],
    average: float,
    result_label: str,
    pass_threshold: float = 40.0,
) -> List[str]:
    insights: List[str] = []

    if isinstance(result_label, str) and result_label.lower() == "fail":
        insights.append("Overall status indicates risk of failing. Raise weak subjects to at least 40 to pass.")

    # Subject-specific improvements to reach pass threshold
    weak_subjects_local = []
    for s, v in marks.items():
        try:
            score = float(v)
        except Exception:
            score = float("nan")
        if not pd.isna(score) and score < pass_threshold:
            weak_subjects_local.append(s)
            gap = pass_threshold - score
            tip = f"{DISPLAY_LABELS.get(s, s)}: increase by +{gap:.0f} to reach {int(pass_threshold)}."
            # add comparison to class average if available
            ca = class_avgs.get(s, None)
            try:
                ca_val = float(ca)
            except Exception:
                ca_val = float("nan")
            if not pd.isna(ca_val):
                diff = ca_val - score
                if diff > 0:
                    tip += f" You are {diff:.0f} below class average."
            insights.append(tip)

    if not weak_subjects_local:
        insights.append("No subjects below 40 detected. Maintain consistency and aim for 60+ across all subjects.")

    # Identify focus areas vs class average (lowest relative difference)
    diffs = []
    for s, v in marks.items():
        try:
            score = float(v)
        except Exception:
            score = float("nan")
        ca = class_avgs.get(s, None)
        try:
            ca_val = float(ca)
        except Exception:
            ca_val = float("nan")
        if not pd.isna(score) and not pd.isna(ca_val):
            diffs.append((s, score - ca_val))
    if diffs:
        diffs.sort(key=lambda x: x[1])  # most behind first
        focus = [DISPLAY_LABELS.get(s, s) for s, _ in diffs[:2]]
        if focus:
            insights.append(f"Focus areas compared to class: {', '.join(focus)}.")

    # Suggest study approach
    insights.append("Plan: 20–30 targeted practice problems per weak subject; review mistakes weekly and track progress.")

    return insights


if __name__ == "__main__":
    # Local run: python app.py
    app.run(host="0.0.0.0", port=5000, debug=True)
