"""
Explainable AI-Based Cyber Threat Detection System
Model training pipeline (CICIDS2017 + Random Forest + SHAP)

Run from the project folder (CSV files may be in the folder or its sub-folders):
    python train_model.py
Or point it at a folder:
    python train_model.py "C:\\path\\to\\csv_folder"

Outputs
    threat_model.pkl    trained Random Forest
    model_metrics.json  accuracy, per-class metrics, confusion matrix, SHAP importance, baselines
    demo_events.csv     held-out test rows that the dashboard replays as "live" events
"""

import glob
import json
import os
import re
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from features import CLASS_NAMES, FEATURE_NAMES, RAW_COLUMNS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "threat_model.pkl")
METRICS_FILE = os.path.join(BASE_DIR, "model_metrics.json")
DEMO_FILE = os.path.join(BASE_DIR, "demo_events.csv")

MAX_ROWS_PER_FILE = 40_000   # downsampling cap per daily capture file
RANDOM_STATE = 42
DEMO_PER_CLASS = 300         # held-out events saved for the dashboard replay
SHAP_SAMPLE = 200            # rows used for the global SHAP importance


def read_csv_safe(path):
    """CICIDS2017 files mix encodings; try UTF-8 first, then Latin-1."""
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, skipinitialspace=True, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError("unsupported file encoding")


def clean_label(label):
    """Normalise odd characters, e.g. 'Web Attack <?> Brute Force' -> 'Web Attack - Brute Force'."""
    return re.sub(r"[^\x20-\x7E]+", "-", str(label)).strip()


def map_label(label):
    if label == "BENIGN":
        return 0                                   # Normal Activity
    if "Patator" in label or "Brute" in label:
        return 1                                   # Brute Force (FTP/SSH-Patator, web brute force)
    if "PortScan" in label:
        return 2                                   # Port Scanning
    return 3                                       # DoS, DDoS, Botnet, other web attacks, infiltration...


def find_csv_files():
    """
    Locate the CICIDS2017 CSV files. Search order:
      1. a folder given on the command line:  python train_model.py "C:\\path\\to\\csv_folder"
      2. the project folder (and its sub-folders)
      3. the current working directory
    """
    skip_dirs = {"venv", ".venv", "env", "site-packages", "node_modules", ".git", "__pycache__"}
    roots = ([os.path.abspath(sys.argv[1])] if len(sys.argv) > 1 else []) + [BASE_DIR, os.getcwd()]
    searched = []
    for root in roots:
        if root in searched or not os.path.isdir(root):
            continue
        searched.append(root)
        found = [f for f in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
                 if os.path.basename(f) != os.path.basename(DEMO_FILE)
                 and not skip_dirs & set(os.path.normpath(f).split(os.sep))]
        if found:
            print(f" -> Searching in: {root}")
            return sorted(found)
    print("ERROR: No CSV files found. Folders searched:")
    for r in searched:
        print(f"   - {r}")
    print("Tip: run  python train_model.py \"<full path to the folder with the CICIDS2017 CSV files>\"")
    sys.exit(1)


def main():
    print("=" * 60)
    print("INITIATING AI MODEL TRAINING PIPELINE")
    print("=" * 60)

    # ------------------------------------------------------------------ 1. Ingest
    print("[1/6] Ingesting CICIDS2017 dataset...")
    csv_files = find_csv_files()
    print(f" -> Detected {len(csv_files)} dataset files.")

    required = list(RAW_COLUMNS) + ["Label"]
    frames = []
    for path in csv_files:
        name = os.path.basename(path)
        try:
            part = read_csv_safe(path)
            part.columns = part.columns.str.strip()
            missing = [c for c in required if c not in part.columns]
            if missing:
                print(f" -> WARNING: skipping {name} (missing columns: {missing})")
                continue
            part = part[required]
            if len(part) > MAX_ROWS_PER_FILE:
                part = part.sample(n=MAX_ROWS_PER_FILE, random_state=RANDOM_STATE)
            frames.append(part)
            print(f" -> Processed: {name} ({len(part):,} rows kept)")
        except Exception as exc:
            print(f" -> WARNING: skipping {name}. Error: {exc}")

    if not frames:
        sys.exit("ERROR: No usable dataset files.")

    df = pd.concat(frames, ignore_index=True).rename(columns=RAW_COLUMNS)
    print(f"[SUCCESS] Records aggregated: {len(df):,}")

    # ------------------------------------------------------------------ 2. Clean
    print("\n[2/6] Cleaning data...")
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_NAMES)
    df["Label"] = df["Label"].apply(clean_label)
    df = df.drop_duplicates(subset=FEATURE_NAMES + ["Label"]).reset_index(drop=True)
    print(f" -> Removed {before - len(df):,} invalid or duplicate rows; {len(df):,} remain.")

    # ------------------------------------------------------------------ 3. Labels
    print("[3/6] Applying label mapping...")
    df["Threat_Label"] = df["Label"].apply(map_label)
    for cid, count in df["Threat_Label"].value_counts().sort_index().items():
        print(f" -> {CLASS_NAMES[cid]:<20} {count:>9,}")

    # ------------------------------------------------------------------ 4. Train
    print("\n[4/6] Training Random Forest (stratified 80/20 split)...")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["Threat_Label"])

    model = RandomForestClassifier(
        n_estimators=100, max_depth=16, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE)
    start = time.time()
    model.fit(train_df[FEATURE_NAMES], train_df["Threat_Label"])
    train_seconds = time.time() - start

    # ------------------------------------------------------------------ 5. Evaluate
    print("\n[5/6] Evaluating on the held-out test set...")
    y_true = test_df["Threat_Label"]
    y_pred = model.predict(test_df[FEATURE_NAMES])
    accuracy = accuracy_score(y_true, y_pred)
    labels = list(CLASS_NAMES)
    names = [CLASS_NAMES[c] for c in labels]
    report = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                   output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    majority = float(y_true.value_counts(normalize=True).max())

    print("=" * 60)
    print(f"[RESULTS] Accuracy: {accuracy * 100:.2f}%   (majority-class baseline: {majority * 100:.2f}%)")
    print(f"[RESULTS] Macro F1: {report['macro avg']['f1-score']:.3f}   Training time: {train_seconds:.1f}s")
    print("=" * 60)
    print(classification_report(y_true, y_pred, labels=labels, target_names=names, zero_division=0))

    # ------------------------------------------------------------------ 6. Explain + export
    print("[6/6] Computing baselines, global SHAP importance and exporting artifacts...")
    benign = train_df[train_df["Threat_Label"] == 0][FEATURE_NAMES]
    baseline = {
        n: {"benign_median": float(benign[n].median()), "overall_median": float(train_df[n].median())}
        for n in FEATURE_NAMES
    }

    global_shap = {}
    try:
        sample = test_df[FEATURE_NAMES].sample(min(SHAP_SAMPLE, len(test_df)), random_state=RANDOM_STATE)
        values = shap.TreeExplainer(model).shap_values(sample, check_additivity=False)
        arr = np.stack(values, axis=-1) if isinstance(values, list) else np.asarray(values)
        mean_abs = np.abs(arr).mean(axis=tuple(i for i in range(arr.ndim) if i != 1))
        global_shap = {n: float(v) for n, v in zip(FEATURE_NAMES, mean_abs)}
    except Exception as exc:
        print(f" -> WARNING: global SHAP skipped ({exc})")

    demo_parts = []
    for cid in labels:
        pool = test_df[test_df["Threat_Label"] == cid]
        if len(pool):
            demo_parts.append(pool.sample(min(DEMO_PER_CLASS, len(pool)), random_state=RANDOM_STATE))
    demo = (pd.concat(demo_parts)[FEATURE_NAMES + ["Threat_Label", "Label"]]
            .rename(columns={"Threat_Label": "actual_class", "Label": "raw_label"}))
    demo.to_csv(DEMO_FILE, index=False)

    joblib.dump(model, MODEL_FILE, compress=3)

    metrics = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_seconds": round(train_seconds, 1),
        "n_total": int(len(df)), "n_train": int(len(train_df)), "n_test": int(len(test_df)),
        "accuracy": float(accuracy), "majority_baseline": majority,
        "class_names": names,
        "test_distribution": {CLASS_NAMES[c]: int((y_true == c).sum()) for c in labels},
        "report": report,
        "confusion_matrix": matrix.tolist(),
        "feature_importance": {n: float(v) for n, v in zip(FEATURE_NAMES, model.feature_importances_)},
        "global_shap": global_shap,
        "baseline": baseline,
    }
    with open(METRICS_FILE, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    size_mb = os.path.getsize(MODEL_FILE) / 1e6
    print(f"\n[SUCCESS] Saved threat_model.pkl ({size_mb:.0f} MB), model_metrics.json, demo_events.csv")
    print("You may now run 'python app.py' to launch the Security Operations Center dashboard.")


if __name__ == "__main__":
    main()