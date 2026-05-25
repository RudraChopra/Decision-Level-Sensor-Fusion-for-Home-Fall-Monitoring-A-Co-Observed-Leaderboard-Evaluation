#!/usr/bin/env python3
"""Subject-held-out external benchmark on UMAFall.

UMAFall is not co-observed with UR and does not include fall-onset frame labels,
so this script evaluates trial-level fall-vs-ADL recognition rather than alert
latency. Its purpose is external replication: does a lightweight inertial model
generalize across subjects on an independent public dataset?
"""

from __future__ import annotations

import argparse
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


URL = "https://ndownloader.figshare.com/files/43076140"


def ensure_data(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "UMAFall_Dataset_corrected_version.zip"
    extract_dir = data_dir / "extracted"
    if not zip_path.exists():
        urllib.request.urlretrieve(URL, zip_path)
    if not extract_dir.exists() or not list(extract_dir.glob("*.csv")):
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    return extract_dir


def parse_meta(path: Path) -> dict[str, str | int]:
    name = path.name
    subject_match = re.search(r"Subject_(\d+)", name)
    movement = "Fall" if "_Fall_" in name else "ADL"
    desc = name.split("_Fall_")[-1].split("_ADL_")[-1].split("_")[0]
    return {
        "file": path.name,
        "subject": int(subject_match.group(1)) if subject_match else -1,
        "label": 1 if movement == "Fall" else 0,
        "movement": movement,
        "description": desc,
    }


def summarize_signal(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {}
    x, y, z = values[:, 0], values[:, 1], values[:, 2]
    mag = np.sqrt(x * x + y * y + z * z)
    out = {}
    for arr_name, arr in [("x", x), ("y", y), ("z", z), ("mag", mag)]:
        out[f"{prefix}_{arr_name}_mean"] = float(np.mean(arr))
        out[f"{prefix}_{arr_name}_std"] = float(np.std(arr))
        out[f"{prefix}_{arr_name}_min"] = float(np.min(arr))
        out[f"{prefix}_{arr_name}_max"] = float(np.max(arr))
        out[f"{prefix}_{arr_name}_range"] = float(np.max(arr) - np.min(arr))
        out[f"{prefix}_{arr_name}_p05"] = float(np.percentile(arr, 5))
        out[f"{prefix}_{arr_name}_p95"] = float(np.percentile(arr, 95))
    if len(mag) > 2:
        jerk = np.diff(mag)
        out[f"{prefix}_mag_jerk_mean_abs"] = float(np.mean(np.abs(jerk)))
        out[f"{prefix}_mag_jerk_max_abs"] = float(np.max(np.abs(jerk)))
    else:
        out[f"{prefix}_mag_jerk_mean_abs"] = 0.0
        out[f"{prefix}_mag_jerk_max_abs"] = 0.0
    return out


def featurize_file(path: Path) -> dict[str, float | str | int]:
    meta = parse_meta(path)
    df = pd.read_csv(
        path,
        sep=";",
        comment="%",
        header=None,
        names=["timestamp", "sample", "x", "y", "z", "sensor_type", "sensor_id"],
        engine="python",
    )
    df = df.dropna()
    row: dict[str, float | str | int] = dict(meta)
    # Use all body positions, separated by sensor ID and modality. This avoids
    # making assumptions about a single wearable placement.
    for (sensor_type, sensor_id), part in df.groupby(["sensor_type", "sensor_id"]):
        prefix = f"s{int(sensor_id)}_t{int(sensor_type)}"
        row.update(summarize_signal(part[["x", "y", "z"]].to_numpy(dtype=float), prefix))
    return row


def load_features(extract_dir: Path, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    rows = []
    files = sorted(extract_dir.glob("*.csv"))
    for i, path in enumerate(files, start=1):
        if i % 100 == 0:
            print(f"featurized {i}/{len(files)}")
        rows.append(featurize_file(path))
    df = pd.DataFrame(rows).fillna(0.0)
    df.to_csv(cache_path, index=False)
    return df


def evaluate(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in {"file", "subject", "label", "movement", "description"}]
    X = df[feature_cols]
    y = df["label"].astype(int)
    groups = df["subject"].astype(int)
    models = {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=2026,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            random_state=2026,
        ),
    }
    splitter = GroupKFold(n_splits=5)
    pred_rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        print(f"fold {fold}")
        for name, model in models.items():
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            prob = model.predict_proba(X.iloc[test_idx])[:, 1]
            pred = (prob >= 0.5).astype(int)
            for idx, p, pr in zip(test_idx, prob, pred):
                pred_rows.append(
                    {
                        "fold": fold,
                        "model": name,
                        "file": df.iloc[idx]["file"],
                        "subject": int(df.iloc[idx]["subject"]),
                        "label": int(y.iloc[idx]),
                        "probability": float(p),
                        "prediction": int(pr),
                    }
                )
    pred_df = pd.DataFrame(pred_rows)
    rows = []
    for name, part in pred_df.groupby("model"):
        rows.append(
            {
                "model": name,
                "n_trials": len(part),
                "accuracy": accuracy_score(part["label"], part["prediction"]),
                "precision": precision_score(part["label"], part["prediction"], zero_division=0),
                "recall": recall_score(part["label"], part["prediction"], zero_division=0),
                "f1": f1_score(part["label"], part["prediction"], zero_division=0),
                "auroc": roc_auc_score(part["label"], part["probability"]),
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "umafall_predictions.csv", index=False)
    summary = pd.DataFrame(rows).sort_values("f1", ascending=False)
    summary.to_csv(out_dir / "umafall_summary.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/umafall"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/umafall"))
    args = parser.parse_args()
    extract_dir = ensure_data(args.data_dir)
    df = load_features(extract_dir, args.data_dir / "umafall_features.csv")
    print(f"Loaded {len(df)} UMAFall trials from {df.subject.nunique()} subjects")
    summary = evaluate(df, args.out_dir)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
