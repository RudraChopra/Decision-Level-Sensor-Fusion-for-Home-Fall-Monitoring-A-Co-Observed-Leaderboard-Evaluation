#!/usr/bin/env python3
"""Full UR event model: depth features + wearable dynamics + YOLO pose."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ur_event_benchmark as b
from experiment_stronger_event_models import add_raw_temporal_features
from ur_yolo_pose_baseline import add_pose_temporal

warnings.simplefilter("ignore", PerformanceWarning)


METHODS = ["full_event_extratrees", "full_event_hgb", "full_event_logistic"]


def fit_model(method: str, X, y, seed: int):
    if method == "full_event_extratrees":
        return ExtraTreesClassifier(
            n_estimators=700,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ).fit(X, y)
    if method == "full_event_hgb":
        return HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=0.15,
            random_state=seed,
        ).fit(X, y)
    if method == "full_event_logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs"),
        ).fit(X, y)
    raise ValueError(method)


def inner_oof(train: pd.DataFrame, features: list[str], method: str, seed: int) -> pd.DataFrame:
    y_group, groups = b.group_frame_labels(train)
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    parts = []
    for tr_idx, val_idx in splitter.split(train, y_group, groups):
        tr = train.iloc[tr_idx]
        val = train.iloc[val_idx]
        mask = tr["trainable_frame"].to_numpy(dtype=bool)
        model = fit_model(method, tr.loc[mask, features], tr.loc[mask, "event_frame"], seed)
        out = val[["sequence", "time_s", "posture", "event_frame", "trainable_frame", "is_fall_sequence"]].copy()
        out[method] = model.predict_proba(val[features])[:, 1]
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def run(data_dir: Path, pose_cache: Path, out_dir: Path):
    df, windows = b.build_dataset(data_dir)
    pose = pd.read_csv(pose_cache)
    pose_cols = [
        c for c in pose.columns
        if c.startswith("kp") or c in {
            "pose_conf", "pose_w", "pose_h", "pose_aspect",
            "shoulder_y", "hip_y", "ankle_y",
            "hip_minus_shoulder_y", "ankle_minus_hip_y",
        }
    ]
    df = df.merge(pose[["sequence", "frame"] + pose_cols], on=["sequence", "frame"], how="left").fillna(0.0)
    df, raw_features = add_raw_temporal_features(df)
    df, pose_features = add_pose_temporal(df)
    features = sorted(set(raw_features + pose_features))
    y_group, groups = b.group_frame_labels(df)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2030)
    thresholds = np.linspace(0.02, 0.98, 49)
    ks = [1, 2, 3, 5, 8, 12]
    pred_parts = []
    tuning = []
    for fold, (tr_idx, te_idx) in enumerate(splitter.split(df, y_group, groups), start=1):
        print(f"fold {fold}")
        train = df.iloc[tr_idx]
        test = df.iloc[te_idx]
        test_out = test[["sequence", "time_s", "posture", "event_frame", "trainable_frame", "is_fall_sequence"]].copy()
        for method in METHODS:
            train_scores = inner_oof(train, features, method, 2030 + fold)
            mask = train["trainable_frame"].to_numpy(dtype=bool)
            model = fit_model(method, train.loc[mask, features], train.loc[mask, "event_frame"], 3030 + fold)
            test_out[method] = model.predict_proba(test[features])[:, 1]
            th, k, tm = b.tune_alert_rule(train_scores, method, windows, thresholds, ks)
            test_out[f"{method}_alert"] = b.alerts_from_score(test_out, method, th, k)
            tuning.append({"fold": fold, "method": method, "threshold": th, "k": k, **tm})
        pred_parts.append(test_out)
    pred = pd.concat(pred_parts, ignore_index=True)
    rows = []
    for method in METHODS:
        m = b.event_metrics(pred, f"{method}_alert", windows)
        m.update(b.frame_metrics(pred, method))
        m["method"] = method
        rows.append(m)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(tuning).to_csv(out_dir / "tuning.csv", index=False)
    summary = pd.DataFrame(rows).sort_values("event_f1", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary[["method", "tp", "fp_episodes", "fn", "event_f1", "event_precision", "event_recall", "false_alarms_per_hour", "median_latency_s", "frame_f1"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/urfall"))
    parser.add_argument("--pose-cache", type=Path, default=Path("results/yolo_pose/pose_features.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/full_multimodal"))
    args = parser.parse_args()
    run(args.data_dir, args.pose_cache, args.out_dir)
