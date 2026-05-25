#!/usr/bin/env python3
"""YOLO pose video baseline for UR Fall.

This extracts pose keypoints from the UR frontal RGB videos with a pretrained
YOLO pose model, builds causal pose-motion features, and evaluates fall-event
alerts under the same grouped event protocol used by the sensor-fusion scripts.
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO

import ur_event_benchmark as b


BASE_URL = "https://fenix.ur.edu.pl/~mkepski/ds/data"


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        last_error = None
        for _ in range(5):
            try:
                urllib.request.urlretrieve(url, path)
                return
            except Exception as exc:  # pragma: no cover - network guard
                last_error = exc
                time.sleep(2)
        raise RuntimeError(f"Could not download {url}") from last_error


def ensure_videos(video_dir: Path) -> None:
    for i in range(1, 31):
        download(f"{BASE_URL}/fall-{i:02d}-cam0.mp4", video_dir / f"fall-{i:02d}-cam0.mp4")
    for i in range(1, 41):
        download(f"{BASE_URL}/adl-{i:02d}-cam0.mp4", video_dir / f"adl-{i:02d}-cam0.mp4")


def pose_row(result, width: int, height: int) -> dict[str, float]:
    row: dict[str, float] = {}
    if result.keypoints is None or len(result.keypoints) == 0:
        for i in range(17):
            row[f"kp{i}_x"] = 0.0
            row[f"kp{i}_y"] = 0.0
            row[f"kp{i}_c"] = 0.0
        row.update({"pose_conf": 0.0, "pose_w": 0.0, "pose_h": 0.0, "pose_aspect": 0.0})
        return row
    # Pick the person with the highest mean keypoint confidence.
    xy = result.keypoints.xy.cpu().numpy()
    conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else np.ones(xy.shape[:2])
    person = int(np.argmax(conf.mean(axis=1)))
    pts = xy[person]
    cf = conf[person]
    valid = cf > 0.2
    for i, (pt, c) in enumerate(zip(pts, cf)):
        row[f"kp{i}_x"] = float(pt[0] / max(width, 1))
        row[f"kp{i}_y"] = float(pt[1] / max(height, 1))
        row[f"kp{i}_c"] = float(c)
    if valid.any():
        xs = pts[valid, 0] / max(width, 1)
        ys = pts[valid, 1] / max(height, 1)
        pose_w = float(xs.max() - xs.min())
        pose_h = float(ys.max() - ys.min())
        row["pose_conf"] = float(cf[valid].mean())
        row["pose_w"] = pose_w
        row["pose_h"] = pose_h
        row["pose_aspect"] = float(pose_h / (pose_w + 1e-6))
    else:
        row.update({"pose_conf": 0.0, "pose_w": 0.0, "pose_h": 0.0, "pose_aspect": 0.0})
    # A few interpretable body-geometry cues.
    for name, ids in {
        "shoulder": [5, 6],
        "hip": [11, 12],
        "ankle": [15, 16],
    }.items():
        good = [i for i in ids if cf[i] > 0.2]
        if good:
            row[f"{name}_y"] = float(np.mean(pts[good, 1]) / max(height, 1))
        else:
            row[f"{name}_y"] = 0.0
    row["hip_minus_shoulder_y"] = row["hip_y"] - row["shoulder_y"]
    row["ankle_minus_hip_y"] = row["ankle_y"] - row["hip_y"]
    return row


def extract_sequence_pose(model: YOLO, video_path: Path, frames: list[int], imgsz: int) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    wanted = set(int(f) for f in frames)
    rows = []
    frame_idx = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in wanted:
            h, w = frame.shape[:2]
            result = model.predict(frame, imgsz=imgsz, verbose=False)[0]
            row = {"frame": frame_idx}
            row.update(pose_row(result, w, h))
            rows.append(row)
        frame_idx += 1
    cap.release()
    return pd.DataFrame(rows)


def build_pose_features(df: pd.DataFrame, video_dir: Path, cache_path: Path, model_name: str, imgsz: int) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_name)
    parts = []
    for seq, part in df.groupby("sequence", sort=True):
        video_path = video_dir / f"{seq}-cam0.mp4"
        print(f"pose {seq}")
        pose = extract_sequence_pose(model, video_path, sorted(part["frame"].astype(int).tolist()), imgsz)
        merged = part.merge(pose, on="frame", how="left").fillna(0.0)
        parts.append(merged)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(cache_path, index=False)
    return out


def add_pose_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.sort_values(["sequence", "time_s"]).copy()
    base = [c for c in out.columns if c.startswith("kp") or c in {
        "pose_conf", "pose_w", "pose_h", "pose_aspect",
        "shoulder_y", "hip_y", "ankle_y",
        "hip_minus_shoulder_y", "ankle_minus_hip_y",
    }]
    new_cols = {}
    for col in base:
        g = out.groupby("sequence", sort=False)[col]
        for lag in [1, 3, 5]:
            new_cols[f"{col}_d{lag}"] = g.diff(lag).fillna(0.0)
        for window in [3, 5, 10]:
            roll = g.rolling(window, min_periods=1)
            new_cols[f"{col}_r{window}_mean"] = roll.mean().reset_index(level=0, drop=True).to_numpy()
            new_cols[f"{col}_r{window}_std"] = roll.std().fillna(0.0).reset_index(level=0, drop=True).to_numpy()
    feat_df = pd.DataFrame(new_cols, index=out.index)
    out = pd.concat([out, feat_df], axis=1)
    return out, base + list(feat_df.columns)


def fit_model(name: str, X, y, seed: int):
    if name == "yolo_pose_logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(X, y)
    if name == "yolo_pose_extratrees":
        return ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ).fit(X, y)
    if name == "yolo_pose_hgb":
        return HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            random_state=seed,
        ).fit(X, y)
    raise ValueError(name)


def inner_oof_scores(train: pd.DataFrame, features: list[str], method: str, seed: int) -> pd.DataFrame:
    y_group, groups = b.group_frame_labels(train)
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    parts = []
    for inner_train_idx, inner_val_idx in splitter.split(train, y_group, groups):
        inner_train = train.iloc[inner_train_idx]
        inner_val = train.iloc[inner_val_idx]
        mask = inner_train["trainable_frame"].to_numpy(dtype=bool)
        model = fit_model(method, inner_train.loc[mask, features], inner_train.loc[mask, "event_frame"], seed)
        val_scores = inner_val[["sequence", "time_s", "posture", "event_frame", "trainable_frame", "is_fall_sequence"]].copy()
        val_scores[method] = model.predict_proba(inner_val[features])[:, 1]
        parts.append(val_scores)
    return pd.concat(parts, ignore_index=True)


def run(args):
    data_dir = args.data_dir
    if not args.skip_download:
        b.ensure_ur_files(data_dir)
        ensure_videos(args.video_dir)
    df, windows = b.build_dataset(data_dir)
    pose_df = build_pose_features(df, args.video_dir, args.pose_cache, args.model, args.imgsz)
    pose_df, features = add_pose_temporal(pose_df)
    y_group, groups = b.group_frame_labels(pose_df)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2028)
    thresholds = np.linspace(0.02, 0.98, 49)
    ks = [1, 2, 3, 5, 8, 12]
    methods = ["yolo_pose_logistic", "yolo_pose_extratrees", "yolo_pose_hgb"]
    pred_parts = []
    tuning = []
    for fold, (tr_idx, te_idx) in enumerate(splitter.split(pose_df, y_group, groups), start=1):
        print(f"fold {fold}")
        train = pose_df.iloc[tr_idx]
        test = pose_df.iloc[te_idx]
        test_out = test[["sequence", "time_s", "posture", "event_frame", "trainable_frame", "is_fall_sequence"]].copy()
        for method in methods:
            mask = train["trainable_frame"].to_numpy(dtype=bool)
            train_scores = inner_oof_scores(train, features, method, 4028 + fold)
            model = fit_model(method, train.loc[mask, features], train.loc[mask, "event_frame"], 2028 + fold)
            test_out[method] = model.predict_proba(test[features])[:, 1]
            th, k, tm = b.tune_alert_rule(train_scores, method, windows, thresholds, ks)
            test_out[f"{method}_alert"] = b.alerts_from_score(test_out, method, th, k)
            tuning.append({"fold": fold, "method": method, "threshold": th, "k": k, **tm})
        pred_parts.append(test_out)
    pred = pd.concat(pred_parts, ignore_index=True)
    rows = []
    for method in methods:
        m = b.event_metrics(pred, f"{method}_alert", windows)
        m.update(b.frame_metrics(pred, method))
        m["method"] = method
        rows.append(m)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_dir / "yolo_pose_predictions.csv", index=False)
    pd.DataFrame(tuning).to_csv(out_dir / "yolo_pose_tuning.csv", index=False)
    summary = pd.DataFrame(rows).sort_values("event_f1", ascending=False)
    summary.to_csv(out_dir / "yolo_pose_summary.csv", index=False)
    print(summary[["method", "tp", "fp_episodes", "fn", "event_f1", "event_precision", "event_recall", "false_alarms_per_hour", "median_latency_s", "frame_f1"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/urfall"))
    parser.add_argument("--video-dir", type=Path, default=Path("data/urfall_videos"))
    parser.add_argument("--pose-cache", type=Path, default=Path("results/yolo_pose/pose_features.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/yolo_pose"))
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
