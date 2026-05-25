#!/usr/bin/env python3
"""Event-level UR Fall decision-fusion benchmark.

This script fixes the central evaluation problem in the earlier manuscript:
fall monitoring is evaluated as an alert event problem, not as a frame
classification problem.  It uses the UR Fall Detection Dataset's extracted
depth features and synchronized accelerometer streams, trains only on
sequence-held-out data, treats ADL lying/transition frames as hard negatives,
and reports event recall, false alert episodes per hour, latency, and
bootstrap confidence intervals over sequences.

The code is intentionally notebook-friendly.  It runs as a script locally or
in Google Colab.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_URL = "https://fenix.ur.edu.pl/~mkepski/ds/data"
FEATURE_COLS = [
    "HeightWidthRatio",
    "MajorMinorRatio",
    "BoundingBoxOccupancy",
    "MaxStdXZ",
    "HHmaxRatio",
    "H",
    "D",
    "P40",
]
ACC_COLS = [
    "sv_last",
    "sv_mean",
    "sv_std",
    "sv_min",
    "sv_max",
    "sv_range",
    "sv_slope",
    "ax_mean",
    "ax_std",
    "ax_range",
    "ay_mean",
    "ay_std",
    "ay_range",
    "az_mean",
    "az_std",
    "az_range",
]
METHODS = [
    "camera",
    "wearable",
    "mean_fusion",
    "max_or",
    "product_agreement",
    "calibrated_logistic",
    "temporal_logistic",
    "temporal_rf",
    "temporal_hgb",
    "and_persist",
    "or_persist",
]


@dataclass(frozen=True)
class EventWindow:
    sequence: str
    is_fall: bool
    start_s: float | None
    deadline_s: float | None
    negative_start_s: float
    negative_end_s: float


def download(url: str, path: Path, retries: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            urllib.request.urlretrieve(url, path)
            return
        except Exception as exc:  # pragma: no cover - network guard
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"Could not download {url}") from last_error


def ensure_ur_files(data_dir: Path) -> None:
    download(f"{BASE_URL}/urfall-cam0-falls.csv", data_dir / "urfall-cam0-falls.csv")
    download(f"{BASE_URL}/urfall-cam0-adls.csv", data_dir / "urfall-cam0-adls.csv")
    for i in range(1, 31):
        download(f"{BASE_URL}/fall-{i:02d}-data.csv", data_dir / f"fall-{i:02d}-data.csv")
        download(f"{BASE_URL}/fall-{i:02d}-acc.csv", data_dir / f"fall-{i:02d}-acc.csv")
    for i in range(1, 41):
        download(f"{BASE_URL}/adl-{i:02d}-data.csv", data_dir / f"adl-{i:02d}-data.csv")
        download(f"{BASE_URL}/adl-{i:02d}-acc.csv", data_dir / f"adl-{i:02d}-acc.csv")


def read_features(data_dir: Path) -> pd.DataFrame:
    cols = ["sequence", "frame", "posture"] + FEATURE_COLS
    falls = pd.read_csv(data_dir / "urfall-cam0-falls.csv", header=None, names=cols)
    adls = pd.read_csv(data_dir / "urfall-cam0-adls.csv", header=None, names=cols)
    falls["sequence_type"] = "fall"
    adls["sequence_type"] = "adl"
    df = pd.concat([falls, adls], ignore_index=True)
    df["sequence"] = df["sequence"].astype(str)
    df["frame"] = df["frame"].astype(int)
    df["posture"] = df["posture"].astype(int)
    return df


def read_sync(data_dir: Path, sequence: str) -> pd.DataFrame:
    sync = pd.read_csv(
        data_dir / f"{sequence}-data.csv",
        header=None,
        names=["frame", "time_ms", "sv_interp"],
    )
    sync["frame"] = sync["frame"].astype(int)
    sync["time_s"] = sync["time_ms"].astype(float) / 1000.0
    return sync[["frame", "time_s", "sv_interp"]]


def read_acc(data_dir: Path, sequence: str) -> pd.DataFrame:
    acc = pd.read_csv(
        data_dir / f"{sequence}-acc.csv",
        header=None,
        names=["time_ms", "sv", "ax", "az", "ay"],
    )
    acc["time_s"] = acc["time_ms"].astype(float) / 1000.0
    return acc[["time_s", "sv", "ax", "ay", "az"]].sort_values("time_s").reset_index(drop=True)


def causal_acc_features(frame_times: np.ndarray, acc: pd.DataFrame, window_s: float = 0.5) -> pd.DataFrame:
    acc_times = acc["time_s"].to_numpy()
    values = acc[["sv", "ax", "ay", "az"]].to_numpy(dtype=float)
    rows: list[list[float]] = []
    for t in frame_times:
        lo = np.searchsorted(acc_times, t - window_s, side="left")
        hi = np.searchsorted(acc_times, t, side="right")
        if hi <= lo:
            nearest = int(np.clip(np.searchsorted(acc_times, t), 0, len(acc_times) - 1))
            window = values[nearest : nearest + 1]
            wt = acc_times[nearest : nearest + 1]
        else:
            window = values[lo:hi]
            wt = acc_times[lo:hi]

        sv = window[:, 0]
        axes = window[:, 1:4]
        duration = max(float(wt[-1] - wt[0]), 1e-6)
        sv_slope = float((sv[-1] - sv[0]) / duration) if len(sv) > 1 else 0.0
        row = [
            float(sv[-1]),
            float(np.mean(sv)),
            float(np.std(sv)),
            float(np.min(sv)),
            float(np.max(sv)),
            float(np.max(sv) - np.min(sv)),
            sv_slope,
        ]
        for j in range(3):
            a = axes[:, j]
            row.extend([float(np.mean(a)), float(np.std(a)), float(np.max(a) - np.min(a))])
        rows.append(row)
    return pd.DataFrame(rows, columns=ACC_COLS)


def build_dataset(data_dir: Path, grace_s: float = 0.5) -> tuple[pd.DataFrame, dict[str, EventWindow]]:
    df = read_features(data_dir)
    joined_parts = []
    windows: dict[str, EventWindow] = {}

    for seq, seq_df in df.groupby("sequence", sort=True):
        sync = read_sync(data_dir, seq)
        acc = read_acc(data_dir, seq)
        merged = seq_df.merge(sync, on="frame", how="inner").sort_values("time_s").reset_index(drop=True)
        acc_feats = causal_acc_features(merged["time_s"].to_numpy(dtype=float), acc)
        merged = pd.concat([merged, acc_feats], axis=1)
        merged["row_id"] = np.arange(len(merged))
        joined_parts.append(merged)

        is_fall = seq.startswith("fall")
        seq_start = float(merged["time_s"].min())
        seq_end = float(merged["time_s"].max())
        if is_fall:
            transition = merged[merged["posture"] == 0]
            if transition.empty:
                transition = merged[merged["posture"] == 1]
            start_s = float(transition["time_s"].min())
            last_transition_s = float(transition["time_s"].max())
            first_ground = merged[(merged["posture"] == 1) & (merged["time_s"] >= start_s)]
            if not first_ground.empty:
                deadline_s = min(seq_end, float(first_ground["time_s"].min()) + grace_s)
            else:
                deadline_s = min(seq_end, last_transition_s + grace_s)
            windows[seq] = EventWindow(seq, True, start_s, deadline_s, seq_start, seq_end)
        else:
            windows[seq] = EventWindow(seq, False, None, None, seq_start, seq_end)

    out = pd.concat(joined_parts, ignore_index=True)
    out["is_fall_sequence"] = out["sequence"].str.startswith("fall").astype(int)
    # Primary event training target: only transition frames in actual fall sequences.
    out["event_frame"] = ((out["is_fall_sequence"] == 1) & (out["posture"] == 0)).astype(int)
    # Late post-fall lying frames are censored for frame-level training, so the
    # camera cannot win just by learning "person is lying on the floor."
    out["trainable_frame"] = ~((out["is_fall_sequence"] == 1) & (out["posture"] == 1))
    return out.reset_index(drop=True), windows


def group_frame_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    groups = df["sequence"].to_numpy()
    seq_type = df["is_fall_sequence"].to_numpy()
    return seq_type, groups


def fit_logistic(X: pd.DataFrame, y: pd.Series):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
    ).fit(X, y)


def predict_positive(model, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    return proba[:, list(model.classes_).index(1)] if hasattr(model, "classes_") else proba[:, 1]


def add_temporal_features(scored: pd.DataFrame, window_frames: int = 5) -> pd.DataFrame:
    out = scored.sort_values(["sequence", "time_s"]).copy()
    for col in ["camera_score", "wearable_score"]:
        roll = (
            out.groupby("sequence", sort=False)[col]
            .rolling(window_frames, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        out[f"{col}_mean"] = roll.to_numpy()
    out["score_product"] = out["camera_score"] * out["wearable_score"]
    out["score_absdiff"] = (out["camera_score"] - out["wearable_score"]).abs()
    out["mean_min"] = np.minimum(out["camera_score_mean"], out["wearable_score_mean"])
    return out


def contiguous_episodes(times: np.ndarray, active: np.ndarray, merge_gap_s: float = 1.0) -> list[tuple[float, float]]:
    episodes: list[tuple[float, float]] = []
    start: float | None = None
    last: float | None = None
    for t, a in zip(times, active):
        if a:
            if start is None:
                start = float(t)
            last = float(t)
        elif start is not None and last is not None:
            episodes.append((start, last))
            start = None
            last = None
    if start is not None and last is not None:
        episodes.append((start, last))

    if not episodes:
        return episodes
    merged = [episodes[0]]
    for start, end in episodes[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= merge_gap_s:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def alerts_from_score(seq_df: pd.DataFrame, score_col: str, threshold: float, persistence_k: int) -> pd.Series:
    alerts = pd.Series(False, index=seq_df.index)
    for _, part in seq_df.sort_values(["sequence", "time_s"]).groupby("sequence", sort=False):
        raw = (part[score_col].to_numpy() >= threshold).astype(int)
        if persistence_k <= 1:
            active = raw.astype(bool)
        else:
            run = np.zeros_like(raw)
            current = 0
            for i, value in enumerate(raw):
                current = current + 1 if value else 0
                run[i] = current
            active = run >= persistence_k
        alerts.loc[part.index] = active
    return alerts


def event_metrics(
    df: pd.DataFrame,
    alert_col: str,
    windows: dict[str, EventWindow],
    sequence_subset: set[str] | None = None,
) -> dict[str, float]:
    tp = fp = fn = 0
    latencies = []
    negative_seconds = 0.0
    detected_sequences: set[str] = set()

    for seq, part in df.groupby("sequence", sort=True):
        if sequence_subset is not None and seq not in sequence_subset:
            continue
        w = windows[seq]
        part = part.sort_values("time_s")
        times = part["time_s"].to_numpy(dtype=float)
        active = part[alert_col].to_numpy(dtype=bool)
        episodes = contiguous_episodes(times, active)

        if w.is_fall and w.start_s is not None and w.deadline_s is not None:
            negative_seconds += max(0.0, w.start_s - w.negative_start_s)
            detection_candidates = [
                start
                for start, end in episodes
                if end >= w.start_s and start <= w.deadline_s
            ]
            if detection_candidates:
                tp += 1
                detected_sequences.add(seq)
                latencies.append(max(0.0, min(detection_candidates) - w.start_s))
                used = False
                for start, end in episodes:
                    overlaps = end >= w.start_s and start <= w.deadline_s
                    if overlaps and not used:
                        used = True
                    elif start < w.start_s or start > w.deadline_s:
                        fp += 1
            else:
                fn += 1
                for start, _ in episodes:
                    if start < w.start_s or start > w.deadline_s:
                        fp += 1
        else:
            negative_seconds += max(0.0, w.negative_end_s - w.negative_start_s)
            fp += len(episodes)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    false_per_hour = fp / (negative_seconds / 3600.0) if negative_seconds > 0 else math.nan
    latency_median = float(np.median(latencies)) if latencies else math.nan
    latency_p90 = float(np.percentile(latencies, 90)) if latencies else math.nan
    return {
        "tp": float(tp),
        "fp_episodes": float(fp),
        "fn": float(fn),
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "false_alarms_per_hour": false_per_hour,
        "median_latency_s": latency_median,
        "p90_latency_s": latency_p90,
        "detected_falls": float(len(detected_sequences)),
    }


def sequence_components(
    df: pd.DataFrame,
    alert_col: str,
    windows: dict[str, EventWindow],
) -> pd.DataFrame:
    """Return additive event-evaluation components per sequence."""
    rows = []
    for seq, part in df.groupby("sequence", sort=True):
        w = windows[seq]
        part = part.sort_values("time_s")
        times = part["time_s"].to_numpy(dtype=float)
        active = part[alert_col].to_numpy(dtype=bool)
        episodes = contiguous_episodes(times, active)

        tp = fp = fn = 0
        latency = math.nan
        if w.is_fall and w.start_s is not None and w.deadline_s is not None:
            negative_seconds = max(0.0, w.start_s - w.negative_start_s)
            detection_candidates = [
                start
                for start, end in episodes
                if end >= w.start_s and start <= w.deadline_s
            ]
            if detection_candidates:
                tp = 1
                latency = max(0.0, min(detection_candidates) - w.start_s)
                used = False
                for start, end in episodes:
                    overlaps = end >= w.start_s and start <= w.deadline_s
                    if overlaps and not used:
                        used = True
                    elif start < w.start_s or start > w.deadline_s:
                        fp += 1
            else:
                fn = 1
                for start, _ in episodes:
                    if start < w.start_s or start > w.deadline_s:
                        fp += 1
        else:
            negative_seconds = max(0.0, w.negative_end_s - w.negative_start_s)
            fp = len(episodes)
        rows.append(
            {
                "sequence": seq,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "negative_seconds": negative_seconds,
                "latency": latency,
            }
        )
    return pd.DataFrame(rows)


def metrics_from_components(comp: pd.DataFrame) -> dict[str, float]:
    tp = float(comp["tp"].sum())
    fp = float(comp["fp"].sum())
    fn = float(comp["fn"].sum())
    neg_seconds = float(comp["negative_seconds"].sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    latency_values = comp["latency"].dropna().to_numpy(dtype=float)
    return {
        "event_f1": f1,
        "event_recall": recall,
        "false_alarms_per_hour": fp / (neg_seconds / 3600.0) if neg_seconds else math.nan,
        "median_latency_s": float(np.median(latency_values)) if len(latency_values) else math.nan,
    }


def tune_alert_rule(
    train_scores: pd.DataFrame,
    score_col: str,
    windows: dict[str, EventWindow],
    thresholds: np.ndarray,
    ks: list[int],
) -> tuple[float, int, dict[str, float]]:
    best: tuple[float, float, float, int, dict[str, float]] | None = None
    sequences = set(train_scores["sequence"].unique())
    work = train_scores.copy()
    for k in ks:
        for threshold in thresholds:
            work["_alert"] = alerts_from_score(work, score_col, float(threshold), k)
            m = event_metrics(work, "_alert", windows, sequences)
            # Prefer event F1, then lower false alarm rate, then lower latency.
            latency = m["median_latency_s"]
            latency_key = latency if not math.isnan(latency) else 1e9
            key = (m["event_f1"], -m["false_alarms_per_hour"], -latency_key)
            if best is None or key > best[:3]:
                best = (key[0], key[1], key[2], k, {"threshold": float(threshold), **m})
    assert best is not None
    return float(best[4]["threshold"]), int(best[3]), best[4]


def train_base_scores(train_df: pd.DataFrame, test_df: pd.DataFrame, random_state: int):
    mask = train_df["trainable_frame"].to_numpy(dtype=bool)
    y = train_df.loc[mask, "event_frame"]
    cam = fit_logistic(train_df.loc[mask, FEATURE_COLS], y)
    wear = fit_logistic(train_df.loc[mask, ACC_COLS], y)
    out = test_df[["sequence", "time_s", "posture", "event_frame", "trainable_frame", "is_fall_sequence"]].copy()
    out["camera_score"] = predict_positive(cam, test_df[FEATURE_COLS])
    out["wearable_score"] = predict_positive(wear, test_df[ACC_COLS])
    return out


def inner_oof_base_scores(train_df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    y_group, groups = group_frame_labels(train_df)
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=random_state)
    parts = []
    for inner_train_idx, inner_val_idx in splitter.split(train_df, y_group, groups):
        inner_train = train_df.iloc[inner_train_idx]
        inner_val = train_df.iloc[inner_val_idx]
        parts.append(train_base_scores(inner_train, inner_val, random_state))
    return pd.concat(parts, ignore_index=True).sort_values(["sequence", "time_s"]).reset_index(drop=True)


def fit_fusion_models(train_scored: pd.DataFrame, random_state: int):
    train_scored = add_temporal_features(train_scored)
    mask = train_scored["trainable_frame"].to_numpy(dtype=bool)
    y = train_scored.loc[mask, "event_frame"]
    current_cols = ["camera_score", "wearable_score", "score_product", "score_absdiff"]
    temporal_cols = current_cols + ["camera_score_mean", "wearable_score_mean", "mean_min"]

    calibrated = fit_logistic(train_scored.loc[mask, current_cols], y)
    temporal = fit_logistic(train_scored.loc[mask, temporal_cols], y)
    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    ).fit(train_scored.loc[mask, temporal_cols], y)
    hgb = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=random_state,
    ).fit(train_scored.loc[mask, temporal_cols], y)
    return calibrated, temporal, rf, hgb, current_cols, temporal_cols


def score_all_methods(scored: pd.DataFrame, fusion_models=None) -> pd.DataFrame:
    out = add_temporal_features(scored)
    out["camera"] = out["camera_score"]
    out["wearable"] = out["wearable_score"]
    out["mean_fusion"] = 0.5 * (out["camera_score"] + out["wearable_score"])
    out["max_or"] = np.maximum(out["camera_score"], out["wearable_score"])
    out["product_agreement"] = out["camera_score"] * out["wearable_score"]
    out["and_persist"] = np.minimum(out["camera_score"], out["wearable_score"])
    out["or_persist"] = np.maximum(out["camera_score"], out["wearable_score"])
    if fusion_models is not None:
        calibrated, temporal, rf, hgb, current_cols, temporal_cols = fusion_models
        out["calibrated_logistic"] = predict_positive(calibrated, out[current_cols])
        out["temporal_logistic"] = predict_positive(temporal, out[temporal_cols])
        out["temporal_rf"] = rf.predict_proba(out[temporal_cols])[:, 1]
        out["temporal_hgb"] = hgb.predict_proba(out[temporal_cols])[:, 1]
    return out


def run_outer_cv(df: pd.DataFrame, windows: dict[str, EventWindow], random_state: int = 2026):
    y_group, groups = group_frame_labels(df)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    pred_parts = []
    tuning_rows = []
    thresholds = np.linspace(0.05, 0.95, 37)
    ks = [1, 2, 3, 5, 8]

    for fold, (train_idx, test_idx) in enumerate(splitter.split(df, y_group, groups), start=1):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        print(f"Fold {fold}: train sequences={train_df.sequence.nunique()}, test sequences={test_df.sequence.nunique()}")

        train_oof = inner_oof_base_scores(train_df, random_state + fold)
        fusion_models = fit_fusion_models(train_oof, random_state + fold)
        train_scored = score_all_methods(train_oof, fusion_models)

        test_base = train_base_scores(train_df, test_df, random_state + fold)
        test_scored = score_all_methods(test_base, fusion_models)
        test_scored["fold"] = fold

        for method in METHODS:
            threshold, k, train_metric = tune_alert_rule(train_scored, method, windows, thresholds, ks)
            test_scored[f"{method}_alert"] = alerts_from_score(test_scored, method, threshold, k)
            tuning_rows.append(
                {
                    "fold": fold,
                    "method": method,
                    "threshold": threshold,
                    "persistence_k": k,
                    "train_event_f1": train_metric["event_f1"],
                    "train_false_alarms_per_hour": train_metric["false_alarms_per_hour"],
                }
            )
        pred_parts.append(test_scored)

    return pd.concat(pred_parts, ignore_index=True), pd.DataFrame(tuning_rows)


def frame_metrics(pred: pd.DataFrame, method: str) -> dict[str, float]:
    # Secondary diagnostic only: fall transition frames are positive; ADL frames
    # and pre-fall upright frames are negative; late post-fall lying is ignored.
    mask = pred["trainable_frame"].to_numpy(dtype=bool)
    y_true = pred.loc[mask, "event_frame"].to_numpy(dtype=int)
    y_pred = pred.loc[mask, f"{method}_alert"].to_numpy(dtype=int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"frame_precision": precision, "frame_recall": recall, "frame_f1": f1}


def summarize(pred: pd.DataFrame, windows: dict[str, EventWindow]) -> pd.DataFrame:
    rows = []
    sequences = set(pred["sequence"].unique())
    for method in METHODS:
        metrics = event_metrics(pred, f"{method}_alert", windows, sequences)
        metrics.update(frame_metrics(pred, method))
        metrics["method"] = method
        rows.append(metrics)
    out = pd.DataFrame(rows)
    return out.sort_values(["event_f1", "false_alarms_per_hour"], ascending=[False, True])


def bootstrap_ci(
    pred: pd.DataFrame,
    windows: dict[str, EventWindow],
    method: str,
    n_boot: int = 2000,
    seed: int = 2026,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    comp = sequence_components(pred, f"{method}_alert", windows)
    seqs = np.arange(len(comp))
    values: dict[str, list[float]] = {
        "event_f1": [],
        "event_recall": [],
        "false_alarms_per_hour": [],
        "median_latency_s": [],
    }
    for _ in range(n_boot):
        sample_idx = rng.choice(seqs, size=len(seqs), replace=True)
        m = metrics_from_components(comp.iloc[sample_idx])
        for key in values:
            if not math.isnan(m[key]):
                values[key].append(float(m[key]))
    return {
        key: (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
        for key, vals in values.items()
        if vals
    }


def write_outputs(
    pred: pd.DataFrame,
    tuning: pd.DataFrame,
    summary: pd.DataFrame,
    windows: dict[str, EventWindow],
    out_dir: Path,
    n_boot: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_dir / "ur_event_predictions.csv", index=False)
    tuning.to_csv(out_dir / "ur_event_tuning.csv", index=False)
    summary.to_csv(out_dir / "ur_event_summary.csv", index=False)
    ci_rows = []
    for method in METHODS:
        ci = bootstrap_ci(pred, windows, method, n_boot=n_boot)
        row = {"method": method}
        for metric, (lo, hi) in ci.items():
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        ci_rows.append(row)
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(out_dir / "ur_event_bootstrap_ci.csv", index=False)
    with open(out_dir / "ur_event_windows.json", "w", encoding="utf-8") as f:
        json.dump({k: w.__dict__ for k, w in windows.items()}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/urfall", type=Path)
    parser.add_argument("--out-dir", default="results/ur_event", type=Path)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    if not args.skip_download:
        ensure_ur_files(args.data_dir)
    df, windows = build_dataset(args.data_dir)
    print(
        f"Loaded {len(df):,} synchronized frames from {df.sequence.nunique()} sequences "
        f"({df.is_fall_sequence.sum()} fall-sequence frames)."
    )
    pred, tuning = run_outer_cv(df, windows)
    summary = summarize(pred, windows)
    write_outputs(pred, tuning, summary, windows, args.out_dir, args.bootstrap)
    print("\nEvent-level leaderboard")
    cols = [
        "method",
        "event_f1",
        "event_precision",
        "event_recall",
        "false_alarms_per_hour",
        "median_latency_s",
        "frame_f1",
    ]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote outputs to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
