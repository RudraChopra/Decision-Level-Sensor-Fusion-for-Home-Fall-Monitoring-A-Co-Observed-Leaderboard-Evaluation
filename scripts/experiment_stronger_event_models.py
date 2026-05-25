#!/usr/bin/env python3
"""Exploratory stronger causal event models for UR Fall.

This is separate from the conservative manuscript benchmark while we search for
a genuinely stronger model. It still obeys grouped folds and train-only
threshold tuning.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ur_event_benchmark as b

warnings.simplefilter("ignore", PerformanceWarning)


BASE = b.FEATURE_COLS + b.ACC_COLS
METHODS = [
    "raw_temporal_logistic",
    "raw_temporal_extratrees",
    "raw_temporal_hgb",
    "raw_temporal_rf",
]


def add_raw_temporal_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.sort_values(["sequence", "time_s"]).copy()
    feature_cols = list(BASE)
    for col in BASE:
        g = out.groupby("sequence", sort=False)[col]
        for lag in [1, 2, 3, 5]:
            name = f"{col}_diff{lag}"
            out[name] = g.diff(lag).fillna(0.0)
            feature_cols.append(name)
        for window in [3, 5, 10, 20]:
            r = g.rolling(window, min_periods=1)
            for suffix, values in [
                ("mean", r.mean()),
                ("std", r.std().fillna(0.0)),
                ("min", r.min()),
                ("max", r.max()),
            ]:
                name = f"{col}_r{window}_{suffix}"
                out[name] = values.reset_index(level=0, drop=True).to_numpy()
                feature_cols.append(name)
    # Compact domain features that often matter for falls.
    out["sv_excess_1g"] = (out["sv_last"] - 1.0).abs()
    out["sv_peak_over_mean"] = out["sv_max"] - out["sv_mean"]
    out["camera_floor_proximity_change"] = out.groupby("sequence", sort=False)["P40"].diff().fillna(0.0)
    out["height_change"] = out.groupby("sequence", sort=False)["H"].diff().fillna(0.0)
    feature_cols += ["sv_excess_1g", "sv_peak_over_mean", "camera_floor_proximity_change", "height_change"]
    return out, feature_cols


def balanced_weights(y: pd.Series) -> np.ndarray:
    y = y.to_numpy(dtype=int)
    pos = max(1, int(y.sum()))
    neg = max(1, int((1 - y).sum()))
    n = len(y)
    return np.where(y == 1, n / (2 * pos), n / (2 * neg))


def fit_model(method: str, X: pd.DataFrame, y: pd.Series, seed: int):
    if method == "raw_temporal_logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs"),
        ).fit(X, y)
    if method == "raw_temporal_extratrees":
        return ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ).fit(X, y)
    if method == "raw_temporal_rf":
        return RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ).fit(X, y)
    if method == "raw_temporal_hgb":
        model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            random_state=seed,
        )
        return model.fit(X, y, sample_weight=balanced_weights(y))
    raise ValueError(method)


def score_model(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def inner_oof_scores(train: pd.DataFrame, features: list[str], method: str, seed: int) -> pd.DataFrame:
    y_group, groups = b.group_frame_labels(train)
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    parts = []
    for tr_idx, val_idx in splitter.split(train, y_group, groups):
        tr = train.iloc[tr_idx]
        val = train.iloc[val_idx].copy()
        mask = tr["trainable_frame"].to_numpy(dtype=bool)
        model = fit_model(method, tr.loc[mask, features], tr.loc[mask, "event_frame"], seed)
        val[method] = score_model(model, val[features])
        parts.append(val[["sequence", "time_s", "posture", "event_frame", "trainable_frame", "is_fall_sequence", method]])
    return pd.concat(parts, ignore_index=True)


def run(data_dir: Path, out_dir: Path):
    df, windows = b.build_dataset(data_dir)
    df, features = add_raw_temporal_features(df)
    y_group, groups = b.group_frame_labels(df)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2027)
    thresholds = np.linspace(0.02, 0.98, 49)
    ks = [1, 2, 3, 5, 8, 12]
    pred_parts = []
    tuning_rows = []
    for fold, (tr_idx, te_idx) in enumerate(splitter.split(df, y_group, groups), start=1):
        train = df.iloc[tr_idx].copy()
        test = df.iloc[te_idx].copy()
        print(f"fold {fold}")
        test_out = test[["sequence", "time_s", "posture", "event_frame", "trainable_frame", "is_fall_sequence"]].copy()
        test_out["fold"] = fold
        for method in METHODS:
            train_scores = inner_oof_scores(train, features, method, 2027 + fold)
            mask = train["trainable_frame"].to_numpy(dtype=bool)
            final_model = fit_model(method, train.loc[mask, features], train.loc[mask, "event_frame"], 3027 + fold)
            test_out[method] = score_model(final_model, test[features])
            threshold, k, tm = b.tune_alert_rule(train_scores, method, windows, thresholds, ks)
            test_out[f"{method}_alert"] = b.alerts_from_score(test_out, method, threshold, k)
            tuning_rows.append({"fold": fold, "method": method, "threshold": threshold, "k": k, **tm})
        pred_parts.append(test_out)
    pred = pd.concat(pred_parts, ignore_index=True)
    rows = []
    for method in METHODS:
        m = b.event_metrics(pred, f"{method}_alert", windows)
        m.update(b.frame_metrics(pred.rename(columns={f"{method}_alert": f"{method}_alert"}), method))
        m["method"] = method
        rows.append(m)
    summary = pd.DataFrame(rows).sort_values("event_f1", ascending=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(out_dir / "tuning.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary[["method", "tp", "fp_episodes", "fn", "event_f1", "event_precision", "event_recall", "false_alarms_per_hour", "median_latency_s", "frame_f1"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/urfall"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/stronger_event"))
    args = parser.parse_args()
    run(args.data_dir, args.out_dir)
