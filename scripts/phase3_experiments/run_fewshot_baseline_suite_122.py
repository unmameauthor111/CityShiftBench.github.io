#!/usr/bin/env python3
"""Run unified few-shot baseline suites on the active `_122` structural slices."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TARGETS = ["road_segments", "intersection_nodes"]
FEATURE_CANDIDATES = [
    "highway_ways",
    "major_road_ways",
    "road_segments",
    "intersection_nodes",
    "building_ways",
    "poi_objects",
    "park_objects",
    "water_objects",
    "landuse_residential",
    "landuse_commercial",
    "landuse_industrial",
    "landuse_retail",
]
TARGET_SAFE_EXCLUSIONS = {
    "road_segments": {"road_segments", "intersection_nodes", "target_road_segments", "target_intersection_nodes"},
    "intersection_nodes": {"road_segments", "intersection_nodes", "target_road_segments", "target_intersection_nodes"},
}
METHOD_LABELS = {
    "source_only_ridge": "Source-only ridge",
    "target_only_ridge": "Target-only ridge",
    "pooled_ridge": "Pooled ridge",
    "tabular_knn": "Tabular kNN",
    "coral_ridge": "CORAL ridge",
    "iw_ridge": "Importance-weighted ridge",
}


def _compact(text: str) -> str:
    return str(text).strip().replace(" ", "")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def transform(value: float) -> float:
    return float(math.log1p(max(value, 0.0)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def feature_names_for_target(df: pd.DataFrame, target: str) -> list[str]:
    exclusions = TARGET_SAFE_EXCLUSIONS[target]
    names: list[str] = []
    for name in FEATURE_CANDIDATES:
        if name in exclusions or name not in df.columns:
            continue
        values = df[name].map(_safe_float).to_numpy(dtype=float)
        if float(np.std(values)) > 1e-9:
            names.append(name)
    return names


def matrix(df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    return np.asarray([[transform(_safe_float(row[name])) for name in feature_names] for _, row in df.iterrows()], dtype=float)


def target_array(df: pd.DataFrame, target: str) -> np.ndarray:
    return df[f"target_{target}"].map(_safe_float).to_numpy(dtype=float)


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train_x, axis=0, keepdims=True)
    std = np.std(train_x, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std


def predict_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float = 1.0,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    train_z, test_z = standardize(train_x, test_x)
    train_aug = np.concatenate([np.ones((train_z.shape[0], 1)), train_z], axis=1)
    test_aug = np.concatenate([np.ones((test_z.shape[0], 1)), test_z], axis=1)
    ident = np.eye(train_aug.shape[1])
    ident[0, 0] = 0.0
    if sample_weight is None:
        lhs = train_aug.T @ train_aug + alpha * ident
        rhs = train_aug.T @ train_y
    else:
        weights = np.asarray(sample_weight, dtype=float)
        weights = np.clip(weights, 1e-6, 1e6)
        weighted_x = train_aug * np.sqrt(weights)[:, None]
        weighted_y = train_y * np.sqrt(weights)
        lhs = weighted_x.T @ weighted_x + alpha * ident
        rhs = weighted_x.T @ weighted_y
    coef = np.linalg.solve(lhs, rhs)
    return np.maximum(test_aug @ coef, 0.0)


def predict_knn(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, k: int = 5) -> np.ndarray:
    train_z, test_z = standardize(train_x, test_x)
    preds = []
    for row in test_z:
        dist = np.sqrt(np.sum((train_z - row[None, :]) ** 2, axis=1))
        idx = np.argsort(dist)[: min(k, len(train_y))]
        preds.append(float(np.mean(train_y[idx])))
    return np.asarray(preds, dtype=float)


def _cov_sqrt(mat: np.ndarray, inverse: bool = False, eps: float = 1e-5) -> np.ndarray:
    values, vectors = np.linalg.eigh(mat)
    values = np.clip(values, eps, None)
    power = -0.5 if inverse else 0.5
    return vectors @ np.diag(values**power) @ vectors.T


def coral_align_source_to_target(source_x: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    if len(source_x) < 2 or len(target_x) < 2:
        return source_x.copy()
    source_mean = np.mean(source_x, axis=0, keepdims=True)
    target_mean = np.mean(target_x, axis=0, keepdims=True)
    source_centered = source_x - source_mean
    target_centered = target_x - target_mean
    source_cov = np.cov(source_centered, rowvar=False) + np.eye(source_x.shape[1]) * 1e-5
    target_cov = np.cov(target_centered, rowvar=False) + np.eye(target_x.shape[1]) * 1e-5
    return source_centered @ _cov_sqrt(source_cov, inverse=True) @ _cov_sqrt(target_cov, inverse=False) + target_mean


def source_importance_weights(source_x: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    if len(source_x) == 0 or len(target_x) == 0:
        return np.ones(len(source_x), dtype=float)
    combined = np.vstack([source_x, target_x])
    mean = np.mean(combined, axis=0, keepdims=True)
    std = np.std(combined, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    source_z = (source_x - mean) / std
    target_z = (target_x - mean) / std
    centroid = np.mean(target_z, axis=0, keepdims=True)
    dist2 = np.sum((source_z - centroid) ** 2, axis=1)
    scale = float(np.median(dist2[dist2 > 1e-12])) if np.any(dist2 > 1e-12) else 1.0
    weights = np.exp(-dist2 / (2.0 * max(scale, 1e-6)))
    weights = weights / max(float(np.mean(weights)), 1e-8)
    return np.clip(weights, 0.05, 20.0)


def make_city_split(city_df: pd.DataFrame, shot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = city_df.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    if shot <= 0:
        return ordered.iloc[0:0].copy(), ordered.copy()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ordered))
    shot_idx = order[: min(shot, len(ordered))]
    eval_idx = order[min(shot, len(ordered)) :]
    return ordered.iloc[shot_idx].copy(), ordered.iloc[eval_idx].copy()


def predict_all_methods(
    source_train: pd.DataFrame,
    target_shot: pd.DataFrame,
    target_all: pd.DataFrame,
    eval_df: pd.DataFrame,
    target: str,
    features: list[str],
) -> dict[str, np.ndarray]:
    source_x = matrix(source_train, features)
    source_y = target_array(source_train, target)
    eval_x = matrix(eval_df, features)
    shot_x = matrix(target_shot, features) if len(target_shot) else np.empty((0, len(features)))
    shot_y = target_array(target_shot, target) if len(target_shot) else np.asarray([], dtype=float)
    target_ref_x = matrix(target_all, features)
    preds: dict[str, np.ndarray] = {
        "source_only_ridge": predict_ridge(source_x, source_y, eval_x),
    }
    if len(target_shot):
        preds["target_only_ridge"] = predict_ridge(shot_x, shot_y, eval_x)
    pooled_x = np.vstack([source_x, shot_x]) if len(target_shot) else source_x
    pooled_y = np.concatenate([source_y, shot_y]) if len(target_shot) else source_y
    preds["pooled_ridge"] = predict_ridge(pooled_x, pooled_y, eval_x)
    preds["tabular_knn"] = predict_knn(pooled_x, pooled_y, eval_x)

    coral_source_x = coral_align_source_to_target(source_x, target_ref_x)
    coral_train_x = np.vstack([coral_source_x, shot_x]) if len(target_shot) else coral_source_x
    coral_train_y = np.concatenate([source_y, shot_y]) if len(target_shot) else source_y
    preds["coral_ridge"] = predict_ridge(coral_train_x, coral_train_y, eval_x)

    source_weights = source_importance_weights(source_x, target_ref_x)
    if len(target_shot):
        iw_train_x = np.vstack([source_x, shot_x])
        iw_train_y = np.concatenate([source_y, shot_y])
        iw_weights = np.concatenate([source_weights, np.full(len(target_shot), 5.0)])
    else:
        iw_train_x = source_x
        iw_train_y = source_y
        iw_weights = source_weights
    preds["iw_ridge"] = predict_ridge(iw_train_x, iw_train_y, eval_x, sample_weight=iw_weights)
    return preds


def run_suite(
    manifest: pd.DataFrame,
    slice_registry: pd.DataFrame,
    slice_ids: list[str],
    shots: list[int],
    seeds: list[int],
) -> pd.DataFrame:
    manifest = manifest.copy()
    manifest["city"] = manifest["city"].astype(str).map(_compact)
    slice_registry = slice_registry.copy()
    slice_registry["city"] = slice_registry["city"].astype(str).map(_compact)
    rows: list[dict[str, Any]] = []
    for slice_id in slice_ids:
        cities = set(slice_registry[slice_registry["slice_id"].eq(slice_id)]["city"])
        slice_df = manifest[manifest["city"].isin(cities)].copy()
        for target in TARGETS:
            target_col = f"target_{target}"
            cur = slice_df[slice_df[target_col].astype(str).str.strip().ne("")].copy()
            features = feature_names_for_target(cur, target)
            if len(features) < 2:
                continue
            for holdout_city in sorted(cur["city"].unique()):
                source_train = cur[cur["city"].ne(holdout_city)].copy()
                target_city = cur[cur["city"].eq(holdout_city)].copy()
                if len(source_train) < 8 or len(target_city) < 6:
                    continue
                for shot in shots:
                    for seed in seeds:
                        target_shot, eval_df = make_city_split(target_city, shot=shot, seed=seed)
                        if len(eval_df) < 3:
                            continue
                        eval_y = target_array(eval_df, target)
                        predictions = predict_all_methods(source_train, target_shot, target_city, eval_df, target, features)
                        for method, pred in predictions.items():
                            rows.append(
                                {
                                    "slice_id": slice_id,
                                    "target": target,
                                    "shot": int(shot),
                                    "method": method,
                                    "method_label": METHOD_LABELS[method],
                                    "holdout_city": holdout_city,
                                    "seed": int(seed),
                                    "feature_count": int(len(features)),
                                    "source_train_count": int(len(source_train)),
                                    "target_shot_count": int(len(target_shot)),
                                    "eval_count": int(len(eval_df)),
                                    "r2": r2_score(eval_y, pred),
                                    "rmse": rmse(eval_y, pred),
                                    "mae": mae(eval_y, pred),
                                }
                            )
    return pd.DataFrame(rows)


def build_citywise(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw.copy()
    raw = raw.copy()
    if "method_label" not in raw.columns:
        raw["method_label"] = raw["method"].map(METHOD_LABELS).fillna(raw["method"])
    if "eval_count" not in raw.columns:
        raw["eval_count"] = 0
    grouped = (
        raw.groupby(["slice_id", "target", "shot", "method", "method_label", "holdout_city"], as_index=False)
        .agg(
            r2_city_mean=("r2", "mean"),
            rmse_city_mean=("rmse", "mean"),
            mae_city_mean=("mae", "mean"),
            seeds_count=("seed", "nunique"),
            eval_count_mean=("eval_count", "mean"),
        )
        .sort_values(["slice_id", "target", "shot", "method", "holdout_city"])
        .reset_index(drop=True)
    )
    return grouped


def summarize_citywise(raw: pd.DataFrame) -> pd.DataFrame:
    citywise = build_citywise(raw)
    if citywise.empty:
        return citywise
    rows: list[dict[str, Any]] = []
    for (slice_id, target, shot, method, method_label), group in citywise.groupby(
        ["slice_id", "target", "shot", "method", "method_label"], sort=True
    ):
        r2_vals = group["r2_city_mean"].to_numpy(dtype=float)
        rows.append(
            {
                "slice_id": slice_id,
                "target": target,
                "shot": int(shot),
                "method": method,
                "method_label": method_label,
                "cities_count": int(group["holdout_city"].nunique()),
                "positive_city_count": int(np.sum(r2_vals > 0.0)),
                "negative_city_count": int(np.sum(r2_vals < 0.0)),
                "positive_negative_cities": f"{int(np.sum(r2_vals > 0.0))}/{int(np.sum(r2_vals < 0.0))}",
                "negative_r2_ratio": float(np.mean(r2_vals < 0.0)),
                "mean_r2": float(np.mean(r2_vals)),
                "mean_rmse": float(np.mean(group["rmse_city_mean"].to_numpy(dtype=float))),
                "mean_mae": float(np.mean(group["mae_city_mean"].to_numpy(dtype=float))),
                "mean_eval_count": float(np.mean(group["eval_count_mean"].to_numpy(dtype=float))),
            }
        )
    return pd.DataFrame(rows).sort_values(["slice_id", "target", "shot", "method"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified few-shot baselines for `_122` structural slices.")
    parser.add_argument("--manifest", default="data/cityshiftbench_scale122_tile_targets.csv")
    parser.add_argument("--slice-registry", default="data/slices/paper_slice_registry_122.csv")
    parser.add_argument("--slice-ids", default="scale122")
    parser.add_argument("--shots", default="0,1,5,10")
    parser.add_argument("--seeds", default="7,19,42,61,97,123,211,307")
    parser.add_argument("--raw-output", default="results/fewshot_baseline_suite_122_raw.csv")
    parser.add_argument("--citywise-output", default="results/fewshot_baseline_suite_122_citywise.csv")
    parser.add_argument("--summary-output", default="results/fewshot_baseline_suite_122_summary.csv")
    args = parser.parse_args()

    manifest = pd.read_csv(ROOT / args.manifest)
    slice_registry = pd.read_csv(ROOT / args.slice_registry)
    slice_ids = [item.strip() for item in args.slice_ids.split(",") if item.strip()]
    shots = [int(item.strip()) for item in args.shots.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    raw = run_suite(manifest, slice_registry, slice_ids, shots, seeds)
    citywise = build_citywise(raw)
    summary = summarize_citywise(raw)

    for rel_path, frame in [(args.raw_output, raw), (args.citywise_output, citywise), (args.summary_output, summary)]:
        path = ROOT / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    payload = {
        "status": "completed",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "slice_ids": slice_ids,
        "shots": shots,
        "seeds": seeds,
        "methods": list(METHOD_LABELS),
        "raw_rows": int(len(raw)),
        "citywise_rows": int(len(citywise)),
        "summary_rows": int(len(summary)),
        "outputs": [args.raw_output, args.citywise_output, args.summary_output],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
