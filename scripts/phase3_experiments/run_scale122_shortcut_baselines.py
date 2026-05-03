#!/usr/bin/env python3
"""Run Scale-122 coordinate and city-metadata shortcut baselines.

These baselines are negative controls: they test whether strict cross-city
performance can be explained by geographic or registry metadata shortcuts.
They intentionally exclude OSM structural input features and all target columns.
"""

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

from scripts.phase3_experiments.run_fewshot_baseline_suite_122 import (
    mae,
    make_city_split,
    predict_ridge,
    r2_score,
    rmse,
    target_array,
)


TARGETS = ["road_segments", "intersection_nodes"]
SHORTCUT_METHOD_LABELS = {
    "coordinate_only_source_ridge": "Coordinate-only ridge",
    "coordinate_only_pooled_ridge": "Coordinate-only pooled ridge",
    "metadata_only_source_ridge": "City-metadata ridge",
    "metadata_only_pooled_ridge": "City-metadata pooled ridge",
    "coord_metadata_source_ridge": "Coordinate+metadata ridge",
    "coord_metadata_pooled_ridge": "Coordinate+metadata pooled ridge",
}


def _compact(text: Any) -> str:
    return str(text).strip().replace(" ", "")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _log1p_column(series: pd.Series) -> pd.Series:
    return np.log1p(pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0))


def prepare_shortcut_frame(manifest: pd.DataFrame, city_meta: pd.DataFrame) -> pd.DataFrame:
    manifest = manifest.copy()
    manifest["city"] = manifest["city"].astype(str).map(_compact)
    manifest["country"] = manifest.get("country", "").astype(str)
    meta_cols = ["city", "country", "region", "population", "gdp_per_capita", "overall_completeness"]
    meta = city_meta[[col for col in meta_cols if col in city_meta.columns]].copy()
    meta["city"] = meta["city"].astype(str).map(_compact)
    if "country" not in meta.columns:
        meta["country"] = ""
    meta["country"] = meta["country"].astype(str)
    for col in ["region", "population", "gdp_per_capita", "overall_completeness"]:
        if col in manifest.columns:
            manifest = manifest.drop(columns=[col])
    df = manifest.merge(meta, on=["city", "country"], how="left")
    df["latitude"] = pd.to_numeric(df.get("latitude", 0.0), errors="coerce").fillna(0.0)
    df["longitude"] = pd.to_numeric(df.get("longitude", 0.0), errors="coerce").fillna(0.0)
    df["population_log"] = _log1p_column(df.get("population", pd.Series(np.zeros(len(df)), index=df.index)))
    df["gdp_per_capita_log"] = _log1p_column(df.get("gdp_per_capita", pd.Series(np.zeros(len(df)), index=df.index)))
    df["overall_completeness_num"] = pd.to_numeric(
        df.get("overall_completeness", pd.Series(np.zeros(len(df)), index=df.index)),
        errors="coerce",
    ).fillna(0.0)
    region = df.get("region", pd.Series(["unknown"] * len(df), index=df.index)).fillna("unknown").astype(str)
    dummies = pd.get_dummies(region, prefix="region", dtype=float)
    if not dummies.empty:
        df = pd.concat([df, dummies], axis=1)
    return df


def shortcut_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    metadata = ["population_log", "gdp_per_capita_log", "overall_completeness_num"]
    metadata.extend([col for col in df.columns if col.startswith("region_")])
    sets = {
        "coordinate_only": ["latitude", "longitude"],
        "metadata_only": metadata,
        "coord_metadata": ["latitude", "longitude", *metadata],
    }
    filtered: dict[str, list[str]] = {}
    for name, columns in sets.items():
        kept = []
        for col in columns:
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if float(np.std(vals)) > 1e-12:
                kept.append(col)
        filtered[name] = kept
    return filtered


def feature_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)


def run_shortcut_suite(
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
    feature_sets = shortcut_feature_sets(manifest)
    rows: list[dict[str, Any]] = []
    for slice_id in slice_ids:
        cities = set(slice_registry[slice_registry["slice_id"].eq(slice_id)]["city"])
        slice_df = manifest[manifest["city"].isin(cities)].copy()
        for target in TARGETS:
            target_col = f"target_{target}"
            if target_col not in slice_df.columns:
                continue
            cur = slice_df[slice_df[target_col].astype(str).str.strip().ne("")].copy()
            for holdout_city in sorted(cur["city"].unique()):
                source_train = cur[cur["city"].ne(holdout_city)].copy()
                target_city = cur[cur["city"].eq(holdout_city)].copy()
                if len(source_train) < 2 or len(target_city) < 3:
                    continue
                for shot in shots:
                    for seed in seeds:
                        target_shot, eval_df = make_city_split(target_city, shot=shot, seed=seed)
                        if len(eval_df) < 2:
                            continue
                        eval_y = target_array(eval_df, target)
                        for feature_set, features in feature_sets.items():
                            if not features:
                                continue
                            source_x = feature_matrix(source_train, features)
                            source_y = target_array(source_train, target)
                            eval_x = feature_matrix(eval_df, features)
                            predictions = {
                                f"{feature_set}_source_ridge": predict_ridge(source_x, source_y, eval_x),
                            }
                            if len(target_shot):
                                shot_x = feature_matrix(target_shot, features)
                                shot_y = target_array(target_shot, target)
                                pooled_x = np.vstack([source_x, shot_x])
                                pooled_y = np.concatenate([source_y, shot_y])
                            else:
                                pooled_x = source_x
                                pooled_y = source_y
                            predictions[f"{feature_set}_pooled_ridge"] = predict_ridge(pooled_x, pooled_y, eval_x)
                            for method, pred in predictions.items():
                                rows.append(
                                    {
                                        "slice_id": slice_id,
                                        "target": target,
                                        "shot": int(shot),
                                        "method": method,
                                        "method_label": SHORTCUT_METHOD_LABELS[method],
                                        "holdout_city": holdout_city,
                                        "seed": int(seed),
                                        "feature_set": feature_set,
                                        "feature_count": int(len(features)),
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
    return (
        raw.groupby(["slice_id", "target", "shot", "method", "method_label", "feature_set", "holdout_city"], as_index=False)
        .agg(
            r2_city_mean=("r2", "mean"),
            rmse_city_mean=("rmse", "mean"),
            mae_city_mean=("mae", "mean"),
            seeds_count=("seed", "nunique"),
            eval_count_mean=("eval_count", "mean"),
            feature_count=("feature_count", "max"),
        )
        .sort_values(["slice_id", "target", "shot", "method", "holdout_city"])
        .reset_index(drop=True)
    )


def summarize_citywise(raw: pd.DataFrame) -> pd.DataFrame:
    citywise = build_citywise(raw)
    if citywise.empty:
        return citywise
    rows: list[dict[str, Any]] = []
    for (slice_id, target, shot, method, method_label, feature_set), group in citywise.groupby(
        ["slice_id", "target", "shot", "method", "method_label", "feature_set"],
        sort=True,
    ):
        r2_vals = group["r2_city_mean"].to_numpy(dtype=float)
        rows.append(
            {
                "slice_id": slice_id,
                "target": target,
                "shot": int(shot),
                "method": method,
                "method_label": method_label,
                "feature_set": feature_set,
                "feature_count": int(group["feature_count"].max()),
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
    parser = argparse.ArgumentParser(description="Run Scale-122 coordinate/city-metadata shortcut baselines.")
    parser.add_argument("--manifest", default="data/cityshiftbench_scale122_tile_targets.csv")
    parser.add_argument("--city-list", default="data/slices/paper_slice_registry_122.csv")
    parser.add_argument("--slice-registry", default="data/slices/paper_slice_registry_122.csv")
    parser.add_argument("--slice-ids", default="scale122")
    parser.add_argument("--shots", default="0,10")
    parser.add_argument("--seeds", default="7,19,42,61,97,123,211,307")
    parser.add_argument("--raw-output", default="results/scale122_shortcut_baselines_raw.csv")
    parser.add_argument("--citywise-output", default="results/scale122_shortcut_baselines_citywise.csv")
    parser.add_argument("--summary-output", default="results/scale122_shortcut_baselines_summary.csv")
    args = parser.parse_args()

    manifest = pd.read_csv(ROOT / args.manifest)
    city_meta = pd.read_csv(ROOT / args.city_list)
    slice_registry = pd.read_csv(ROOT / args.slice_registry)
    prepared = prepare_shortcut_frame(manifest, city_meta)
    raw = run_shortcut_suite(
        prepared,
        slice_registry,
        [item.strip() for item in args.slice_ids.split(",") if item.strip()],
        [int(item.strip()) for item in args.shots.split(",") if item.strip()],
        [int(item.strip()) for item in args.seeds.split(",") if item.strip()],
    )
    citywise = build_citywise(raw)
    summary = summarize_citywise(raw)
    for rel_path, frame in [(args.raw_output, raw), (args.citywise_output, citywise), (args.summary_output, summary)]:
        path = ROOT / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    print(
        json.dumps(
            {
                "status": "completed",
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "raw_rows": int(len(raw)),
                "citywise_rows": int(len(citywise)),
                "summary_rows": int(len(summary)),
                "outputs": [args.raw_output, args.citywise_output, args.summary_output],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
