#!/usr/bin/env python3
"""Merge seed-sharded Scale-122 SATCA outputs and recompute paired evidence."""

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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _mean_or_zero(series: pd.Series) -> float:
    values = _safe_numeric(series).dropna()
    return float(values.mean()) if len(values) else 0.0


def _paired_t_p_value(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if len(delta) < 2:
        return math.nan
    std = float(np.std(delta, ddof=1))
    if std <= 1e-12:
        return 1.0 if abs(float(np.mean(delta))) <= 1e-12 else 0.0
    try:
        from scipy import stats

        return float(stats.ttest_1samp(delta, popmean=0.0).pvalue)
    except Exception:
        t_stat = abs(float(np.mean(delta)) / (std / math.sqrt(len(delta))))
        return float(math.erfc(t_stat / math.sqrt(2.0)))


def _wilcoxon_p_value(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    delta = delta[np.abs(delta) > 1e-12]
    if len(delta) == 0:
        return 1.0
    try:
        from scipy import stats

        return float(stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        positives = int(np.sum(delta > 0))
        n = int(len(delta))
        lower_tail = sum(math.comb(n, k) for k in range(0, min(positives, n - positives) + 1)) / (2**n)
        return float(min(1.0, 2.0 * lower_tail))


def _bootstrap_ci95(values: np.ndarray, seed: int = 122, draws: int = 5000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    if len(values) == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, len(values), size=(draws, len(values)))
    boot_means = values[sample_idx].mean(axis=1)
    return float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))


def _holm_adjust(p_values: pd.Series) -> list[float]:
    numeric = pd.to_numeric(p_values, errors="coerce")
    finite = [(idx, float(value)) for idx, value in numeric.items() if math.isfinite(float(value))]
    if not finite:
        return [math.nan for _ in numeric]
    ordered = sorted(finite, key=lambda item: item[1])
    m = len(ordered)
    adjusted_by_idx: dict[Any, float] = {}
    running = 0.0
    for rank, (idx, p_value) in enumerate(ordered, start=1):
        adjusted = min(1.0, (m - rank + 1) * p_value)
        running = max(running, adjusted)
        adjusted_by_idx[idx] = running
    return [adjusted_by_idx.get(idx, math.nan) for idx in numeric.index]


def add_holm_columns(paired_df: pd.DataFrame) -> pd.DataFrame:
    if paired_df.empty:
        return paired_df
    out = paired_df.copy()
    for source_col, target_col in [
        ("wilcoxon_p_delta_r2", "holm_p_delta_r2"),
        ("wilcoxon_p_delta_rmse", "holm_p_delta_rmse"),
        ("wilcoxon_p_delta_mae", "holm_p_delta_mae"),
    ]:
        if source_col in out.columns:
            out[target_col] = _holm_adjust(out[source_col])
    if "holm_p_delta_r2" in out.columns:
        out["holm_p"] = out["holm_p_delta_r2"]
    return out


def build_paired_rows(slice_id: str, citywise: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    df = citywise.copy()
    df["shot"] = df["shot"].astype(int)
    df["r2_city_mean"] = _safe_numeric(df["r2_city_mean"])
    df["rmse_city_mean"] = _safe_numeric(df["rmse_city_mean"])
    df["mae_city_mean"] = _safe_numeric(df["mae_city_mean"])
    df = df[df["shot"].eq(10)]
    for target, group in df.groupby("target", sort=True):
        source = group[group["strategy"].eq("source_only")][["holdout_city", "r2_city_mean", "rmse_city_mean", "mae_city_mean"]]
        satca = group[group["strategy"].eq("shiftaware_adapt")][["holdout_city", "r2_city_mean", "rmse_city_mean", "mae_city_mean"]]
        merged = source.merge(satca, on="holdout_city", suffixes=("_source", "_satca"))
        if merged.empty:
            continue
        delta_r2 = merged["r2_city_mean_satca"].to_numpy(dtype=float) - merged["r2_city_mean_source"].to_numpy(dtype=float)
        delta_rmse = merged["rmse_city_mean_satca"].to_numpy(dtype=float) - merged["rmse_city_mean_source"].to_numpy(dtype=float)
        delta_mae = merged["mae_city_mean_satca"].to_numpy(dtype=float) - merged["mae_city_mean_source"].to_numpy(dtype=float)
        r2_ci_lo, r2_ci_hi = _bootstrap_ci95(delta_r2, seed=122)
        rmse_ci_lo, rmse_ci_hi = _bootstrap_ci95(delta_rmse, seed=123)
        mae_ci_lo, mae_ci_hi = _bootstrap_ci95(delta_mae, seed=124)
        wilcoxon_p_r2 = _wilcoxon_p_value(delta_r2)
        wilcoxon_p_rmse = _wilcoxon_p_value(delta_rmse)
        wilcoxon_p_mae = _wilcoxon_p_value(delta_mae)
        rows.append(
            {
                "slice_id": slice_id,
                "target": target,
                "shot": 10,
                "comparison": "shiftaware_adapt_vs_source_only",
                "paired_city_count": int(len(merged)),
                "mean_source_r2": float(merged["r2_city_mean_source"].mean()),
                "mean_satca_r2": float(merged["r2_city_mean_satca"].mean()),
                "mean_delta_r2": float(np.mean(delta_r2)),
                "median_delta_r2": float(np.median(delta_r2)),
                "delta_r2_ci95_lo": r2_ci_lo,
                "delta_r2_ci95_hi": r2_ci_hi,
                "bootstrap_delta_r2_ci95_lo": r2_ci_lo,
                "bootstrap_delta_r2_ci95_hi": r2_ci_hi,
                "bootstrap_ci95_lo": r2_ci_lo,
                "bootstrap_ci95_hi": r2_ci_hi,
                "positive_delta_r2_count": int(np.sum(delta_r2 > 0)),
                "negative_delta_r2_count": int(np.sum(delta_r2 < 0)),
                "zero_delta_r2_count": int(np.sum(delta_r2 == 0)),
                "paired_t_p_delta_r2": _paired_t_p_value(delta_r2),
                "wilcoxon_p_delta_r2": wilcoxon_p_r2,
                "wilcoxon_p": wilcoxon_p_r2,
                "mean_delta_rmse": float(np.mean(delta_rmse)),
                "median_delta_rmse": float(np.median(delta_rmse)),
                "delta_rmse_ci95_lo": rmse_ci_lo,
                "delta_rmse_ci95_hi": rmse_ci_hi,
                "bootstrap_delta_rmse_ci95_lo": rmse_ci_lo,
                "bootstrap_delta_rmse_ci95_hi": rmse_ci_hi,
                "paired_t_p_delta_rmse": _paired_t_p_value(delta_rmse),
                "wilcoxon_p_delta_rmse": wilcoxon_p_rmse,
                "mean_delta_mae": float(np.mean(delta_mae)),
                "median_delta_mae": float(np.median(delta_mae)),
                "delta_mae_ci95_lo": mae_ci_lo,
                "delta_mae_ci95_hi": mae_ci_hi,
                "bootstrap_delta_mae_ci95_lo": mae_ci_lo,
                "bootstrap_delta_mae_ci95_hi": mae_ci_hi,
                "paired_t_p_delta_mae": _paired_t_p_value(delta_mae),
                "wilcoxon_p_delta_mae": wilcoxon_p_mae,
            }
        )
    return rows


def _existing_shard_dirs(shard_dirs: list[Path]) -> list[Path]:
    existing = []
    for path in shard_dirs:
        if (path / "kshot_transfer_raw.csv").exists() and (path / "run_config.json").exists():
            existing.append(path)
    return existing


def build_citywise_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    for col in ["r2", "rmse", "mae"]:
        df[col] = _safe_numeric(df[col])
    group_cols = ["target", "target_label", "shot", "strategy", "holdout_city"]
    optional_mean_cols = [
        "effective_shift",
        "blend_alpha",
        "shift_signal",
        "expert_weight_source",
        "expert_weight_affine",
        "expert_weight_weighted",
        "expert_error_source",
        "expert_error_affine",
        "expert_error_weighted",
    ]
    aggregations: dict[str, tuple[str, str]] = {
        "r2_city_mean": ("r2", "mean"),
        "rmse_city_mean": ("rmse", "mean"),
        "mae_city_mean": ("mae", "mean"),
        "seeds_count": ("seed", "nunique"),
    }
    for col in optional_mean_cols:
        if col in df.columns:
            aggregations[f"{col}_mean"] = (col, "mean")
    citywise = (
        df.groupby(group_cols, as_index=False)
        .agg(**aggregations)
        .sort_values(["target", "shot", "strategy", "holdout_city"])
        .reset_index(drop=True)
    )
    return citywise


def build_summary_from_citywise(citywise: pd.DataFrame) -> pd.DataFrame:
    if citywise.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (target, target_label, shot, strategy), group in citywise.groupby(
        ["target", "target_label", "shot", "strategy"],
        sort=True,
    ):
        r2_vals = group["r2_city_mean"].to_numpy(dtype=float)
        worst_idx = int(np.nanargmin(r2_vals)) if len(r2_vals) else 0
        row: dict[str, Any] = {
            "target": target,
            "target_label": target_label,
            "shot": int(shot),
            "strategy": strategy,
            "cities_count": int(group["holdout_city"].nunique()),
            "positive_city_count": int(np.sum(r2_vals > 0.0)),
            "negative_city_count": int(np.sum(r2_vals < 0.0)),
            "negative_r2_ratio": float(np.mean(r2_vals < 0.0)),
            "mean_r2": float(np.nanmean(r2_vals)),
            "std_r2": float(np.nanstd(r2_vals, ddof=1)) if len(r2_vals) > 1 else 0.0,
            "mean_rmse": float(group["rmse_city_mean"].mean()),
            "mean_mae": float(group["mae_city_mean"].mean()),
            "worst_city": str(group.iloc[worst_idx]["holdout_city"]),
            "worst_city_r2": float(group.iloc[worst_idx]["r2_city_mean"]),
        }
        for col in group.columns:
            if col.endswith("_mean") and col not in row:
                row[col] = _mean_or_zero(group[col])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["target", "shot", "strategy"]).reset_index(drop=True)


def merge_shards(shard_dirs: list[Path], slice_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    existing = _existing_shard_dirs(shard_dirs)
    if not existing:
        raise FileNotFoundError("No shard directories contain both kshot_transfer_raw.csv and run_config.json.")

    raw_frames = []
    configs = []
    for shard_dir in existing:
        raw = pd.read_csv(shard_dir / "kshot_transfer_raw.csv")
        raw["shard_dir"] = str(shard_dir)
        raw_frames.append(raw)
        configs.append(_read_json(shard_dir / "run_config.json"))

    merged_raw = pd.concat(raw_frames, ignore_index=True)
    merged_raw = merged_raw.drop_duplicates(
        subset=["target", "holdout_city", "seed", "shot", "strategy"],
        keep="last",
    ).reset_index(drop=True)
    citywise = build_citywise_from_raw(merged_raw)
    summary = build_summary_from_citywise(citywise)
    paired = add_holm_columns(pd.DataFrame(build_paired_rows(slice_id, citywise)))

    seeds = sorted({int(seed) for config in configs for seed in config.get("seeds", [])})
    targets = sorted({str(target) for config in configs for target in config.get("targets", [])})
    shots = sorted({int(shot) for config in configs for shot in config.get("shots", [])})
    cities = sorted({str(city) for config in configs for city in config.get("main_cities_resolved", [])})
    audit = {
        "status": "completed",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "slice_id": slice_id,
        "shard_dirs_requested": [str(path) for path in shard_dirs],
        "shard_dirs_merged": [str(path) for path in existing],
        "raw_rows": int(len(merged_raw)),
        "citywise_rows": int(len(citywise)),
        "summary_rows": int(len(summary)),
        "paired_rows": int(len(paired)),
        "cities": cities,
        "shots": shots,
        "seeds": seeds,
        "targets": targets,
    }
    return summary, citywise, paired, audit


def write_merged_output_dir(output_dir: Path, summary: pd.DataFrame, citywise: pd.DataFrame, raw: pd.DataFrame, audit: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "kshot_transfer_summary.csv", index=False)
    citywise.to_csv(output_dir / "kshot_transfer_citywise.csv", index=False)
    raw.to_csv(output_dir / "kshot_transfer_raw.csv", index=False)
    config = {
        "main_cities_resolved": audit["cities"],
        "shots": audit["shots"],
        "seeds": audit["seeds"],
        "targets": audit["targets"],
        "merged_from_shards": audit["shard_dirs_merged"],
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def collect_raw_for_write(shard_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for shard_dir in _existing_shard_dirs(shard_dirs):
        raw = pd.read_csv(shard_dir / "kshot_transfer_raw.csv")
        raw["shard_dir"] = str(shard_dir)
        frames.append(raw)
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["target", "holdout_city", "seed", "shot", "strategy"], keep="last")
        .reset_index(drop=True)
    )


def write_report(path: Path, audit: dict[str, Any], paired: pd.DataFrame) -> None:
    lines = [
        "# Scale-122 Sharded SATCA Merge Audit",
        "",
        f"- Generated UTC: {audit['generated_utc']}",
        f"- Slice: `{audit['slice_id']}`",
        f"- Shards merged: `{len(audit['shard_dirs_merged'])}`",
        f"- Cities: `{len(audit['cities'])}`",
        f"- Shots: `{audit['shots']}`",
        f"- Seeds: `{audit['seeds']}`",
        f"- Targets: `{audit['targets']}`",
        f"- Raw rows: `{audit['raw_rows']}`",
        f"- Citywise rows: `{audit['citywise_rows']}`",
        "",
        "## k=10 paired SATCA-source contrast",
    ]
    if paired.empty:
        lines.append("- No paired rows were generated.")
    else:
        for _, row in paired.sort_values(["target"]).iterrows():
            lines.append(
                "- `{target}`: n={n}, source={source:.3f}, satca={satca:.3f}, "
                "DeltaR2={delta:+.3f}, CI95=[{lo:.3f}, {hi:.3f}], HolmP={holm:.3f}, +/-={pos}/{neg}".format(
                    target=row["target"],
                    n=int(row["paired_city_count"]),
                    source=float(row["mean_source_r2"]),
                    satca=float(row["mean_satca_r2"]),
                    delta=float(row["mean_delta_r2"]),
                    lo=float(row.get("bootstrap_delta_r2_ci95_lo", math.nan)),
                    hi=float(row.get("bootstrap_delta_r2_ci95_hi", math.nan)),
                    holm=float(row.get("holm_p", math.nan)),
                    pos=int(row["positive_delta_r2_count"]),
                    neg=int(row["negative_delta_r2_count"]),
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge seed-sharded Scale-122 SATCA outputs.")
    parser.add_argument("--slice-id", default="scale122_sharded_k010")
    parser.add_argument("--shard-dirs", nargs="+", required=True)
    parser.add_argument(
        "--merged-output-dir",
        default="review_runs/satca_scale122_sharded_k010",
    )
    parser.add_argument("--summary-output", default="results/satca_scale122_sharded_k010_summary.csv")
    parser.add_argument("--citywise-output", default="results/satca_scale122_sharded_k010_citywise.csv")
    parser.add_argument("--paired-output", default="results/satca_scale122_sharded_k010_paired_significance.csv")
    parser.add_argument("--audit-output", default="reports/SATCA_SCALE122_SHARDED_K010_MERGE_AUDIT.md")
    args = parser.parse_args()

    shard_dirs = [ROOT / path for path in args.shard_dirs]
    summary, citywise, paired, audit = merge_shards(shard_dirs, args.slice_id)
    merged_raw = collect_raw_for_write(shard_dirs)
    write_merged_output_dir(ROOT / args.merged_output_dir, summary, citywise, merged_raw, audit)

    for rel_path, frame in [
        (args.summary_output, summary),
        (args.citywise_output, citywise),
        (args.paired_output, paired),
    ]:
        path = ROOT / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    write_report(ROOT / args.audit_output, audit, paired)
    print(
        json.dumps(
            {
                "status": "completed",
                "outputs": [args.summary_output, args.citywise_output, args.paired_output, args.audit_output],
                "summary": audit,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
