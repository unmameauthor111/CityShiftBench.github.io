# Server Runbook: Scale-122 Structural Experiments

This runbook records the compute shape used for the larger Scale-122 reruns.
It is intentionally scheduler-neutral so the anonymous review artifact does not
depend on private cluster paths or local job wrappers.

## Upload

Copy this repository to a CPU server or cluster login node. If using `rsync`,
exclude transient files:

```bash
rsync -av --exclude ".git" --exclude ".venv" --exclude "tmp_review_outputs" ./ <USER>@<HOST>:/path/to/CityShiftBench/
```

## Environment

```bash
cd /path/to/CityShiftBench
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

The experiments are CPU-oriented. The review run used 32 CPU cores and 128 GB
memory for the standard baseline suite, and larger 48-core/256-GB seed shards
for the SATCA adaptation probe.

## Standard Baseline Suite

```bash
python scripts/phase3_experiments/run_fewshot_baseline_suite_122.py \
  --manifest data/cityshiftbench_scale122_tile_targets.csv \
  --slice-ids scale122 \
  --shots 0,1,5,10 \
  --seeds 7,19,42,61,97,123,211,307 \
  --raw-output results/fewshot_baseline_suite_scale122_raw.csv \
  --citywise-output results/fewshot_baseline_suite_scale122_citywise.csv \
  --summary-output results/fewshot_baseline_suite_scale122_summary.csv
```

Expected released summaries:

```text
results/fewshot_baseline_suite_scale122_citywise.csv
results/fewshot_baseline_suite_scale122_summary.csv
```

## Shortcut-Control Suite

```bash
python scripts/phase3_experiments/run_scale122_shortcut_baselines.py \
  --shots 0,10 \
  --seeds 7,19,42,61,97,123,211,307 \
  --raw-output results/scale122_shortcut_baselines_raw.csv \
  --citywise-output results/scale122_shortcut_baselines_citywise.csv \
  --summary-output results/scale122_shortcut_baselines_summary.csv
```

## SATCA Merge

The anonymous artifact includes merged SATCA result files. If raw shard
directories are available in the expected `kshot_transfer_*.csv` format, merge
them with:

```bash
python scripts/phase3_experiments/merge_satca_scale122_shards.py \
  --slice-id scale122_sharded_k010 \
  --shard-dirs path/to/seedA path/to/seedB \
  --merged-output-dir review_runs/satca_merged \
  --summary-output results/satca_scale122_local_sharded_k010_summary.csv \
  --citywise-output results/satca_scale122_local_sharded_k010_citywise.csv \
  --paired-output results/satca_scale122_local_sharded_k010_paired_significance.csv \
  --audit-output review_runs/SATCA_SCALE122_SHARDED_K010_MERGE_AUDIT.md
```

## Paper Decision Rule

Use the full-scale baseline suite as main-text evidence if it completes on the
118 integrity-passing cities and the ordinary baselines remain weak under city
isolation. Use SATCA paired inference as a bounded adaptation claim only when
matched city-wise contrasts retain positive confidence intervals after
corrected testing. Otherwise, keep SATCA outputs as diagnostic evidence.

