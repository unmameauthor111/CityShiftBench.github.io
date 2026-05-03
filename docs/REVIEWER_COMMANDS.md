# Reviewer Commands

These commands are intended for a quick anonymous review check from the
repository root. They regenerate small or moderate outputs into
`tmp_review_outputs/` so that released result files are not overwritten.

## Environment

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Artifact Integrity Check

```bash
python scripts/verify_artifact.py
```

## Baseline Smoke Run

This command reruns the target-safe standard baselines for one target-shot
setting and one seed on the full Scale-122 city surface.

```bash
python scripts/phase3_experiments/run_fewshot_baseline_suite_122.py \
  --slice-ids scale122 \
  --shots 10 \
  --seeds 7 \
  --raw-output tmp_review_outputs/baseline_raw.csv \
  --citywise-output tmp_review_outputs/baseline_citywise.csv \
  --summary-output tmp_review_outputs/baseline_summary.csv
```

## Shortcut-Control Smoke Run

```bash
python scripts/phase3_experiments/run_scale122_shortcut_baselines.py \
  --shots 10 \
  --seeds 7 \
  --raw-output tmp_review_outputs/shortcut_raw.csv \
  --citywise-output tmp_review_outputs/shortcut_citywise.csv \
  --summary-output tmp_review_outputs/shortcut_summary.csv
```

## Released Result Files

The paper-facing result summaries are already included under `results/`.
SATCA seed-shard outputs can be merged with
`scripts/phase3_experiments/merge_satca_scale122_shards.py` when raw shard
directories are available. The released review artifact includes the merged
SATCA summaries and paired inference CSVs used by the manuscript.

