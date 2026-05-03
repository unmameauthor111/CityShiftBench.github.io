# CityShiftBench Compute Resources

This file records the compute profile used for the Scale-122 NeurIPS
Evaluations and Datasets review artifact.

## Environment

- Main experiments are CPU-oriented and do not require a GPU.
- Server reruns used a conda environment named `cityshiftbench`, but the
  released smoke commands run with the active Python interpreter once
  `requirements.txt` is installed.
- Python dependencies are listed in `requirements.txt`.
- Quick reviewer commands are listed in `docs/REVIEWER_COMMANDS.md`.

## Full-Scale Baseline Suite

Command:

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

Slurm request used in the review run:

- Workers: CPU only.
- CPU cores: 32.
- Memory: 128 GB.
- Time limit: 12 hours.
- Output summary: `results/fewshot_baseline_suite_scale122_summary.csv`.

## Scale-122 SATCA Shards

SATCA was run as seed shards to make long jobs recoverable. Each target was
split into seed group A (`7,19,42,61`) and seed group B
(`97,123,211,307`).

The anonymous artifact includes the merged SATCA summaries and paired
inference files used by the manuscript. If raw shard directories are available,
the released merge script recomputes the summary, citywise, and paired
inference CSVs:

```bash
python scripts/phase3_experiments/merge_satca_scale122_shards.py \
  --slice-id scale122_sharded_k010 \
  --shard-dirs path/to/seedA path/to/seedB \
  --merged-output-dir tmp_review_outputs/satca_merged \
  --summary-output tmp_review_outputs/satca_summary.csv \
  --citywise-output tmp_review_outputs/satca_citywise.csv \
  --paired-output tmp_review_outputs/satca_paired.csv \
  --audit-output tmp_review_outputs/satca_merge_audit.md
```

Slurm request used per SATCA shard:

- Workers: CPU only.
- CPU cores: 48.
- Memory: 256 GB.
- Time limit: 24 hours.
- Merge job: 8 CPU cores, 32 GB memory, 2 hours.

The merged paired inference files are:

- `results/satca_scale122_local_sharded_k010_summary.csv`
- `results/satca_scale122_local_sharded_k010_citywise.csv`
- `results/satca_scale122_local_sharded_k010_paired_significance.csv`

## Paper Tables

Paper-facing LaTeX tables are included under `paper_tables/`. They are derived
from the released CSV summaries in `results/` and are provided to make the
reported numbers easy to audit against the manuscript PDF.
