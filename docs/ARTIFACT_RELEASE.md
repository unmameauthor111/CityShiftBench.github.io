# CityShiftBench NeurIPS ED Review Artifact

This file defines the minimum artifact expected for anonymous NeurIPS
Evaluations and Datasets review.

## Required Review Contents

- The anonymous paper PDF is submitted through OpenReview. This repository is
  the companion review artifact for data, scripts, metadata, and released
  result summaries.
- Core data registries: `data/tile_manifest_122.csv`,
  `data/slices/paper_slice_registry_122.csv`, and active integrity records.
- Main result summaries: Scale-122 baseline summaries, SATCA summaries,
  paired significance CSVs, ablation CSVs, shortcut-control CSVs, and table
  LaTeX files under `paper_tables/`.
- Executable checks and reruns: `scripts/verify_artifact.py`,
  `scripts/phase3_experiments/run_fewshot_baseline_suite_122.py`,
  `scripts/phase3_experiments/run_scale122_shortcut_baselines.py`, and
  `scripts/phase3_experiments/merge_satca_scale122_shards.py`.
- Documentation: `docs/DATA_CARD.md`, `docs/BENCHMARK_CARD.md`,
  `docs/COMPUTE_RESOURCES.md`, `docs/LICENSES.md`, and
  `docs/REVIEWER_COMMANDS.md`.
- Croissant metadata: the review artifact should include core Croissant fields,
  source/provenance fields, and Responsible-AI fields, including whether the
  benchmark contains synthetic data.

## Anonymization

For double-blind review, the public artifact should avoid author names in:

- repository owner or organization name,
- README badges or contact blocks,
- commit metadata in a submitted archive when feasible,
- absolute local paths in documentation.

If an anonymized GitHub repository is used, the OpenReview submission should
link to that repository directly. If the repository is not ready, upload an
anonymous ZIP artifact through OpenReview and include the same contents listed
above. The manuscript PDF itself should remain anonymous; code and data links
belong in the OpenReview metadata fields or anonymous supplement.

## Large Files

GitHub should not be used for raw OSM JSON dumps, large Sentinel rasters,
Photoshop/PSB editing files, or generated preview PDFs. Release these through a
dataset host such as Hugging Face, Dataverse, OpenML, or Kaggle. If any asset
exceeds 4 GB, include a small sample that reviewers can inspect.

## Rebuild Commands

From the project root:

```bash
python scripts/verify_artifact.py
python scripts/phase3_experiments/run_fewshot_baseline_suite_122.py --slice-ids scale122 --shots 10 --seeds 7 --raw-output tmp_review_outputs/baseline_raw.csv --citywise-output tmp_review_outputs/baseline_citywise.csv --summary-output tmp_review_outputs/baseline_summary.csv
python scripts/phase3_experiments/run_scale122_shortcut_baselines.py --shots 10 --seeds 7 --raw-output tmp_review_outputs/shortcut_raw.csv --citywise-output tmp_review_outputs/shortcut_citywise.csv --summary-output tmp_review_outputs/shortcut_summary.csv
```

For full experiment reruns and compute assumptions, use
`docs/COMPUTE_RESOURCES.md` and `docs/SERVER_SCALE122_RUNBOOK.md`.
