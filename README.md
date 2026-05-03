---
license: other
task_categories:
- tabular-regression
language:
- en
configs:
- config_name: scale122_targets
  data_files:
  - split: tile_targets
    path: data/cityshiftbench_scale122_tile_targets.csv
- config_name: scale122_splits
  data_files:
  - split: target_shot_registry
    path: data/splits/target_shot_registry_scale122.csv
- config_name: scale122_results
  data_files:
  - split: baseline_summary
    path: results/fewshot_baseline_suite_scale122_summary.csv
  - split: satca_paired
    path: results/satca_scale122_local_sharded_k010_paired_significance.csv
tags:
- urban-computing
- geospatial
- cross-city-transfer
- benchmark
- openstreetmap
- neurips-2026-ed
pretty_name: CityShiftBench Scale-122
size_categories:
- 1K<n<10K
---

# CityShiftBench Scale-122

CityShiftBench is an anonymous NeurIPS 2026 Evaluations and Datasets review
artifact for low-shot cross-city urban regression under strict city isolation.
The active Scale-122 surface contains 118 OSM-integrity-passing cities and
8,359 tile records. The paper-core targets are OSM-derived Road
(`target_road_segments`) and Connectivity (`target_intersection_nodes`).

## Contents

- `data/cityshiftbench_scale122_tile_targets.csv`: tile-level target and
  target-safe descriptor table used by the benchmark.
- `data/tile_manifest_122.csv`: tile registry and geometry descriptors.
- `data/splits/target_shot_registry_scale122.csv`: fixed support/evaluation
  row assignments for shots `0,1,5,10,20` and seeds
  `7,19,42,61,97,123,211,307`.
- `results/`: released benchmark summaries, city-wise scores, paired
  significance files, control summaries, and diagnostics.
- `scripts/`: executable baseline, shortcut-control, merge, and artifact-check
  entry points.
- `docs/`: data card, benchmark card, compute resources, license notes,
  reviewer commands, artifact release notes, and Croissant metadata.
- `croissant_cityshiftbench_scale122.json`: Croissant metadata with
  Responsible-AI fields for OpenReview upload.

## Quick Review Checks

Install dependencies with `pip install -r requirements.txt`, then run:

```bash
python scripts/verify_artifact.py
```

For executable smoke tests, see `docs/REVIEWER_COMMANDS.md`. The smoke tests
write to `tmp_review_outputs/` and do not overwrite the released result CSVs.

## Intended Use

Use this artifact to evaluate low-shot cross-city transfer while preserving the
registered city universe, target definitions, shot indices, city-wise metrics,
paired inference, and validity controls.

## Not Intended For

The artifact is not an operational urban planning system and should not be used
to rank cities or allocate resources without local validation.

## Licenses

Code is released under MIT. Author-created benchmark metadata and generated
result summaries are released under CC BY 4.0 unless otherwise stated.
OpenStreetMap-derived target fields remain subject to ODbL 1.0 and require
attribution to OpenStreetMap contributors.
