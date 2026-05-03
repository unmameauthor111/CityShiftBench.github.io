# Data Card: CityShiftBench

## Dataset Summary

CityShiftBench is a low-shot cross-city urban regression benchmark with a
122-city registry and an active 118-city Scale-122 evaluation surface after
OSM-integrity checks. The paper-core targets are structural:

- Road: `road_segments`
- Connectivity: `intersection_nodes`

The benchmark fixes target-shot budgets, city-isolated splits, city-wise
reporting, paired inference fields, and guardrail controls.

## Data Sources

- OpenStreetMap-derived structural truth layers for Road and Connectivity.
- City and country metadata used for coverage descriptors and shortcut
  controls.
- Optional Sentinel-derived features and broader diagnostics outside the
  paper-core claim.
- Country-level GDP per capita proxy from the World Bank Open Data API,
  indicator `NY.GDP.PCAP.CD`, used only as descriptive metadata.

GDP is a country-level proxy and must not be described as city GDP or as a
causal socioeconomic label.

## Active Scale-122 Characteristics

- Registered cities: 122.
- Active integrity-passing cities: 118.
- Active tile count: 8,359.
- Countries: 68.
- Continents: six.
- Shot budgets: `k = 0, 1, 5, 10, 20`.
- Inference unit: target city.

## Target-Safe Feature Policy

For Road and Connectivity experiments, direct target columns and direct
cross-target OSM structural proxies are excluded from the corresponding feature
set. In code, this is represented by the `target_safe` feature policy and the
target exclusion sets in:

- `scripts/phase3_experiments/run_fewshot_baseline_suite_122.py`
- `scripts/phase3_experiments/run_scale122_shortcut_baselines.py`

This policy is part of the evaluation contract and should be preserved by
future benchmark users.

## Known Data Characteristics

- OSM completeness varies substantially by city and region.
- Structural targets are auditable but do not measure mobility, economic
  welfare, policy quality, or human outcomes.
- City-wise `R^2` can be brittle when a target city has low evaluation-set
  variance.
- Some broader Sentinel validation rows are marked unreadable and should be
  treated as quality flags until revalidated.

## Recommended Use

- Evaluate low-shot cross-city transfer under locked split registries.
- Report city-wise paired inference on fixed target-city evaluation sets.
- Use Scale-122 as the main full-scale surface.
- Use narrower slices and diagnostics for auditability, not for selecting the
  best-looking headline result.

## Not Recommended

- Do not use random or in-city splits as evidence of cross-city transfer.
- Do not change shot indices, city registries, target definitions, or reporting
  fields when making direct benchmark comparisons.
- Do not use city-level predictions for operational planning without local
  validation.
- Do not interpret GDP-proxy groups causally.

## Key Artifacts

- `data/tile_manifest_122.csv`
- `data/slices/paper_slice_registry_122.csv`
- `data/targets_all_122.csv`
- `results/fewshot_baseline_suite_scale122_summary.csv`
- `results/satca_scale122_local_sharded_k010_paired_significance.csv`
- `paper_tables/*.tex`
- `docs/CITYSHIFTBENCH_CROISSANT_METADATA.json`
- `docs/LICENSES.md`
- `docs/COMPUTE_RESOURCES.md`
