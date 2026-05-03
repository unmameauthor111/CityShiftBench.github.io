# Benchmark Card: CityShiftBench

## Benchmark Purpose

CityShiftBench evaluates low-shot cross-city urban regression under strict city
isolation. It is designed for claim-bounded evaluation: a method should compete
by keeping the same city registry, target definitions, fixed shot budgets,
city-wise metrics, paired inference, and validity controls.

## Active Review Surface

- Registered benchmark universe: 122 cities.
- Active main evaluation: 118 OSM-integrity-passing cities.
- Tile count: 8,359 tiles.
- Countries: 68.
- Continents: six.
- Paper-core targets: Road (`road_segments`) and Connectivity
  (`intersection_nodes`).
- Shot budgets: `k = 0, 1, 5, 10, 20`.
- Inference unit: target city, not pooled tiles.

## Evidence Roles

- **Scale-122**: main full-scale evidence for standard baselines, SATCA
  adaptation-probe contrasts, shot curves, ablations, shortcut controls, and
  failure diagnostics.
- **Structural-43**: historical structural audit slice retained for
  comparison with earlier development runs.
- **Audited-9**: high-control sanity-check slice.
- **Heldout-17**: corroboration slice used for frozen-control and historical
  checks.
- **150-city diagnostics**: exploratory scale, zero-inflation, temporal,
  multi-task, and data-quality diagnostics. These are not promoted to headline
  significance claims in the NeurIPS ED draft.

## Claim Policy

- Headline inference is restricted to matched city-wise contrasts on fixed
  evaluation sets.
- Scale-122 carries the retained full-scale claim.
- Road and Connectivity both show positive full-scale SATCA paired gains at
  `k=10`, but the claim remains bounded because Connectivity has more
  city-level regressions and component ablations do not isolate mechanism
  terms.
- Random and in-city splits may be reported only as protocol-sensitivity
  diagnostics, not as cross-city transfer evidence.
- GDP-proxy and region strata are descriptive metadata, not causal or fairness
  labels.

## Core Metrics

- City-wise `R^2`, RMSE, and MAE.
- Negative-`R^2` ratio.
- Positive/negative city counts.
- Bootstrap confidence intervals over cities.
- Wilcoxon signed-rank tests and Holm-adjusted p-values for paired claims.

## Validity Controls

- Target-safe feature policy excludes Road and Connectivity target columns and
  direct target proxies from the feature set for the corresponding target.
- Coordinate-only and city-metadata-only shortcut baselines are reported as
  negative controls.
- Frozen representation controls are retained as guardrails.
- Shuffled-label controls check that the pipeline does not manufacture
  plausible performance from broken supervision.

## Known Failure Modes

- Cross-city transfer is heterogeneous: average gains can coexist with target
  cities that regress.
- OSM mapping completeness varies across regions and cities.
- City-wise `R^2` is sensitive to target variance and fixed evaluation tiles.
- Component ablations are nearly tied with full SATCA, so the current evidence
  supports an adaptation gain but not a mechanism proof.

## Review Checklist

- Verify that headline numbers map to released CSV artifacts.
- Verify that Scale-122, Structural-43, Audited-9, and Heldout-17 are described
  with separate evidence roles.
- Verify that target-safe feature exclusions are preserved.
- Verify that OSM/ODbL attribution, compute resources, and Croissant metadata
  are included in the review package.
