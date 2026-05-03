# CityShiftBench License and Attribution Notes

This document summarizes the license position used for the NeurIPS
Evaluations and Datasets review package. It complements the root `LICENSE`
file.

## Original CityShiftBench Materials

- Code, scripts, and experiment wrappers: MIT License.
- Author-created benchmark metadata, split registries, shot registries,
  result summaries, generated tables, and generated figures: CC BY 4.0 for
  research reuse unless a file states otherwise.
- Anonymous review artifacts should keep author names out of URLs, package
  names, and repository owner metadata where possible.

## OpenStreetMap-Derived Targets

CityShiftBench Road and Connectivity targets are derived from OpenStreetMap
truth layers. These derived fields remain subject to the Open Data Commons
Open Database License (ODbL) 1.0.

Required attribution:

```text
OpenStreetMap contributors
```

Relevant upstream terms:

- OpenStreetMap copyright and license: `https://www.openstreetmap.org/copyright`
- ODbL 1.0: `https://opendatacommons.org/licenses/odbl/1-0/`

The benchmark code license does not relicense OSM-derived database material.

## Other Data Sources

- World Bank GDP-per-capita proxy fields use the World Bank Open Data API,
  indicator `NY.GDP.PCAP.CD`. These values are descriptive country-level
  proxies and are not city-level causal or fairness labels.
- Sentinel-derived feature descriptors, when included, retain their upstream
  data terms. The Scale-122 core claim in the paper is based on OSM-derived
  structural targets and target-safe feature policies.
- Pretrained representation controls retain their model/provider licenses and
  are included only as evaluation controls, not as newly released models.

## Release Boundary

The public release should include:

- `LICENSE`
- `docs/LICENSES.md`
- `docs/DATA_CARD.md`
- `docs/BENCHMARK_CARD.md`
- `docs/CITYSHIFTBENCH_CROISSANT_METADATA.json`
- `docs/COMPUTE_RESOURCES.md`

Large raw assets may be hosted outside GitHub, but the repository should keep a
small inspectable sample and the scripts needed to regenerate the paper tables
from the released summaries.
