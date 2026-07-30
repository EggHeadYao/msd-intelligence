# Artist Distance Experiments

`pilot_results.csv` preserves the initial one-run comparison. Its `elapsed_seconds` value came from YARN application status and should be treated as preliminary evidence only.

The complete benchmark writes `results.csv`, `summary.csv`, and `comparisons.csv`. All formal runs use one fixed source artist and the full directed artist graph.

## Detailed Results

- `run_id`: repetition number.
- `order_index`: position of the combination within that repetition.
- `source_id`: fixed BFS source artist.
- `engine`: `mapreduce` or `spark`.
- `format`: `avro` or `parquet`.
- `wall_seconds`: full submitted command duration measured by `/usr/bin/time`.
- `yarn_seconds`: sum of YARN application durations.
- `memory_seconds`: aggregate YARN memory allocation.
- `vcore_seconds`: aggregate YARN CPU allocation.
- `application_count`: number of YARN applications submitted by the BFS.
- `expected_total`: number of vertices in the reference graph.
- `reachable`: vertices reachable from the source.
- `unreachable`: vertices outside the source's directed reachable component.
- `max_distance`: largest finite shortest-path distance.
- `verified`: whether the complete output passed the independent verifier.

## Aggregate Results

`summary.csv` reports robust timing statistics and resource use for each combination. `comparisons.csv` reports median speedup as `baseline wall time / candidate wall time`, so a value greater than one favors the candidate.

## Generate Figure

```bash
python3 src/artistdistance/experiments/scripts/plot_performance.py
```

The script writes presentation-ready PDF and PNG versions of `artistdistance_performance` to `src/slides/img/`.
