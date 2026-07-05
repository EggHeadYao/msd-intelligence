# Artist Distance Experiments

Benchmark scripts write elapsed wall-clock time to `results.csv`.

- `run_id`: repetition number supplied to the benchmark script.
- `source_id`: artist id used as the BFS source.
- `engine`: `mapreduce` or `spark`.
- `format`: `avro` or `parquet`.
- `elapsed_seconds`: read from `yarn application -status`.
- `verified`: `true` when the copied output matches the reference verifier.
