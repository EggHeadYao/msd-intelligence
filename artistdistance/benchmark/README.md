# Artist Distance Benchmark

Run from `p1team02/artistdistance` after Hadoop commands are available:

## `run_one.sh`

Run one engine/format pair:

```bash
./benchmark/run_one.sh mapreduce avro 1
./benchmark/run_one.sh mapreduce parquet 1
./benchmark/run_one.sh spark avro 1
./benchmark/run_one.sh spark parquet 1
```

Arguments:

- `engine`: `mapreduce` or `spark`.
- `format`: `avro` or `parquet`.
- `run_id`: repetition number written to `experiments/results.csv`.

Use `SOURCE_ID` to choose a BFS source artist:

```bash
SOURCE_ID=AR002UA1187B9A637D ./benchmark/run_one.sh spark parquet 1
```

## `run_all.sh`

Run all four required combinations:

```bash
./benchmark/run_all.sh 3
```

The optional argument is the number of repetitions. Each repetition runs:

- `mapreduce avro`
- `mapreduce parquet`
- `spark avro`
- `spark parquet`

`SOURCE_ID` also applies to `run_all.sh`.

## Benchmark Workflow

### Restart and Data Setup

`run_one.sh` restarts HDFS and YARN with `stop-all.sh` and `start-all.sh` before each submitted task, then waits for HDFS to leave safemode.

Inputs are uploaded to HDFS under `/user/$USER/artistdistance-benchmark/input`. Outputs are written to `/user/$USER/artistdistance-benchmark/output`.

### YARN Execution and Timing

MapReduce jobs are submitted with `mapreduce.framework.name=yarn`. Spark jobs are submitted with `spark-submit --master yarn`.

Timing is read from `yarn application -status`. MapReduce BFS submits one application per BFS job, so its time is the sum of those application durations.

### Verification and Cleanup

After timing is recorded, `run_one.sh` copies the output back to `experiments/output`, verifies it against the reference BFS result, and records the result in the `verified` column. Verification time is not included in `elapsed_seconds`. If verification succeeds, the script removes this run's HDFS output, local output copy, newly uploaded input, and matching YARN staging directories.
