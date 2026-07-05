# Artist Distance Benchmark

Run from `p1team02/artistdistance` after Hadoop commands are available:

```bash
./benchmark/run_one.sh mapreduce avro 1
./benchmark/run_one.sh mapreduce parquet 1
./benchmark/run_one.sh spark avro 1
./benchmark/run_one.sh spark parquet 1
./benchmark/run_all.sh 3
```

- `run_one.sh`: runs one engine/format pair and appends elapsed seconds to `experiments/results.csv`.
- `run_all.sh`: runs the four required combinations.
- `SOURCE_ID`: optional environment variable for changing the BFS source artist.

`run_one.sh` restarts HDFS and YARN with `stop-all.sh` and `start-all.sh`
before each submitted task.

Inputs are uploaded to HDFS under `/user/$USER/artistdistance-benchmark/input`.
Outputs are written to `/user/$USER/artistdistance-benchmark/output`.

MapReduce jobs are submitted with `mapreduce.framework.name=yarn`. Spark jobs
are submitted with `spark-submit --master yarn`.

Timing is read from `yarn application -status`. MapReduce BFS submits one
application per BFS job, so its time is the sum of those application durations.

After timing is recorded, `run_one.sh` copies the output back to
`experiments/output`, verifies it against the reference BFS result, and records
the result in the `verified` column. Verification time is not included in
`elapsed_seconds`. If verification succeeds, the script removes this run's HDFS
output, local output copy, newly uploaded input, and matching YARN staging
directories.
