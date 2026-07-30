# Artist Distance Benchmark

Run the commands from `src/artistdistance` after Hadoop, HDFS, YARN, Spark, Java, and Maven are available. From the repository root, enter the module with `cd src/artistdistance`.

## Complete Experiment

Run five repetitions of all four engine/format combinations:

```bash
./benchmark/run_experiment.sh 5
```

`run_all.sh` is a compatibility alias for the same command. The experiment uses `ARGUACZ1187FB3F35C` by default because its BFS covers most of the full graph. Set `SOURCE_ID` to use another fixed source for every run:

```bash
SOURCE_ID=AR002UA1187B9A637D ./benchmark/run_experiment.sh 5
```

The script builds the Maven project, creates both full graph inputs from SQLite when they are absent, refreshes the corresponding HDFS inputs, and runs MapReduce/Avro, MapReduce/Parquet, Spark/Avro, and Spark/Parquet. Set `REBUILD_INPUTS=true` to recreate existing local inputs explicitly. The execution order rotates between repetitions to reduce order bias. Before each measured task, `run_one.sh` restarts HDFS and YARN and waits for the previously observed number of NodeManagers to return. Set `YARN_EXPECTED_RUNNING_NODES` when the cluster was not fully running before the command started.

Verified combinations already present in `experiments/results.csv` are skipped, so the same command resumes an interrupted experiment. Do not change `SOURCE_ID` while resuming the same CSV.

## One Combination

Run a single combination when debugging:

```bash
./benchmark/run_one.sh mapreduce avro 1
./benchmark/run_one.sh mapreduce parquet 1
./benchmark/run_one.sh spark avro 1
./benchmark/run_one.sh spark parquet 1
```

The arguments are the engine, storage format, and positive repetition number. `RESULTS_CSV` can redirect the row to a separate result file.

## Measurement and Verification

- `wall_seconds` is measured by `/usr/bin/time` around the complete `hadoop jar` or `spark-submit` command. It includes all BFS iterations and framework orchestration, but excludes cluster restart, input upload, output download, verification, and cleanup.
- `yarn_seconds` is the sum of the completed YARN application durations. MapReduce launches one application per BFS iteration, while Spark normally uses one application for the complete BFS.
- `memory_seconds` and `vcore_seconds` are aggregate allocations reported by YARN across the applications.
- `expected_total`, `reachable`, `unreachable`, and `max_distance` come from the independent reference BFS verifier.
- `verified` is true only when every output distance and predecessor is accepted by the verifier.

Successful runs remove their local output copy, HDFS output, and staging files. The two common HDFS inputs remain available between combinations and are removed when the complete experiment finishes. A failed run keeps its output for diagnosis and stops the experiment.

## Result Summary

The complete experiment writes:

- `experiments/results.csv`: one detailed row per measured run.
- `experiments/summary.csv`: median, minimum, maximum, IQR, YARN resources, and graph outcome for each combination.
- `experiments/comparisons.csv`: median wall-time speedups between engines and formats.

The summary can be regenerated independently:

```bash
./benchmark/summarize_results.sh experiments/results.csv
```
