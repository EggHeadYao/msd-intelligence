![Build badge](https://focs.gc.sjtu.edu.cn/git/ece472/p1team02/actions/workflows/push.yaml/badge.svg?branch=master) ![Build badge](https://focs.gc.sjtu.edu.cn/git/ece472/p1team02/actions/workflows/release.yaml/badge.svg?tag=p1m1)


# p1team02

## Layout

- `artistdistance/`: Maven module for the artist distance task.
- `artistdistance/src/main/java/artistdistance/convert`: SQLite to Avro/Parquet conversion.
- `artistdistance/src/main/java/artistdistance/mapreduce`: MapReduce BFS implementations.
- `artistdistance/src/main/java/artistdistance/spark`: Spark DataFrame BFS implementations.
- `artistdistance/src/main/java/artistdistance/validate`: Reference BFS and output verifiers.
- `artistdistance/benchmark`: YARN benchmark scripts.
- `artistdistance/experiments`: Benchmark result CSV and notes.
- `.gitea/`: issue, sprint, scrum, and PR templates.

## Requirements

- Java 17
- Maven
- Hadoop 3.5.0 with HDFS/YARN commands available
- Spark 4.1.2
- SQLite artist similarity database at `../msd/AdditionalFiles/artist_similarity.db`

## Build

Run from `p1team02/artistdistance`:

```bash
mvn -q package dependency:copy-dependencies -DincludeScope=runtime
```

## Convert Input Data

```bash
mvn exec:java -Dexec.mainClass=artistdistance.convert.ArtistGraphConverter -Dexec.args="../../msd/AdditionalFiles/artist_similarity.db ../data/artistdistance-output"
```

This writes Avro and Parquet graph inputs under `p1team02/data/artistdistance-output`.

## Benchmark

Run one combination:

```bash
./benchmark/run_one.sh mapreduce avro 1
```

Run all four combinations:

```bash
./benchmark/run_all.sh 3
```

Use `SOURCE_ID=<artist_id>` to change the BFS source artist. Benchmark timing is read from YARN application status, and outputs are verified against the reference BFS before a run is marked as correct.
