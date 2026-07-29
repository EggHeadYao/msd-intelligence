![Build badge](https://focs.gc.sjtu.edu.cn/git/ece472/p1team02/actions/workflows/push.yaml/badge.svg?branch=master) ![Build badge](https://focs.gc.sjtu.edu.cn/git/ece472/p1team02/actions/workflows/release.yaml/badge.svg?tag=p1m2)

# p1team02

This repository contains our Million Song Dataset pipelines for Apache Drill analytics, distributed artist-distance computation, MERLIN song recommendation, and year prediction. The implementations use Hadoop, Spark, Parquet, Avro, FAISS, Spark ML, and LightGBM over a shared million-track data foundation.

## Layout

- `tools/hdf5/`: HDF5 and SQLite extraction utilities for producing Hadoop-friendly Parquet data.
- `drill/`: the four required Apache Drill queries, runner, configuration, and verified CSV results.
- `artistdistance/`: Avro and Parquet graph conversion, MapReduce and Spark BFS, validation, and YARN benchmarks.
- `merlin/`: prepared recommendation tables, audio and graph embeddings, FAISS retrieval, ranking, inference, and evaluation.
- `year_prediction/`: artist-disjoint datasets, feature contracts, distributed model training, evaluation, and experiment results.
- `slides/` and `poster/`: presentation and investment-poster sources with reproducible Makefiles.
- `.gitea/`: issue, sprint, scrum, pull-request, and CI templates.

## Requirements

- Java 17, Maven, Hadoop 3.5.0, and Spark 4.1.2.
- Python 3 with the dependencies required by the selected MERLIN or Year Prediction stage.
- Apache Drill 1.22.0 for the analytical queries.
- The Million Song Dataset and AdditionalFiles under `../msd` for raw-data workflows.

## Artist Distance

Build from `p1team02/artistdistance`:

```bash
mvn -q package dependency:copy-dependencies -DincludeScope=runtime
```

Convert the directed artist-similarity graph to Avro and Parquet:

```bash
mvn exec:java -Dexec.mainClass=artistdistance.convert.ArtistGraphConverter -Dexec.args="../../msd/AdditionalFiles/artist_similarity.db ../data/artistdistance-output"
```

Run all four MapReduce/Spark and Avro/Parquet combinations with five repetitions:

```bash
./benchmark/run_experiment.sh 5
```

The benchmark records wall-clock and YARN measurements, validates every output against the reference BFS, resumes interrupted runs, and writes aggregate CSV files under `artistdistance/experiments`.

## Drill

Run the four required queries from the repository root:

```bash
./drill/scripts/run_all.sh
```

See [`drill/README.md`](drill/README.md) for the data workspace, query semantics, and verified results.

## MERLIN

MERLIN combines 128-dimensional PCA audio embeddings, typed graph walks with Spark Word2Vec, FAISS candidate retrieval, artist-distance and tag evidence, and a learned ranking pipeline. Start with [`merlin/prepare/README.md`](merlin/prepare/README.md), then follow the audio, graph, and inference READMEs for stage-specific commands and artifact contracts.

## Year Prediction

The Year Prediction pipeline freezes an artist-disjoint dataset contract and compares constant baselines, distributed Ridge, RFF-Ridge, LightGBM, Ordinal-MoE, and prediction ensembles. Data, feature, training, evaluation, and experiment documentation is located under [`year_prediction/`](year_prediction/); the recorded ensemble reaches a test MAE of 4.82 years and RMSE of 7.24 years.

Run the Python test suite from the repository root:

```bash
python3 -m unittest discover -s year_prediction/tests -p 'test_*.py'
```

## Presentation

Build the slides and poster independently:

```bash
make -C slides
make -C poster
```
