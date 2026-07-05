# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [P1M1] - 2026-07-05

### Added

- Initial repository setup with Gitea issue templates, PR template, workflows,
  and local git hooks.
- Artist Distance Maven module using Java 17, Avro, Parquet, Hadoop 3.5.0,
  Spark 4.1.2, and SQLite JDBC.
- Avro schemas and generated specific records for graph edges, adjacency rows,
  BFS vertices, and final artist distances.
- Converter from `artist_similarity.db` to Avro and Parquet graph inputs while
  preserving the directed graph.
- Shared BFS path/rule helpers.
- Avro and Parquet MapReduce BFS implementations.
- Avro and Parquet Spark DataFrame BFS implementations.
- Reference BFS and Avro/Parquet output verifiers.
- YARN benchmark scripts for the four required combinations with source id,
  elapsed time, and verification result recorded in CSV.
- README files for schemas, converter, BFS helpers, MapReduce, Spark, validation,
  benchmark, and experiment outputs.

### Changed

- Refactored MapReduce BFS around shared runner, format, counter, and iteration
  logic so Avro and Parquet implementations share the same control flow.
- Updated benchmark execution to run on YARN with HDFS inputs/outputs instead of
  local mode.
- Made each benchmark task restart HDFS/YARN and wait for HDFS to leave safemode
  before submission.
- Deferred Drill query work to p1m2.

### Fixed

- Aligned Hadoop/Spark dependency versions with the local environment.
- Removed incompatible SLF4J bindings from Hadoop dependencies.
- Added Spark Avro runtime jar handling for Spark benchmark runs.
- Verified MapReduce final output from the `final` subdirectory.
- Cleaned benchmark output, temporary local copies, uploaded inputs, and staging
  files after successful verification.
- Improved git hook handling for Java/Scala files and filename checks.
