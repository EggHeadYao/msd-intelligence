# Artifact contracts

This package centralizes where inference artifacts live, how they are stored, and how their provenance is verified. Other packages should use these helpers instead of defining local path, hashing, or Parquet conventions.

## Modules

- `paths.py` defines `InferenceArtifactPaths` and the canonical C1, C2, Candidate, and Ranker locations below `parquets_new/merlin`.
- `integrity.py` computes file and directory hashes and validates FAISS manifests and parent lineage.
- `io.py` reads and writes JSON, JSONL-Gzip, and Parquet row artifacts. Its `PartitionedParquetWriter` supports bounded buffers, partitioned output, and resumable high-volume jobs.

## Contracts

- Published JSON is written atomically.
- Formal row datasets use Parquet with Zstandard compression by default.
- A manifest identifies its schema/version, scope, parents, row counts, and content hashes where applicable.
- Consumers validate lineage before loading data; a mismatched parent fails closed rather than silently mixing runs.
- Smoke outputs must use explicit non-canonical paths.

See the [top-level architecture](../README.md) for the artifact flow.
