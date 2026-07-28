# MERLIN C3 Ranking and Inference

`merlin.inference` is the stable boundary between C1/C2 artifacts, C3 offline
training and evaluation, and the production recommendation pipeline. Core
contracts and online components are pure Python; high-volume preparation and
training entry points may use FAISS or Spark.

## Architecture

```text
                          C1 Audio FAISS
                         /
query track -> four-source recall -- C2 Graph FAISS -> candidate union
                         \  BFS / Tag                    |
                                                         v
                                                canonical pair features
                                                         |
                                                         v
                                          frozen Logistic Ranker + C1 order
                                                         |
                                                         v
                                            guarded quota-interleaved top 20
```

Recall nominates candidates and preserves source evidence. Unused union capacity
is deterministically backfilled from additional Tag and BFS nominations without
removing any source's primary quota. Feature computation
then derives query/candidate signals independently of recall provenance. The
ranker applies frozen Set-A preprocessing and scores every candidate by its raw
LR margin. A Set-B-selected relation-evidence gate either keeps the pure C1
order or applies the frozen LR/C1 quota interleave in the top 20. Relation
evidence measures the density of valid BFS, Tag, and same-release signals over
the whole candidate list, rather than its single largest value. The production
path does not apply MMR.

## Package map

| Package | Responsibility |
| --- | --- |
| [`artifacts/`](artifacts/README.md) | Canonical paths, JSON/Parquet IO, hashes, and lineage |
| [`data/`](data/README.md) | Catalog identity, graph adjacency, and sparse artist Tags |
| [`retrieval/`](retrieval/README.md) | FAISS adapters and individual candidate retrievers |
| [`recall/`](recall/README.md) | Quotas, online/streaming recall, and candidate pools |
| [`ranking/`](ranking/README.md) | Pair features, LR artifacts, inference, and model selection |
| [`training/`](training/README.md) | Splits, weak labels, pair sampling, and validation groups |
| [`evaluation/`](evaluation/README.md) | Reproducible Set-C development protocol and ranking statistics |
| [`runtime/`](runtime/README.md) | Production assembly, validation, and recommendation APIs |
| [`scripts/`](scripts/README.md) | Supported recall, training, and evaluation commands |

Import public package paths such as `merlin.inference.recall`,
`merlin.inference.retrieval`, and `merlin.inference.ranking.features`. Older
flat modules, numbered feature modules, and compatibility scripts are not part
of the supported interface.

## Online request flow

1. Audio, Graph, BFS, and Tag retrievers nominate candidates under the frozen
   source quotas.
2. Duplicate track IDs are merged while source scores and ranks are retained
   for audit.
3. Additional Tag, then BFS nominations fill unused union capacity; the union
   remains capped at 1,000 unique candidates and excludes same-song items.
4. Canonical pair features are filled and scaled with frozen Set-A statistics.
5. Queries below the frozen relation-evidence gate use C1-only order; other
   queries deterministically interleave C1 and LR order in the top 20.

Load the fail-closed production pipeline with the frozen C2 contract:

```python
from merlin.inference import load_inference_pipeline

pipeline = load_inference_pipeline(
    graph_contract_key="c2_graph_version",
    graph_contract_version="<frozen-version>",
)
recommendations = pipeline.recommend(query_track_id)
```

A separate `ColdAudioPipeline` accepts a C1-compatible 128D query embedding and
ranks Audio neighbors directly. It never invokes Graph, BFS, Tag, or LR.

## Offline supervised flow

```text
split assignments
  -> weak-label thresholds and positives
  -> Set-A candidate pool and tuning pairs/features
  -> Set-B candidate pool and validation groups/features
  -> Set-B tune selection and independent confirmation
  -> streamed A+B+C+Remaining retrain pairs/features
  -> Full and no-hard-negative models
  -> reproducible Set-C development protocol
  -> Set-C candidates, groups, features, and development report
```

Set-A tuning artifacts live below `ranker/tuning/`. Root
`training_pairs.parquet` and `raw_pair_features.parquet` are the canonical
retrain datasets. Retraining streams bounded query batches and checkpoints
published parts instead of persisting an all-catalog candidate pool.

The exact command order and stage-specific requirements are documented in
[`scripts/ranker/README.md`](scripts/ranker/README.md). Recall-only commands are
documented in [`scripts/recall/README.md`](scripts/recall/README.md).

## Frozen contracts

- Production C1/C2 indexes are normalized 128D `IndexFlatIP` artifacts with a
  contiguous row-to-track Parquet mapping and matching manifest.
- Training and inference share the same feature order, fill values,
  interactions, means, standard deviations, and zero-variance handling.
- Recall-source flags and popularity are audit fields, not ranker features.
- Splits are song-safe. Set C is reusable known development data and joins Set
  A, Set B, and Remaining during final retraining. It is not an unbiased test.
- Set B is song-safe and split into a 1% tune fold and an independent 2%
  confirmation fold. Both use frozen Audio-dominant, Relation-dominant, and
  Mixed query groups built from cleaned pre-PCA C1 signals.
- Positive loss mass is balanced per query between Audio-derived and
  relation-derived labels. Set-B tune jointly selects LR regularization, the C1
  quota, and a relation-evidence gate. Confirmation failure publishes a strict
  C1-order fallback and marks fusion unpublishable. The fallback may still be
  measured on Set C for development diagnosis.
- Every formal consumer validates artifact version, hashes, and parent lineage
  before loading data.
- A schema, hash, lineage, or policy mismatch fails closed.

The canonical feature list and persistence rules are in
[`ranking/features/README.md`](ranking/features/README.md). Set-C development
protocol and metric rules are in [`evaluation/README.md`](evaluation/README.md).

## Operational notes

- Formal high-volume row artifacts use partitioned Parquet with Zstandard
  compression; legacy JSONL-Gzip remains readable for smoke compatibility.
- High-volume commands reserve free disk space and place temporary data under
  the output filesystem. Use `--scratch-root` for a dedicated volume.
- Never delete an active Spark block-manager directory.
- On hosts whose FAISS wheel requires unsupported SIMD instructions, use
  `MERLIN_FAISS_SEARCH_ENGINE=numpy` with the command's low-memory mode.
- Smoke outputs must use explicit paths outside the canonical formal artifact
  directory.
