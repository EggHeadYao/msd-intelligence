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
                                                frozen Logistic Ranker
                                                         |
                                                         v
                                                deterministic top 20
```

Recall nominates candidates and preserves source evidence. Feature computation
then derives query/candidate signals independently of recall provenance. The
ranker applies frozen Set-A preprocessing and scores every candidate by its raw
LR margin. The production path does not apply MMR.

## Package map

| Package | Responsibility |
| --- | --- |
| [`artifacts/`](artifacts/README.md) | Canonical paths, JSON/Parquet IO, hashes, and lineage |
| [`data/`](data/README.md) | Catalog identity, graph adjacency, and sparse artist Tags |
| [`retrieval/`](retrieval/README.md) | FAISS adapters and individual candidate retrievers |
| [`recall/`](recall/README.md) | Quotas, online/streaming recall, and candidate pools |
| [`ranking/`](ranking/README.md) | Pair features, LR artifacts, inference, and model selection |
| [`training/`](training/README.md) | Splits, weak labels, pair sampling, and validation groups |
| [`evaluation/`](evaluation/README.md) | Frozen Set-C protocol and ranking statistics |
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
3. The union is capped at 1,000 unique candidates and same-song items are
   excluded.
4. Canonical pair features are filled and scaled with frozen Set-A statistics.
5. Candidates are sorted by `(-raw_margin, track_id)` and the top 20 are
   returned.

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
  -> regularization selection and tuning model
  -> streamed A+B+Remaining retrain pairs/features
  -> Full and no-hard-negative models
  -> frozen Set-C protocol
  -> Set-C candidates, groups, features, and evaluation report
```

Set-A tuning artifacts live below `ranker/tuning/`. Root
`training_pairs.parquet` and `raw_pair_features.parquet` are the canonical
retrain datasets. Retraining streams bounded query batches and checkpoints
published parts instead of persisting a 980K-query candidate pool.

The exact command order and stage-specific requirements are documented in
[`scripts/ranker/README.md`](scripts/ranker/README.md). Recall-only commands are
documented in [`scripts/recall/README.md`](scripts/recall/README.md).

## Ranker features

The fixed feature order is exported as `FEATURE_ORDER` from
`merlin.inference.ranking.features`:

```text
cos_audio, cos_graph, has_graph, bfs_score, has_bfs,
tag_tfidf_cosine, has_tags, same_release, has_release,
year_gap, has_year, audio_tag_interaction, graph_bfs_interaction
```

Pair signals are computed independently of recall provenance. Missing continuous
signals use Set-A fill statistics and retain availability masks. Interactions are
computed after filling and before scaling. Popularity and recall-source flags are
audit fields, not ranker features.

## Ranker handoff

Person A must hand off the frozen Set-A preprocessing and trained model together:

```text
ranker_feature_schema.json  # contract and ordered feature names
ranker_scaler.json          # fill values, means, and standard deviations
ranker_coefficients.json    # coefficients and intercept
```

Training and inference must use the same fill, interaction, and scaling order.
The pipeline rejects a ranker whose schema version differs from its feature
computer. Fixed Spark/Python pairs must match on features and raw margin within
`1e-6` before the artifact is accepted.

## FAISS artifacts

Install `faiss-cpu`, `numpy`, and `pyarrow` in the inference environment. Production
C1/C2 must publish a normalized 128D `IndexFlatIP`, matching Parquet map, and
lineage manifest:

```text
index_<space>.faiss
index_<space>_track_ids.parquet  # row_id: long, track_id: string
index_<space>_manifest.json
```

Load either embedding space with the same adapter:

```python
from merlin.inference.retrieval.faiss import load_audio_index
from merlin.inference.retrieval import VectorRetriever

audio = load_audio_index()  # parquets_new/merlin/audio, shared_audio_628_v1
audio_retriever = VectorRetriever("audio", audio.search)
```

The production loader requires the index, mapping, and manifest together. It
verifies the 128D `IndexFlatIP`, row count, contract version, index hash, and
mapping hash before exposing search. C2 must provide its frozen graph contract
before the equivalent production graph loader is enabled.

## Production assembly

The full pipeline has one fail-closed entry point. C2 must supply its frozen
manifest key and version explicitly:

```python
from merlin.inference import load_inference_pipeline

pipeline = load_inference_pipeline(
    graph_contract_key="c2_graph_version",
    graph_contract_version="<frozen-version>",
)
recommendations = pipeline.recommend(query_track_id)
```

Assembly validates both FAISS lineages, the canonical candidate-policy
manifest, Ranker schema/scaler/coefficients hashes and parent hashes, then loads
same-song identity from prepared metadata. BFS and Tag consume only the typed
`artist_similarity` and `artist_term` partitions of canonical
`parquets_new/prepared/graph_edges.parquet`.

The mapping must contain exactly one unique track per contiguous row ID from
zero. Its row order must match the order in which vectors were added to FAISS.
The existing 71D Audio index and old-graph artifacts are historical v1 inputs;
they may be used for adapter tests but must not be used for Set-C evaluation.

## Set-C evaluation

Set C remains unopened during tuning, retraining, and ablation construction.
After both the Full and no-hard-negative models are complete, freeze their
lineage and the evaluation rules before producing any Set-C-derived artifact:

```bash
python -m merlin.inference.scripts.ranker.freeze_set_c_protocol \
  --scope formal
```

Then run the evaluation stages in order:

```bash
python -m merlin.inference.scripts.recall.export_candidates \
  --split-assignments parquets_new/merlin/ranker/split_assignments.parquet \
  --split-manifest parquets_new/merlin/ranker/split_manifest.json \
  --query-split set_c \
  --evaluation-protocol parquets_new/merlin/ranker/set_c_evaluation/protocol.json

spark-submit --master 'local[6]' --driver-memory 5g \
  --conf spark.hadoop.fs.defaultFS=file:/// \
  merlin/inference/scripts/ranker/build_validation_groups.py \
  --apply-split set_c \
  --evaluation-protocol parquets_new/merlin/ranker/set_c_evaluation/protocol.json \
  --scope formal \
  --shuffle-partitions 64 \
  --scratch-root parquets_new/merlin/ranker/.c3-scratch

python -m merlin.inference.scripts.ranker.export_ranker_features \
  --pair-kind validation \
  --stage final_evaluation \
  --evaluation-protocol parquets_new/merlin/ranker/set_c_evaluation/protocol.json \
  --scope formal

python -m merlin.inference.scripts.ranker.evaluate_set_c \
  --scope formal
```

Canonical Set-C outputs live under
`parquets_new/merlin/ranker/set_c_evaluation/`. The resulting
`evaluation_report.json` includes candidate-layer diagnostics, stratified
nDCG/Recall metrics, paired query and artist-cluster bootstrap comparisons,
ablation results, and robustness slices. The protocol binds every stage to the
frozen splits, indexes, policies, preprocessing, and both model manifests; a
lineage mismatch fails closed.

## Cold Audio-only queries

Cold queries use a separate path and must provide a C1-compatible 128D audio
embedding. They never invoke Graph, BFS, Tag, LR, or MMR:

```python
from merlin.inference import ColdAudioPipeline
from merlin.inference.retrieval.faiss import load_audio_index

cold = ColdAudioPipeline(load_audio_index(), track_to_song)
recommendations, audit = cold.recommend_with_audit(
    query_embedding,
    query_song_id="optional-song-id",
)
```

The adapter converts to float32, rejects non-finite and zero-norm vectors, L2
normalizes, overfetches 3001 Audio neighbors, filters the query/same-song items,
keeps at most 1000 candidates, and returns the top 20 by C1 cosine. Transforming
raw 563D features into this embedding remains a C1 artifact responsibility.
