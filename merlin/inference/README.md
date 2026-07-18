# MERLIN Pure-Python Inference

This package defines the stable boundary between C1/C2 artifacts, Person A's
ranker output, and the online recommendation pipeline. Importing it does not
require Spark or FAISS.

## Flow

1. `CandidateRetriever` implementations nominate Audio, Graph, BFS, and Tag candidates.
2. `merge_candidates` unions candidates and preserves source evidence.
3. `PairFeatureComputer` produces named `ranker-v2` pair features.
4. `LogisticRanker` standardizes features and computes the raw LR margin.
5. Candidates are sorted by raw margin to produce the final top 20.

`VectorRetriever` accepts an injected nearest-neighbor function so C1 and C2
can expose their FAISS index without coupling this package to index construction.
`BfsRetriever` consumes mappings derived from `artist_similarity_edges` and
`track_artist`; `TagRetriever` uses artist-level terms from `artist_term`.
Expanded song-level terms are historical v1 data and are not canonical input.

The main pipeline does not apply MMR. It ranks the canonical candidate union by
the LR raw margin and returns the top 20. MMR is deferred future work.

## Ranker-v2 features

The fixed feature order is exported as `RANKER_V2_FEATURES`:

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
ranker_feature_schema.json  # version and ordered feature names
ranker_scaler.json          # fill values, means, and standard deviations
ranker_coefficients.json    # coefficients and intercept
```

Training and inference must use the same fill, interaction, and scaling order.
The pipeline rejects a ranker whose schema version differs from its feature
computer. Fixed Spark/Python pairs must match on features and raw margin within
`1e-6` before the artifact is accepted.

The example artifact is only for integration testing. It is not a trained model.

## FAISS artifacts

Install `faiss-cpu`, `numpy`, and `pyarrow` in the inference environment. Final
C1/C2 v2 must publish a normalized 128D `IndexFlatIP`, matching Parquet map, and
lineage manifest:

```text
index_<space>.faiss
index_<space>_track_ids.parquet  # row_id: long, track_id: string
index_<space>_manifest.json
```

Load either embedding space with the same adapter:

```python
from merlin.inference.loaders import load_audio_index
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
they may be used for adapter tests but must not be used for final evaluation.

## Cold Audio-only queries

Cold queries use a separate path and must provide a C1-compatible 128D audio
embedding. They never invoke Graph, BFS, Tag, LR, or MMR:

```python
from merlin.inference import ColdAudioPipeline
from merlin.inference.loaders import load_audio_index

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
