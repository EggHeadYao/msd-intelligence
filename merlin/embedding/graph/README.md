# MERLIN C2 Graph Index

The C2 index projects the canonical prepared graph into a deterministic typed vocabulary and compact adjacency tables. Walk generation consumes this index as its only graph-access layer.

## Input

`index.py` reads `parquets_new/prepared/graph_edges.parquet`, whose exact schema is:

```text
src_type, src_id, dst_type, dst_id, directed, edge_type
```

The graph must contain exactly four relations:

| Relation            | Endpoint types   | Direction                       |
| ------------------- | ---------------- | ------------------------------- |
| `track_artist`      | track to artist  | bidirectional traversal         |
| `track_release`     | track to release | bidirectional traversal         |
| `artist_term`       | artist to term   | bidirectional traversal         |
| `artist_similarity` | artist to artist | original forward direction only |

The loader rejects old edge names, extra columns such as `weight`, empty IDs, incorrect endpoint types, and incorrect direction flags. It does not require a separate manifest or lineage framework.

## Typed vocabulary

Vocabulary identity is `(node_type, raw_id)`, encoded as an unambiguous compact JSON pair. IDs are assigned after sorting by node type and raw ID, so equal raw strings in different node types cannot collide and repeated builds over the same graph are deterministic.

`vocab.json` contains `vocab_version`, `node_to_int`, `int_to_node`, and `int_to_type`. It is required downstream because walk tokens are stored as compact integer IDs.

## Adjacency tables

The index writes seven Parquet datasets:

```text
track_to_artist.parquet
artist_to_tracks.parquet
track_to_release.parquet
release_to_tracks.parquet
artist_to_terms.parquet
term_to_artists.parquet
artist_to_similar_artists.parquet
```

Each row contains `node_id`, `neighbor_ids`, and `weights`. Neighbor IDs and float32 weights are produced from one sorted struct array and then encoded together, so their positions cannot drift apart. Uniform relations store weight `1.0`.

Workers decode these rows into a compact CSR-style representation with a dense row lookup, offsets, and contiguous neighbor/weight arrays. The validated representation and deterministic artist/track eligibility structures are cached within each Python worker, avoiding repeated Parquet reads and repeated expansion of high-degree term neighborhoods.

For P3, only terms connected to at least two active artists are usable. The source-artist term weight is:

```text
idf(t) = log((N_artist + 1) / (df_artist(t) + 1)) + 1
weight(t) = min(idf(t), q99_eligible_idf)
```

Term-to-artist selection remains uniform. Singleton terms stay in the typed graph vocabulary but are omitted from P3 adjacency because they cannot reach another artist.

## Mixed track walks

Each walk starts from a track and stores only track tokens. At every transition, the generator recomputes which of the following paths can reach a different endpoint track, then selects uniformly among the eligible paths:

| Path | Transition                                 | Sampling policy                                                               |
| ---- | ------------------------------------------ | ----------------------------------------------------------------------------- |
| P1   | Track to Artist to Track                   | Uniform endpoint track, excluding the current track                           |
| P2   | Track to Artist to Similar Artist to Track | Uniform eligible target artist, then uniform endpoint track                   |
| P3   | Track to Artist to Term to Artist to Track | Capped-IDF term, uniform different target artist, then uniform endpoint track |
| P4   | Track to Release to Track                  | Uniform endpoint track, excluding the current track                           |

The selected path is recomputed after every transition rather than fixed for an entire walk. A walk ends at 40 track tokens or when no path can move to a different track. Every walk uses a generator derived from the global seed, typed root-track ID, and walk ID, so equal inputs produce equal walks independent of Spark partition order.

`walk_sequences.parquet` contains:

```text
track_id
walk_id
walk_seq
walk_len
transition_count
path_counts
path_eligible_counts
termination_reason
```

Both path arrays use the fixed order `[P1, P2, P3, P4]`. `termination_reason` is either `target_length` or `no_eligible_path`. Self-transitions are never used to extend a walk.

The default configuration generates 10 walks per track with a target length of 40 and seed 42.

## Graph embeddings

`train_word2vec.py` trains Spark Word2Vec directly on the integer track-token sequences. The canonical configuration is 128 dimensions, window size 5, minimum count 1, five iterations, step size 0.025, 20 Word2Vec partitions, maximum sentence length 40, and seed 42.

Each output embedding is the learned vector of the corresponding root track token. Walk-neighborhood vectors are not averaged after training. Root `track_id` values are mapped to their first `walk_seq` token, the resulting vectors are converted to float32, and every vector is L2 normalized. Training fails if coverage is incomplete or if an embedding is duplicated, non-finite, zero, incorrectly sized, or not normalized.

The outputs are:

```text
song_embeddings_graph.parquet/
word2vec_model/
graph_encoder_metadata.json
```

The embedding table contains `node_id`, `track_id`, and `embedding`.

## Graph FAISS

`build_faiss.py` sorts embeddings by `node_id`, validates their numeric contract in bounded batches, and builds an exact `IndexFlatIP`. The mapping is written from the same sorted table used to populate FAISS, so row identity does not depend on Parquet file or Spark partition order.

The outputs are:

```text
index_graph.faiss
index_graph_track_ids.parquet/
graph_faiss_metadata.json
```

The mapping contains `row_id`, `node_id`, and `track_id`. `validate_artifacts.py` independently checks embedding and mapping coverage, uniqueness, dimensions, normalization, the FAISS metric and size, the index hash, and deterministic self-retrieval samples.

## Commands

From the repository root, train on an existing canonical walk artifact:

```bash
PYTHONPATH=. spark-submit --master 'local[*]' --driver-memory 22g \
  merlin/embedding/graph/train_word2vec.py \
  --walks ../parquets_new/merlin/graph/walk_sequences.parquet \
  --output ../parquets_new/merlin/graph
```

Then build and validate the exact graph index:

```bash
python3 -m merlin.embedding.graph.build_faiss \
  --embeddings ../parquets_new/merlin/graph/song_embeddings_graph.parquet \
  --output ../parquets_new/merlin/graph

python3 -m merlin.embedding.graph.validate_artifacts \
  --output ../parquets_new/merlin/graph
```

## Tests

The tests cover typed raw-ID collisions, strict input rejection, deterministic vocabulary assignment, all seven adjacency directions, paired binary decoding, hierarchical path sampling, eligibility recomputation, self exclusion, early termination, deterministic mixed walks, direct Word2Vec track-vector export, normalization, exact inner-product indexing, mapping identity, and persisted artifact validation.
