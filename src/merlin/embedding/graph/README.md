# MERLIN C2 Graph Retrieval

MERLIN C2 is a graph-based similar-track retriever over the full Million Song Dataset catalog. It converts the canonical relation graph into deterministic typed random walks, exports one 128-dimensional Word2Vec root vector per track for coverage auditing, and serves context-bearing vectors through a normalized inner-product FAISS index.

The implementation also includes a masked-artist retrieval experiment. That experiment removes the direct artist relation from 10,000 query tracks, rebuilds the complete C2 representation, and measures whether release and other graph structure can reconstruct the hidden same-artist neighborhood.

## Pipeline

```text
prepared graph
  -> typed vocabulary and paired adjacency
  -> mixed eligibility-aware track walks
  -> Spark Word2Vec track-token vectors
  -> float32 L2-normalized embeddings
  -> exact FAISS IndexFlatIP and row mapping
```

The main stages are implemented in:

| Stage               | Entry point                           | Responsibility                                                                  |
| ------------------- | ------------------------------------- | ------------------------------------------------------------------------------- |
| Graph index         | `index.py`                            | Validate graph input, assign typed IDs, and build paired adjacency tables       |
| Walk generation     | `main.py`, `walk.py`                  | Generate deterministic mixed meta-path walks                                    |
| Encoder             | `train_word2vec.py`                   | Train and validate one vector per root track token                              |
| Search index        | `build_faiss.py`                      | Build an exact inner-product index and stable row mapping                       |
| Artifact validation | `validate_artifacts.py`               | Independently verify embeddings, mappings, metadata, hashes, and self-retrieval |
| Masked input        | `prepare_masked_artist_retrieval.py`  | Select queries, construct labels, and remove their artist edges                 |
| Masked evaluation   | `evaluate_masked_artist_retrieval.py` | Compare C2 with release-only and random baselines                               |
| Retrieval metrics   | `retrieval_metrics.py`                | Provide deterministic Recall, Hit, nDCG, MRR, and random expectations           |

## Input contract

`index.py` reads `graph_edges.parquet` from a prepared namespace. Its exact column order is:

```text
src_type, src_id, dst_type, dst_id, directed, edge_type
```

The graph contains exactly four canonical relations:

| Relation            | Endpoint types   | Traversal                       |
| ------------------- | ---------------- | ------------------------------- |
| `track_artist`      | track to artist  | Both directions                 |
| `track_release`     | track to release | Both directions                 |
| `artist_term`       | artist to term   | Both directions                 |
| `artist_similarity` | artist to artist | Original forward direction only |

The loader rejects unknown relations, old schemas, extra columns such as `weight`, empty IDs, invalid endpoint types, and incorrect direction flags. The prepared graph is unweighted; the only non-uniform sampling weight is the P3 term weight derived during adjacency construction.

## Typed vocabulary and adjacency

Vocabulary identity is the pair `(node_type, raw_id)`, serialized as an unambiguous compact JSON value. IDs are assigned after sorting by node type and raw ID, so identical raw strings in different node types cannot collide and repeated builds over the same graph are deterministic.

`vocab.json` contains `vocab_version`, `node_to_int`, `int_to_node`, and `int_to_type`. Walk sequences store compact integer track tokens, so this vocabulary is required to decode and audit them.

The index writes seven adjacency datasets:

```text
track_to_artist.parquet
artist_to_tracks.parquet
track_to_release.parquet
release_to_tracks.parquet
artist_to_terms.parquet
term_to_artists.parquet
artist_to_similar_artists.parquet
```

Each row stores `node_id`, `neighbor_ids`, and `weights`. Neighbor IDs and float32 weights are produced from the same sorted struct array and encoded together, preventing positional drift. Workers load them into compact CSR-style arrays and cache the validated representation within each Python process.

For P3, only terms connected to at least two active artists are eligible. The source-artist term weight is:

```text
idf(t) = log((N_artist + 1) / (df_artist(t) + 1)) + 1
weight(t) = min(idf(t), q99_eligible_idf)
```

Term-to-artist and endpoint-track selection remain uniform. Singleton terms stay in the typed vocabulary but cannot be selected for P3 because they cannot reach a different artist.

## Mixed track walks

Every walk starts from a track and stores only track tokens. Before each transition, the generator recomputes which paths can reach a different endpoint track and samples uniformly among the eligible paths.

| Path | Transition                                 | Sampling policy                                                        |
| ---- | ------------------------------------------ | ---------------------------------------------------------------------- |
| P1   | Track to Artist to Track                   | Uniform different endpoint track                                       |
| P2   | Track to Artist to Similar Artist to Track | Uniform eligible target artist, then uniform endpoint track            |
| P3   | Track to Artist to Term to Artist to Track | Capped-IDF term, uniform different artist, then uniform endpoint track |
| P4   | Track to Release to Track                  | Uniform different endpoint track                                       |

The path is selected again after every transition rather than fixed for the entire walk. A walk ends at 40 track tokens or when no path can move to a different track. The random generator is derived from the global seed, typed root-track ID, and walk ID, making output independent of Spark partition order.

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

Both path arrays use the fixed order `[P1, P2, P3, P4]`. `termination_reason` is `target_length` or `no_eligible_path`, and self-transitions never extend a walk. The canonical configuration is 10 walks per track, target length 40, and seed 42.

## Graph encoder and FAISS

`train_word2vec.py` trains Spark Word2Vec directly on the track-token sequences with the following frozen configuration:

| Parameter               | Value |
| ----------------------- | ----: |
| Vector size             |   128 |
| Window size             |     5 |
| Minimum count           |     1 |
| Iterations              |     5 |
| Step size               | 0.025 |
| Word2Vec partitions     |    20 |
| Maximum sentence length |    40 |
| Seed                    |    42 |

Each embedding is the learned vector of the root track token; the implementation does not average all tokens in a walk. Vectors are converted to float32 and L2-normalized. Training fails on incomplete coverage, duplicate IDs, missing root vectors, non-finite values, zero norms, incorrect dimensions, or normalization errors.

`build_faiss.py` sorts embeddings by `node_id`, validates them in bounded batches, and builds an exact `IndexFlatIP`. Because all vectors are unit-normalized, inner product is equivalent to cosine similarity. The row mapping is written from the same sorted table used to populate FAISS, so row identity never depends on Parquet file order.

### Serving coverage gate

An exported root vector is not automatically a valid graph-retrieval signal. A track that appears only in length-one sentences has no Word2Vec context pair; its vector may remain in the encoder export for coverage auditing, but it must not enter the candidate-serving FAISS mapping. For Ranker features, `has_graph` is true only when both tracks have context-bearing vectors present in that serving mapping.

The currently persisted full-catalog FAISS bundle contains all 1,000,000 exported root vectors. Before the multi-view Candidate stage consumes it, a context-count audit must produce the valid serving ID set and rebuild or filter the FAISS index and mapping. The final audit must report encoder export coverage and valid serving coverage separately.

The durable graph bundle contains:

```text
walk_sequences.parquet/
vocab.json
song_embeddings_graph.parquet/
word2vec_model/
graph_encoder_metadata.json
index_graph.faiss
index_graph_track_ids.parquet/
index_graph_manifest.json
```

`index_graph_manifest.json` uses the shared C1/C2 FAISS manifest contract `merlin_faiss_index_v1`. It records the `graph` embedding space, frozen `c2_graph_version`, exact index and mapping paths, and SHA-256 lineage for the index, row mapping, and graph encoder metadata. C3 rejects missing, historical, or mismatched manifests before loading the index.

## Masked-artist retrieval experiment

### Question

The experiment asks whether C2 can reconstruct a track's same-artist neighborhood after removing that track's direct artist relation. It evaluates graph representation quality rather than user preference or personalized recommendation.

### Protocol

1. Select one eligible query track from each of 10,000 distinct artists with seed 42.
2. Stratify sampling by artist track count: `2`, `3-5`, `6-20`, and `21+`.
3. Define positives as other tracks by the same artist, excluding the query and tracks with the same `song_id`.
4. Remove each query's `track_artist` edge and rebuild both traversal directions from the masked graph.
5. Rebuild full-catalog adjacency, 10 million walks, Word2Vec embeddings, and FAISS. Reusing embeddings trained on the complete graph is forbidden.
6. A masked query can initially move only through P4, `Track-Release-Track`. Queries with no different track in their release are counted as failures in all-query metrics.
7. Search the exact 1-million-track FAISS catalog, remove the query and same-song candidates, and report Recall, Hit, nDCG at 10, 20, and 50 plus MRR within the top 50.
8. Compare against a deterministic same-release ranking and the exact expectation of uniform random ranking without replacement.

This is a transductive cross-relation metadata reconstruction experiment. It is not strict relation-blind or inductive inference: a query may reach another track through release and then recover artist context from that track's unmasked relations.

### Query and graph audit

| Check                                            |    Result |
| ------------------------------------------------ | --------: |
| Catalog tracks                                   | 1,000,000 |
| Eligible query tracks and distinct query artists |    10,000 |
| Positive query-track pairs                       |   150,033 |
| Connectable queries                              |     9,171 |
| Structurally unconnectable queries               |       829 |
| Connectable coverage                             |    91.71% |
| `track_artist` edges before masking              | 1,000,000 |
| `track_artist` edges after masking               |   990,000 |
| Masked query artist-edge leaks                   |         0 |
| Other edge rows changed                          |         0 |

The smallest artist stratum contained only 2,439 eligible artists. The balanced quota allocator therefore selected `2,439 / 2,521 / 2,520 / 2,520` queries across the four ordered strata instead of silently reducing the experiment size.

### Full walk audit

| Check                           |                    Result |
| ------------------------------- | ------------------------: |
| Walk rows                       |                10,000,000 |
| Root tracks                     |                 1,000,000 |
| Walks per root                  |       Exactly 10, IDs 0-9 |
| Length range                    |                      1-40 |
| Mean length                     |                   39.9536 |
| Reached target length           |                 9,985,837 |
| `no_eligible_path` terminations |                    14,163 |
| P1 selections / eligible        |  90,774,404 / 358,505,779 |
| P2 selections / eligible        | 100,905,901 / 386,294,577 |
| P3 selections / eligible        | 100,562,070 / 385,458,241 |
| P4 selections / eligible        |  97,293,640 / 372,279,742 |

All 8,290 walks rooted at the 829 unconnectable queries had length 1, zero transitions, and `no_eligible_path`. Among the 91,710 walks rooted at connectable queries, every first transition reached a different track in the same release, proving that the masked query entered through P4. Forty-eight of those walks later reached a valid graph dead end; this is expected dynamic behavior and represents only 0.052% of connectable query walks.

Of the connectable first transitions, 61,824 reached a different same-artist track and none reached the same `song_id`. This confirms that release is a strong artist proxy and motivates the explicit release-only baseline.

### Encoder and index audit

| Check                                             |                                                             Result |
| ------------------------------------------------- | -----------------------------------------------------------------: |
| Embedding rows / distinct tracks / distinct nodes |                                  1,000,000 / 1,000,000 / 1,000,000 |
| Dimension and dtype                               |                                                       128, float32 |
| Normalized norm range                             |                                              0.99999998-1.00000002 |
| Raw norm range                                    |                                                   0.02204-14.87440 |
| FAISS type and metric                             |                                       `IndexFlatIP`, inner product |
| FAISS rows                                        |                                                          1,000,000 |
| FAISS SHA-256                                     | `9c61a4a7a0d862b5111fced012485ce62d28e47c2f4b9e3e89615fb0a9a106fa` |
| Independent self-retrieval audit                  |                                             100/100 queries passed |
| Word2Vec elapsed time                             |                                                    6,758.9 seconds |

### Retrieval results

The primary cutoff is 20. All-query metrics include the 829 structurally unconnectable queries as zero-valued failures; connectable-conditional metrics describe only the 9,171 queries that can enter the masked graph.

| Scope       | Method             | Recall@20 |   Hit@20 |  nDCG@20 |   MRR@50 | Mean returned candidates |
| ----------- | ------------------ | --------: | -------: | -------: | -------: | -----------------------: |
| All queries | C2                 |    0.5861 |   0.6997 |   0.6431 |   0.6318 |                   45.855 |
| All queries | Release-only       |    0.4305 |   0.6913 |   0.4984 |   0.6416 |                    6.742 |
| All queries | Random expectation |  0.000020 | 0.000300 | 0.000019 | 0.000067 |              999,999.000 |
| Connectable | C2                 |    0.6390 |   0.7629 |   0.7012 |   0.6889 |                   50.000 |
| Connectable | Release-only       |    0.4695 |   0.7538 |   0.5434 |   0.6995 |                    7.352 |
| Connectable | Random expectation |  0.000020 | 0.000312 | 0.000020 | 0.000070 |              999,999.000 |

The cutoff curves show how each method behaves as retrieval depth grows:

| Scope and method          | Recall@10 | Recall@20 | Recall@50 | Hit@10 | Hit@20 | Hit@50 | nDCG@10 | nDCG@20 | nDCG@50 |
| ------------------------- | --------: | --------: | --------: | -----: | -----: | -----: | ------: | ------: | ------: |
| All, C2                   |    0.4708 |    0.5861 |    0.6848 | 0.6802 | 0.6997 | 0.7182 |  0.6370 |  0.6431 |  0.6461 |
| All, release-only         |    0.4009 |    0.4305 |    0.4342 | 0.6857 | 0.6913 | 0.6927 |  0.5599 |  0.4984 |  0.4603 |
| Connectable, C2           |    0.5134 |    0.6390 |    0.7467 | 0.7417 | 0.7629 | 0.7831 |  0.6946 |  0.7012 |  0.7045 |
| Connectable, release-only |    0.4372 |    0.4695 |    0.4735 | 0.7477 | 0.7538 | 0.7553 |  0.6106 |  0.5434 |  0.5019 |

Artist-size slices at the primary cutoff reveal where the additional graph context helps:

| Artist tracks | Queries | C2 Recall@20 | Release Recall@20 | C2 Hit@20 | Release Hit@20 | C2 nDCG@20 | Release nDCG@20 |
| ------------- | ------: | -----------: | ----------------: | --------: | -------------: | ---------: | --------------: |
| 2             |   2,439 |       0.3920 |            0.3620 |    0.3920 |         0.3620 |     0.2898 |          0.3006 |
| 3-5           |   2,521 |       0.5875 |            0.4520 |    0.6041 |         0.6002 |     0.5247 |          0.4392 |
| 6-20          |   2,520 |       0.8652 |            0.6648 |    0.8813 |         0.8813 |     0.8598 |          0.7110 |
| 21+           |   2,520 |       0.4934 |            0.2411 |    0.9115 |         0.9111 |     0.8868 |          0.5364 |

The persisted report also contains release-degree and popularity slices in `evaluation/report.json`, while `evaluation/query_metrics.parquet` preserves every query-level metric for independent analysis.

### What the results show

1. C2 reconstructs substantial hidden artist structure. At K=20 it improves all-query Recall by 15.55 percentage points and nDCG by 14.47 points over release-only, while exceeding random ranking by several orders of magnitude.
2. Release is a strong shortcut for finding at least one relevant track. Release-only has slightly higher Hit@10 and MRR because its small candidate set places album mates near the top. This is why C2 should not be described as strict relation-blind artist inference.
3. C2 recovers a broader discography rather than merely one album mate. Release-only Recall saturates near 0.434 by K=50, whereas C2 rises to 0.685 overall and 0.747 for connectable queries.
4. The advantage grows with artist catalog size. For artists with 21 or more tracks, C2 nearly doubles Recall@20, from 0.241 to 0.493, while preserving essentially the same Hit rate.
5. Structural coverage remains a real limitation. The 8.29% of queries without a usable release connection cannot enter the masked graph; the complete MERLIN system must cover them through C1, BFS, or tag retrieval.
6. These results validate cross-relation metadata reconstruction, not user satisfaction, personalization, chronological prediction, or inductive performance on unseen catalog nodes.

## Reproducible commands

Run from the repository root and expose the Python packages under `src/`:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

Train, index, and validate an existing canonical C2 walk artifact:

```bash
spark-submit --master 'local[*]' --driver-memory 22g \
  src/merlin/embedding/graph/train_word2vec.py \
  --walks parquets_new/merlin/graph/walk_sequences.parquet \
  --output parquets_new/merlin/graph

python3 -m merlin.embedding.graph.build_faiss \
  --embeddings parquets_new/merlin/graph/song_embeddings_graph.parquet \
  --output parquets_new/merlin/graph

python3 -m merlin.embedding.graph.validate_artifacts \
  --output parquets_new/merlin/graph
```

Prepare a new masked-artist experiment in an empty output namespace:

```bash
spark-submit --master 'local[*]' --driver-memory 12g \
  src/merlin/embedding/graph/prepare_masked_artist_retrieval.py \
  --metadata parquets_new/prepared/songs_metadata.parquet \
  --graph parquets_new/prepared/graph_edges.parquet \
  --output parquets_new/merlin/masked_artist \
  --queries 10000 \
  --seed 42
```

Generate walks and train a separate full-catalog model from `masked_artist/prepared`. Never point this experiment at the canonical C2 model output or reuse embeddings trained on the unmasked graph. After building and validating its FAISS bundle, run:

```bash
python3 -m merlin.embedding.graph.evaluate_masked_artist_retrieval \
  --experiment parquets_new/merlin/masked_artist \
  --graph-output parquets_new/merlin/masked_artist/graph \
  --output parquets_new/merlin/masked_artist/evaluation \
  --cutoffs 10 20 50 \
  --overfetch 3 \
  --batch-size 64
```

The preparation, training, indexing, and evaluation entry points reject conflicting durable outputs unless an explicit overwrite option is supplied. Use a new namespace for experiments and keep canonical and masked artifacts separate.

## Test coverage

The implementation has four validation layers:

| Layer                          | Coverage                                                                                                                   | Result     |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------- | ---------- |
| Static checks                  | Ruff formatting/lint and Pyrefly checks for the experiment modules                                                         | Passed     |
| Pure and Spark unit tests      | Quota redistribution, metrics, filtering, positives, edge masking, and isolated-token Word2Vec coverage                    | 6/6 passed |
| End-to-end smoke test          | 1,000 tracks, 10,000 walks, 128D Word2Vec, FAISS, and 20-query artifact validation                                         | Passed     |
| Full artifact and metric audit | 10M walks, 1M vectors, exact mapping, 100 self-retrieval queries, 10k evaluation rows, and independent macro recomputation | Passed     |

The smoke walk set contained exactly 10 walks for each of 1,000 roots; 9,994 reached length 40 and six ended at valid dead ends. Its Word2Vec output contained 1,000 finite unit vectors, and all FAISS mapping and retrieval checks passed. The final evaluation report was independently recomputed from the 10,000 query rows; every macro metric, candidate mean, and candidate-shortage count matched the persisted report.
