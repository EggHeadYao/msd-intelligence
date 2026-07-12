# MERLIN C2 -- Hypergraph Meta-Path Random Walk

Build adjacency index and generate meta-path guided random walks on the
MERLIN heterogeneous graph for downstream Word2Vec training.

## Inputs

`main.py` expects the prepared Parquet directory produced by `merlin/prepare/`:

- `graph_edges.parquet` -- 7 edge types, 134.7M rows, partitioned by `edge_type`
- `songs_metadata.parquet` -- 1M songs (used for `track_id` list only)

## Outputs

- `walk_sequences.parquet` -- one row per walk sequence.

  **Schema**:

  | Column | Type | Description |
  |--------|------|-------------|
  | `track_id` | string | Starting song MSD track ID |
  | `walk_id` | int32 | Walk index (0 .. r-1) |
  | `path_name` | string | Meta-path used for this walk (P1-P6) |
  | `walk_seq` | array\<int32\> | Sequence of song **integer node IDs** (see note below) |
  | `walk_len` | int32 | Actual number of song nodes (<= target length) |

  **Note on `walk_seq`**: stored as integer node IDs from the C2 internal
  vocabulary for compactness.  Downstream Word2Vec training should convert
  back to string track IDs using `int_to_node` from the index metadata
  (`vocab.json` under the temp directory).  A single `F.array_transform`
  call with a broadcast mapping is sufficient.

## Architecture

```
graph_edges.parquet (7 types, 134.7M rows)
        |
        v
  build_node_vocabulary      --> 1,583,933 unique nodes
        |
        v
  save_adjacency_parquet     --> 12 intermediate Parquet files
  (forward: 5, reverse: 7)      stored under /tmp/c2_index/
        |
        v
  generate_walks_for_partition   mapPartitions: each executor loads adjacency,
  (r=3, L=40 song nodes)         generates walks in parallel
        |
        v
  walk_sequences.parquet     --> 3,000,000 rows, ~431 MB
```

### Edge Types

| Edge type | Rows | Weight | Directed | Walk usage |
|-----------|-----:|:------:|:--------:|------------|
| `song_artist` | 1,000,000 | 1.0 | false | P1 step 0, P4 step 0 |
| `song_album` | 1,000,000 | 0.8 | false | P5 step 0 |
| `song_tag` | 28,979,585 | 0.6 | false | P3 step 0 |
| `artist_tag` | 1,109,381 | 0.3 | false | P4 step 1/2 |
| `song_year` | 515,576 | 0.4 | false | P6 step 0 |
| `artist_similarity` | 2,201,916 | 1.0 | **true** | P1 step 1 |
| `song_similar_artist` | 99,879,745 | 0.5 | **true** | P2 step 0 |

### Meta-Path Definitions (P1-P6)

| # | Edge sequence | Hops | Semantic |
|---|--------------|:----:|----------|
| P1 | song_artist -> artist_similarity -> rev:song_artist | 3 | Artist-level Echo Nest similarity |
| P2 | song_similar_artist -> rev:song_artist | 2 | Per-song Echo Nest similar artists |
| P3 | song_tag -> rev:song_tag | 2 | Per-song shared tags |
| P4 | song_artist -> artist_tag -> rev:artist_tag -> rev:song_artist | 4 | Artist-level shared tags |
| P5 | song_album -> rev:song_album | 2 | Same album |
| P6 | song_year -> rev:song_year | 2 | Same year (year=0 excluded) |

`rev:X` denotes the reverse direction of edge type X (swap src<->dst).
Reverse adjacency is available for all undirected edges.
`artist_similarity` and `song_similar_artist` are directed -- their
reverse is NOT available.

### Walk Algorithm

1. For each song, generate `r` walks.
2. Each walk picks a meta-path template via weighted random selection.
3. The walk cycles through the template's edge-type sequence, collecting
   song-type nodes until reaching `L` song nodes.
4. Neighbor selection is uniform random (all edges of the same type have
   equal weight).
5. Walks terminate early when no valid neighbor exists for the required
   edge type (e.g., year=0 songs on P6, songs without tags on P3/P4).

## Commands

Build index and generate walks:

```bash
spark-submit --driver-memory 8g p1team02/merlin/embedding/graph/main.py \
  --input parquets/prepared --output parquets/walks --walks 3 --length 40
```

Dev mode (10K songs, ~2 min):

```bash
spark-submit --driver-memory 8g p1team02/merlin/embedding/graph/main.py \
  --input parquets/prepared --output parquets/walks --sample 10000
```

## Full-Run Verification

### Index Build

| Metric | Value |
|--------|------:|
| Total unique nodes | 1,583,933 |
| Songs | 1,000,000 |
| Artists | 44,745 |
| Albums | 221,753 |
| Tags | 7,643 |
| Years | 89 |
| Forward adjacency files | 5 |
| Reverse adjacency files | 7 |
| Build time | ~2 min |

### Walk Generation

| Metric | Value |
|--------|------:|
| Total walks | 3,000,000 = 1M x 3 |
| Song coverage | 1,000,000 / 1,000,000 (100%) |
| Walks per song | exactly 3 |
| Walk length (min / median / max) | 1 / 40 / 40 |
| Walk length (average) | 36.1 |
| Output size | 431 MB |

### Meta-Path Distribution (with updated weights)

| Path | Weight | Walks | Share | Avg walk_len |
|------|:------:|------:|:-----:|:------------:|
| P1 | 1.2 | 898,887 | 30.0% | 39.7 |
| P2 | 0.2 | 150,175 | 5.0% | 2.6 |
| P3 | 0.7 | 526,057 | 17.5% | 39.9 |
| P4 | 0.5 | 374,885 | 12.5% | 39.9 |
| P5 | 1.0 | 750,896 | 25.0% | 40.0 |
| P6 | 0.4 | 299,100 | 10.0% | 21.1 |

**P2 note**: Echo Nest `similar_artists` from HDF5 references many artists
outside the MSD database.  When the walk reaches such an artist at step 1
(`rev:song_artist`), no reverse edge exists and the walk terminates with
walk_len=1.  This is why P2 weight was reduced from 0.8 to 0.2 --
~20% of walks were wasted on single-node sequences.  The remaining 5%
allocation preserves the signal for songs whose similar artists ARE in MSD.
