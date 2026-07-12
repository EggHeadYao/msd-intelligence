# C2 Graph Embedding -- Meta-Path Random Walk Framework

Build adjacency index and generate meta-path guided random walks on the
MERLIN heterogeneous graph for downstream Word2Vec training.

## Input

- `parquets/prepared/graph_edges.parquet` -- 7 edge types, 134.7M rows
- `parquets/prepared/songs_metadata.parquet` -- 1M songs (track_id list)

## Output

- `walk_sequences.parquet` -- 3M rows, schema: (track_id, walk_id, walk_seq: array<string>, walk_len)

## Usage

```bash
# Dev mode (10K songs)
spark-submit --driver-memory 2g merlin/embedding/graph/main.py \
  --input parquets/prepared --output parquets/walks --sample 10000

# Full 1M
spark-submit --driver-memory 2g --executor-memory 4g \
  merlin/embedding/graph/main.py \
  --input parquets/prepared --output parquets/walks --walks 3 --length 40
```
