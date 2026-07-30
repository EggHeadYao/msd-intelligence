# Recall commands

These commands publish the frozen Stage-1 contract and export candidate pools for the ranker. Run them from the repository root.

## Environment

The current 9-vCPU VM should use four FAISS/OpenMP threads. Audio and Graph search run concurrently, so a larger value can oversubscribe the VM.

```bash
export MERLIN_ROOT="$PWD/src"
export MERLIN_PYTHON=/absolute/path/to/merlin-faiss/bin/python
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$MERLIN_ROOT"
export OPENBLAS_CORETYPE=generic
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
cd "$MERLIN_ROOT"
```

## Publish the recall contract

Run this after the candidate-policy or Tag-IDF contract changes. `--overwrite` replaces both artifacts as one versioned unit.

```bash
"$MERLIN_PYTHON" -m merlin.inference.scripts.recall.build_recall_artifacts \
  --graph-edges parquets_new/prepared/graph_edges.parquet \
  --candidate-policy parquets_new/merlin/ranker/candidate_policy_manifest.json \
  --tag-idf parquets_new/merlin/ranker/tag_idf.json \
  --overwrite
```

## Export split candidate pools

Set A supplies tuning pairs; Set B supplies validation groups. Output paths are selected automatically from `--query-split`.

```bash
"$MERLIN_PYTHON" -m merlin.inference.scripts.recall.export_candidates \
  --split-assignments parquets_new/merlin/ranker/split_assignments.parquet \
  --split-manifest parquets_new/merlin/ranker/split_manifest.json \
  --query-split set_a \
  --min-free-gb 8

"$MERLIN_PYTHON" -m merlin.inference.scripts.recall.export_candidates \
  --split-assignments parquets_new/merlin/ranker/split_assignments.parquet \
  --split-manifest parquets_new/merlin/ranker/split_manifest.json \
  --query-split set_b \
  --min-free-gb 8
```

Set C is reusable development data and requires the bound development protocol:

```bash
"$MERLIN_PYTHON" -m merlin.inference.scripts.recall.export_candidates \
  --split-assignments parquets_new/merlin/ranker/split_assignments.parquet \
  --split-manifest parquets_new/merlin/ranker/split_manifest.json \
  --query-split set_c \
  --development-protocol \
    parquets_new/merlin/ranker/development_evaluation/protocol.json \
  --min-free-gb 8
```

`validate_recall.py` runs deterministic four-source recall for a query list; `audit_candidates.py` validates a persisted pool and summarizes structural source coverage. Structural coverage is not relevance or ranking accuracy.
