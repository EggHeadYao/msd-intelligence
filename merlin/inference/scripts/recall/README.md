# Recall commands

These commands produce and inspect ranker-independent Stage-1 artifacts.

## Entry points

- `build_recall_artifacts.py` publishes the frozen candidate policy and Tag-IDF
  artifacts.
- `validate_recall.py` runs deterministic four-source recall for a query list
  and writes structural coverage and repeatability diagnostics.
- `export_candidates.py` exports a candidate pool for Set A, Set B, or Set C.
- `audit_candidates.py` validates a persisted candidate pool and summarizes its
  source coverage.

Run them as modules, for example:

```bash
python -m merlin.inference.scripts.recall.build_recall_artifacts
python -m merlin.inference.scripts.recall.export_candidates \
  --split-assignments parquets_new/merlin/ranker/split_assignments.parquet \
  --split-manifest parquets_new/merlin/ranker/split_manifest.json \
  --query-split set_a
```

Set-C export additionally requires the frozen evaluation protocol. Structural
recall reports must not describe source coverage as relevance or accuracy.
