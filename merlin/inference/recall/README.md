# Multi-source recall

This package turns Audio, Graph, BFS, and Tag retrievers into the canonical
Stage-1 candidate union.

## Modules

- `policy.py` owns source quotas, the 1,000-candidate cap, and policy
  validation.
- `pipeline.py` provides deterministic single-query and bounded-batch online
  recall with per-source audits.
- `streaming.py` provides integer-coded, bounded-memory batched recall for
  large training and evaluation jobs.
- `pool.py` persists and validates candidate pools and their audit summaries.
- `factory.py` assembles online or streaming recall from canonical artifacts.

## Online and offline paths

`RecallPipeline` is the readable online path. It calls each available source,
merges duplicate candidates, and reports counts, shortages, exclusivity, and
deduplication.

`StreamingRecallEngine` is the high-volume path. It batch-searches Audio and
Graph, batches sparse BFS/Tag work, represents tracks with integer codes, and
keeps only the current query batch in memory. Training uses it to avoid
materializing a 980K-query candidate pool.

## Invariants

- Source names and quota keys must match exactly.
- The canonical union contains at most 1,000 unique tracks.
- Recall is deterministic for fixed artifacts and policy.
- Source evidence is audit data; it is not automatically a ranker feature.
- Candidate relevance is not inferred from source availability or coverage.

Supported recall commands are documented in
[`scripts/recall`](../scripts/recall/README.md).
