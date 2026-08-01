# Multi-source recall

This package turns Audio, Graph, BFS, and Tag retrievers into the canonical Candidate union.

## Modules

- `policy.py` owns primary quotas, backfill limits/order, the 1,000-candidate cap, and policy validation.
- `pipeline.py` provides deterministic single-query and bounded-batch online recall with per-source audits.
- `streaming.py` provides integer-coded, bounded-memory batched recall for large training and evaluation jobs.
- `pool.py` persists and validates candidate pools and their audit summaries.
- `factory.py` assembles online or streaming recall from canonical artifacts.

## Online and offline paths

`RecallPipeline` is the readable online path. It calls each available source, merges duplicate candidates, and reports counts, shortages, exclusivity, and deduplication.

`StreamingRecallEngine` is the high-volume path. It batch-searches Audio and Graph, batches sparse BFS/Tag work, represents tracks with integer codes, and keeps only the current query batch in memory. Training uses it to avoid materializing an all-catalog candidate pool.

Every source keeps its first 250 nominations. BFS preserves hop-distance order and cycles through the artists at each distance before taking another track from the same artist. Tag considers up to 200 similar artists and likewise cycles through them, using a query-seeded rotation within each artist. If cross-source duplication or a source shortage leaves capacity, Tag and then BFS contribute additional nominations, up to 500 each and never above 1,000 unique candidates. Audio and Graph searches are not expanded by this policy.

Persisted candidates keep both `recall_sources` (all primary and backfill origins) and `primary_recall_sources` (rank at most 250). Candidate Recall@250 uses only the latter, so backfill can improve union recall without inflating a single-source metric.

## Invariants

- Source names and quota keys must match exactly.
- The canonical union contains at most 1,000 unique tracks.
- Backfill never removes a primary-quota nomination.
- Recall is deterministic for fixed artifacts and policy.
- BFS takes at most ten tracks per reachable artist and allocates them one per artist per pass within each hop distance.
- Tag takes at most three tracks from each of up to 200 similar artists, allocates them one per artist per pass, and rotates each stable artist track list by query ID, artist ID, and seed.
- Source evidence is audit data; it is not automatically a ranker feature.
- Candidate relevance is not inferred from source availability or coverage.

Supported recall commands are documented in [`scripts/recall`](../scripts/recall/README.md).
