# Training data construction

This package contains the reusable algorithms behind C3 split, weak-label,
pair-sampling, and validation-group stages. Spark and CLI orchestration remains
under `scripts/ranker`.

## Modules

- `split.py` deterministically assigns song-safe Set A, Set B, reusable
  development Set C, and Remaining partitions.
- `weak_labels.py` fits frozen similarity thresholds and selects weak positive
  pairs.
- `pairs.py` builds positive/negative rows, performs candidate-aware and random
  sampling, writes partitioned pairs/features, and manages resume checkpoints.
- `validation_groups.py` builds compact nested Audio-dominant,
  Relation-dominant, and Mixed query labels.

## Invariants

- Splitting happens before pair construction, and a song cannot cross splits.
- Set B contains disjoint song-safe tune (1%) and confirm (2%) folds; tune may
  choose one configuration and confirm may only accept it or force C1 fallback.
- Tuning uses only Set A. Final retraining uses Set A, Set B, Set C, and
  Remaining; Set C metrics therefore describe known-data optimization only.
- A negative must pass the complete frozen positive predicate.
- Ranker positives must also occur in that query's canonical recalled candidate
  set; unrecalled weak positives are excluded and counted in the pair manifest.
- Canonical training maintains the required 1:3 positive/negative ratio.
- Candidate-aware negatives rotate deterministically across available Audio,
  Graph, BFS, and Tag sources before random backfill covers any shortage.
- Positive loss mass is balanced within each query between Audio-derived and
  relation-derived positives; a missing modality receives no artificial mass.
- Set-B thresholds use cleaned, scaled pre-PCA C1 signals rather than the
  PCA-128 FAISS cosine.
- Validation stores each `(query, candidate)` feature vector once and attaches
  compact group labels plus its immutable selection fold instead of duplicating
  rows.
- Streamed retraining checkpoints bind the input/output contract and may only
  resume an identical command configuration.

The execution order is documented in
[`scripts/ranker`](../scripts/ranker/README.md).
