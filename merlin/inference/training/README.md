# Training data construction

This package contains the reusable algorithms behind C3 split, weak-label,
pair-sampling, and validation-group stages. Spark and CLI orchestration remains
under `scripts/ranker`.

## Modules

- `split.py` deterministically assigns song-safe Set A, Set B, Set C, and
  Remaining partitions.
- `weak_labels.py` fits frozen similarity thresholds and selects weak positive
  pairs.
- `pairs.py` builds positive/negative rows, performs candidate-aware and random
  sampling, writes partitioned pairs/features, and manages resume checkpoints.
- `validation_groups.py` builds compact nested Audio-dominant,
  Relation-dominant, and Mixed query labels.

## Invariants

- Splitting happens before pair construction, and a song cannot cross splits.
- No Set-C endpoint may enter tuning or retraining data.
- A negative must pass the complete frozen positive predicate.
- Canonical training maintains the required 1:3 positive/negative ratio.
- Set-B thresholds use cleaned, scaled pre-PCA C1 signals rather than the
  PCA-128 FAISS cosine.
- Validation stores each `(query, candidate)` feature vector once and attaches
  compact group labels instead of duplicating rows.
- Streamed retraining checkpoints bind the input/output contract and may only
  resume an identical command configuration.

The execution order is documented in
[`scripts/ranker`](../scripts/ranker/README.md).
