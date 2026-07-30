# Ranking

Ranking converts a recalled `(query, candidate)` pair into canonical features
and scores it with the frozen logistic-regression model.

## Modules

- `features/` owns pair-signal computation, feature order, raw-feature schemas,
  and feature artifacts.
- `model.py` implements Python LR inference and reads/writes the schema,
  scaler, coefficients, and training manifest.
- `selection.py` selects regularization, the Audio quota, and a query-level
  relation gate on Set-B tune, then verifies the single proposal on Set-B
  confirm.

## Scoring contract

The ranker receives features in the frozen schema order, applies Set-A fill
values and scaling statistics, then computes the raw LR margin. The published
model also freezes an Audio quota and relation-evidence threshold. A query below
the threshold uses quota 20 and is therefore exactly C1-only; otherwise LR and
C1 Audio order are deterministically interleaved for the top 20, with track ID
as score tie-breaker.

Relation evidence is a list-density signal: the maximum of mean valid BFS,
mean valid Tag, and same-release fraction across the recalled candidates.
Missing pair signals contribute zero. This avoids a single saturated candidate
forcing every query through the learned branch.

Fusion is publishable only when the tune guards pass and the untouched Set-B
confirmation fold improves the three-strata macro over C1 by at least 1% while
retaining the Audio, Relation, and Mixed guards. Failure produces a usable C1
fallback. Set-C development evaluation can diagnose either outcome.

Training and Python inference must agree on feature values and raw margin
within `1e-6`. Feature-schema, scaler, coefficient, or parent-lineage mismatch
causes model loading to fail.

Recall-source flags and popularity are audit fields, not model inputs. The
current production ranker does not apply MMR.
