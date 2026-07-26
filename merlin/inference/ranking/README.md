# Ranking

Ranking converts a recalled `(query, candidate)` pair into canonical features
and scores it with the frozen logistic-regression model.

## Modules

- `features/` owns pair-signal computation, feature order, raw-feature schemas,
  and feature artifacts.
- `model.py` implements Python LR inference and reads/writes the schema,
  scaler, coefficients, and training manifest.
- `selection.py` selects the regularization parameter on frozen Set-B query
  groups using paired bootstrap comparisons.

## Scoring contract

The ranker receives features in the frozen schema order, applies Set-A fill
values and scaling statistics, then computes the raw LR margin. Candidates are
sorted by descending margin with track ID as the deterministic tie-breaker.

Training and Python inference must agree on feature values and raw margin
within `1e-6`. Feature-schema, scaler, coefficient, or parent-lineage mismatch
causes model loading to fail.

Recall-source flags and popularity are audit fields, not model inputs. The
current production ranker does not apply MMR.
