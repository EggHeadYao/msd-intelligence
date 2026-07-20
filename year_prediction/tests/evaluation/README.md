# Evaluation Tests

- `test_metrics.py`: verifies year denormalization, clipping, MAE, RMSE, and raw out-of-range accounting on deterministic predictions.
- `ridge/`: contains Ridge-specific evaluation tests; see `ridge/README.md`.

These tests validate formulas and evaluation behavior on deterministic fixtures; full-dataset outputs measure model quality separately.
