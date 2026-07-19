# Feature Tests

- `test_key_contracts.py`: verifies K0-K3 names, encoded fields, metadata, field order, and invalid contract rejection.
- `test_preprocessing.py`: verifies train-only fitted statistics, key and time-signature encoding, missing segment imputation, and binary validation.
- `test_views.py`: verifies fixed finite linear vectors, normalized targets, retained dimensions, and train-standardized feature means.

The tests use small synthetic Spark DataFrames so feature contract regressions are detected without rebuilding the full dataset.
