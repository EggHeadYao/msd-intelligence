# Feature Tests

- `test_preprocessing.py`: verifies train-only fitted statistics, unknown time-signature handling, missing segment imputation, and binary validation.
- `test_views.py`: verifies fixed finite linear vectors, normalized targets, retained dimensions, and train-standardized feature means.

The tests use small synthetic Spark DataFrames so feature contract regressions are detected without rebuilding the full dataset.
