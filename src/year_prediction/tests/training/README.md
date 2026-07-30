# Training Tests

- `ridge/test_prepare_t90.py`: checks target normalization, train-only T90 fitting, missing-value imputation, full artifact generation, and independent validation.
- `ridge/test_objectives.py`: compares production Ridge loss and gradients with the independent oracle and checks one golden parameter update.
- `ridge/test_distributed.py`: verifies that production Spark aggregation matches the local oracle with 1, 2, and 4 partitions.
- `ridge/test_optimizer.py`: covers production gradient updates.
- `lightgbm/`: reserved for full-tabular loading and feature-audit tests.

Together these tests check production mathematics and partition invariance; full artifact generation is covered by `tests/integration/`.
