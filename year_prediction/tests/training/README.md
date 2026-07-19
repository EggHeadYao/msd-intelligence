# Training Tests

- `test_objectives.py`: compares production Ridge loss and gradients with the
  independent oracle and checks one golden parameter update.
- `test_distributed.py`: verifies that production Spark aggregation matches
  the local oracle with 1, 2, and 4 partitions.

Together these tests check production mathematics and partition invariance; full artifact generation is covered by `tests/integration/`.
