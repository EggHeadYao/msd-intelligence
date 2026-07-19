# Year Prediction Tests

The test suite separates feature correctness, independent mathematical oracles, production training logic, evaluation metrics, and end-to-end integration.

- `features/`: feature fitting, transformation, and vector-view tests.
- `oracles/`: independent mathematical contracts that do not import production optimizer code.
- `training/`: production Ridge objective, update, and Spark aggregation tests.
- `evaluation/`: regression metric tests.
- `integration/`: small Parquet-to-model pipeline tests and cross-split leakage guards.
