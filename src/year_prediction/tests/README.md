# Year Prediction Tests

The test suite separates feature correctness, independent mathematical oracles, production training logic, evaluation metrics, and end-to-end integration.

- `features/`: deterministic feature projection and contract tests.
- `oracles/`: independent mathematical contracts that do not import production optimizer code.
- `training/`: model-ready preprocessing, production objectives, updates, and Spark aggregation tests.
- `evaluation/`: regression metric tests.
- `integration/`: small Parquet-to-model pipeline tests and cross-split leakage guards.
