# Integration Tests

- `test_t90_ridge_pipeline.py`: runs a small T90-vector-to-Ridge pipeline, checks all model artifacts, and rejects artist overlap between train and validation.
- `test_lightgbm_pipeline.py`: reserved for the full-tabular LightGBM pipeline.

The synthetic integration dataset checks pipeline behavior, not final MSD prediction quality.
