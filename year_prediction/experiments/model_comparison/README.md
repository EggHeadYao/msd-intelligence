# Year Prediction Model Comparison

This directory stores lightweight reports for the retained model comparisons. Every `metrics.json` uses the same top-level `model_id`, `validation`, and `test` fields. The validation and test sets are artist-disjoint from the training set.

The reported test metrics reflect the course-approved test-guided tuning protocol. They should not be described as estimates from an untouched test set.

- `ridge-t90`: custom Spark SGD Ridge baseline on the T90 representation.
- `rff-ridge-d*`: RBF random Fourier feature dimension comparison on T90.
- `lightgbm-t90`: nonlinear T90 comparison.
- `lightgbm-audio`: audio-only 594-predictor LightGBM.
- `lightgbm-metadata`: metadata-only 392-predictor LightGBM.
- `lightgbm-fused`: fused 762-predictor LightGBM.
- `lightgbm-ensemble`: validation-fitted combination of three LightGBM models; `model.json` records its weights and intercept.

`summary.csv` contains the headline MAE and RMSE values. Full model artifacts and per-track prediction Parquet files remain under `parquets/year_prediction/` and are intentionally excluded from Git.
