# Linear prediction ensemble

This module fits an ordinary least-squares combination of the fused,
metadata-only, and audio-only LightGBM predictions. The coefficients are fitted
on artist-disjoint validation predictions and then frozen before test inference.

Run from the repository root:

```bash
export PYSPARK_PYTHON="$PWD/p1team02/year_prediction/.synapseml-venv/bin/python"

p1team02/year_prediction/.synapseml-venv/bin/spark-submit --master 'local[4]' \
  --conf spark.hadoop.fs.defaultFS=file:/// \
  p1team02/year_prediction/src/training/linear_ensemble/fit.py \
  --fused-validation parquets/year_prediction/models/lightgbm-audio-metadata-tags-rmse/validation_predictions.parquet \
  --fused-test parquets/year_prediction/models/lightgbm-audio-metadata-tags-rmse/test_predictions.parquet \
  --metadata-validation parquets/year_prediction/models/lightgbm-metadata-rmse/validation_predictions.parquet \
  --metadata-test parquets/year_prediction/models/lightgbm-metadata-rmse/test_predictions.parquet \
  --audio-validation parquets/year_prediction/models/lightgbm-l2-regularized/validation_predictions.parquet \
  --audio-test parquets/year_prediction/results/model_comparison/lightgbm/lightgbm-l2-regularized/test/test_predictions.parquet \
  --output parquets/year_prediction/models/lightgbm-linear-ensemble-rmse \
  --overwrite
```

The output contains the fitted coefficients, run arguments, validation/test
metrics, and one prediction table for each evaluated split.
