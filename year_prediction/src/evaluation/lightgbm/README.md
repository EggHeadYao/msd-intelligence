# Spark LightGBM evaluation

The evaluator loads a native SynapseML LightGBM model, verifies the saved
594-feature order hash, reads only the fixed test split, and computes metrics
with Spark aggregations.

```bash
SYNAPSEML=com.microsoft.azure:synapseml_2.12:1.1.3

spark-submit \
  --master spark://spark-master:7077 \
  --packages "$SYNAPSEML" \
  year_prediction/src/evaluation/lightgbm/evaluate.py \
  --model-root artifacts/lightgbm \
  --input data/year_prediction/full_tabular.parquet \
  --manifest data/year_prediction/manifest.json \
  --output artifacts/lightgbm-test
```

Outputs include partitioned predictions, overall and decade metrics, and Spark
run metadata.
