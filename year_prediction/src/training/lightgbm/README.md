# Spark LightGBM year prediction

This implementation uses the SynapseML `LightGBMRegressor`. Spark partitions
transfer feature blocks to LightGBM workers, and the native distributed learner
fits a Huber regression objective. No pandas or standalone `lightgbm.train` path
is used.

## Runtime

- Java 17
- Apache Spark 3.5.x with PySpark
- SynapseML 1.1.3 for Scala 2.12

On Linux, Maven resolution can supply the Spark extension:

```bash
export PYSPARK_PYTHON=python3
SYNAPSEML=com.microsoft.azure:synapseml_2.12:1.1.3
```

## Training

Training reads only `train` and `validation`. Test data remains locked for the
separate evaluator.

```bash
spark-submit \
  --master spark://spark-master:7077 \
  --packages "$SYNAPSEML" \
  year_prediction/src/training/lightgbm/lightgbm_train.py \
  --input data/year_prediction/full_tabular.parquet \
  --manifest data/year_prediction/manifest.json \
  --output artifacts/lightgbm
```

The output contains the native `model.txt`, feature contract, arguments,
validation predictions, quality metrics, baselines, and Spark timing metadata.

## Validation-only tuning

```bash
spark-submit \
  --master spark://spark-master:7077 \
  --packages "$SYNAPSEML" \
  year_prediction/src/training/lightgbm/lightgbm_tune.py \
  --input data/year_prediction/full_tabular.parquet \
  --manifest data/year_prediction/manifest.json \
  --output artifacts/lightgbm-tuning
```

## Test evaluation

```bash
spark-submit \
  --master spark://spark-master:7077 \
  --packages "$SYNAPSEML" \
  year_prediction/src/evaluation/lightgbm/evaluate.py \
  --model-root artifacts/lightgbm \
  --input data/year_prediction/full_tabular.parquet \
  --manifest data/year_prediction/manifest.json \
  --output artifacts/lightgbm-test
```
