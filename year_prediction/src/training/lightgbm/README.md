# Spark LightGBM year prediction

This trainer uses SynapseML LightGBM with manifest-defined feature views. The supported experiment configurations use either the 90-predictor `t90.parquet` view or the 594-predictor `full_tabular.parquet` view. Training keeps the fixed artist-disjoint splits, supports Huber/L1 and L2/RMSE objectives, and never includes test rows in gradient updates.

## Runtime

- Java 17
- Apache Spark 3.5.x with Scala 2.12
- SynapseML 1.1.3

Create the isolated Python runtime from the repository parent:

```bash
unset SPARK_HOME PYTHONPATH
python3 -m venv p1team02/year_prediction/.synapseml-venv
p1team02/year_prediction/.synapseml-venv/bin/pip install \
  -r p1team02/year_prediction/requirements-synapseml.txt
```

On ARM Linux, SynapseML also needs locally built `lib_lightgbm.so` and `lib_lightgbm_swig.so` under `p1team02/year_prediction/.synapseml-native/lib/`.

Build prerequisites on Ubuntu are `swig`, `libboost-dev`, `libboost-system-dev`, `libboost-filesystem-dev`, and `libeigen3-dev`. After installing them, build the ARM64 libraries with:

```bash
p1team02/year_prediction/src/training/lightgbm/build_arm_native.sh
```

The script pins LightGBM commit `0957ab7`, whose JNI surface matches the `lightgbmlib-3.3.510` dependency bundled with SynapseML 1.1.3.

## Training

```bash
unset SPARK_HOME PYTHONPATH
export SYNAPSEML=com.microsoft.azure:synapseml-lightgbm_2.12:1.1.3
export PYSPARK_PYTHON="$PWD/p1team02/year_prediction/.synapseml-venv/bin/python"
export LD_LIBRARY_PATH="$PWD/p1team02/year_prediction/.synapseml-native/lib"

p1team02/year_prediction/.synapseml-venv/bin/spark-submit \
  --master 'local[2]' \
  --driver-memory 4g \
  --packages "$SYNAPSEML" \
  --conf spark.hadoop.fs.defaultFS=file:/// \
p1team02/year_prediction/src/training/lightgbm/train.py \
  --config p1team02/year_prediction/config/experiment_a/lightgbm_l2_v2.json
```

Use `lightgbm_t90_l2.json` instead to train the T90 model. The model directory contains `model.txt`, the selected feature view and ordered predictors, resolved arguments, validation predictions and metrics, constant baselines, and Spark run metadata. Command-line arguments override values from the JSON config.

`bin_sample_count` controls only the rows used to construct histogram bins. Every train row is still used to fit the trees. The assembled feature frame uses disk-only persistence so either feature view fits on memory-constrained development machines.

## Smoke test

```bash
p1team02/year_prediction/.synapseml-venv/bin/spark-submit \
  --master 'local[2]' \
  --driver-memory 4g \
  --packages "$SYNAPSEML" \
  --conf spark.hadoop.fs.defaultFS=file:/// \
  p1team02/year_prediction/src/training/lightgbm/train.py \
  --config p1team02/year_prediction/config/experiment_a/lightgbm_full.json \
  --output /tmp/year-prediction-lightgbm-smoke \
  --max-rows-per-split 1000 \
  --num-iterations 20 \
  --early-stopping-rounds 5 \
  --num-tasks 2 \
  --overwrite
```

## Validation tuning

`lightgbm_tune.py` reads only train and validation rows. It records every trial in `trials.json` and the lowest-validation-MAE trial in `best.json`. Transfer the selected parameters into `lightgbm_full.json` before the final full run.
