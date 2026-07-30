# Spark LightGBM year prediction

This trainer uses SynapseML LightGBM with manifest-defined audio, metadata-only, and fused feature views. Training keeps the fixed artist-disjoint splits, supports Huber/L1 and L2/RMSE objectives, and never includes test rows in gradient updates.

## Runtime

- Java 17
- Apache Spark 3.5.x with Scala 2.12
- SynapseML 1.1.3

Create the isolated Python runtime from the repository root:

```bash
unset SPARK_HOME PYTHONPATH
python3 -m venv src/year_prediction/.synapseml-venv
src/year_prediction/.synapseml-venv/bin/pip install \
  -r src/year_prediction/requirements-synapseml.txt
```

On ARM Linux, SynapseML also needs locally built `lib_lightgbm.so` and `lib_lightgbm_swig.so` under `src/year_prediction/.synapseml-native/lib/`.

Build prerequisites on Ubuntu are `swig`, `libboost-dev`, `libboost-system-dev`, `libboost-filesystem-dev`, and `libeigen3-dev`. After installing them, build the ARM64 libraries with:

```bash
src/year_prediction/src/training/lightgbm/build_arm_native.sh
```

The script pins LightGBM commit `0957ab7`, whose JNI surface matches the `lightgbmlib-3.3.510` dependency bundled with SynapseML 1.1.3.

## Training

```bash
unset SPARK_HOME PYTHONPATH
export SYNAPSEML=com.microsoft.azure:synapseml-lightgbm_2.12:1.1.3
export PYSPARK_PYTHON="$PWD/src/year_prediction/.synapseml-venv/bin/python"
export LD_LIBRARY_PATH="$PWD/src/year_prediction/.synapseml-native/lib"

src/year_prediction/.synapseml-venv/bin/spark-submit \
  --master 'local[2]' \
  --driver-memory 4g \
  --packages "$SYNAPSEML" \
  --conf spark.hadoop.fs.defaultFS=file:/// \
  src/year_prediction/src/training/lightgbm/train.py \
  --config src/year_prediction/config/lightgbm_l2_regularized.json
```

The regularized configuration is the RMSE-focused 594-predictor model. Use `lightgbm_metadata_rmse.json` for metadata only and `lightgbm_audio_metadata_tags_rmse.json` for the 762-predictor fused model. Use `lightgbm_full.json` for the Huber/MAE model or `lightgbm_t90_l2.json` for the T90 comparison. The model directory contains `model.txt`, the selected feature view and ordered predictors, resolved arguments, validation predictions and metrics, constant baselines, and Spark run metadata. Command-line arguments override values from the JSON config.

`bin_sample_count` controls only the rows used to construct histogram bins. Every train row is still used to fit the trees. The assembled feature frame uses disk-only persistence so either feature view fits on memory-constrained development machines.

Optional decade weighting is enabled with a positive `decade_weight_power`. It derives capped inverse-frequency weights from train rows only, normalizes their train-row mean to one, and leaves validation and test metrics unweighted. The default value of zero disables weighting.

## Smoke test

```bash
src/year_prediction/.synapseml-venv/bin/spark-submit \
  --master 'local[2]' \
  --driver-memory 4g \
  --packages "$SYNAPSEML" \
  --conf spark.hadoop.fs.defaultFS=file:/// \
  src/year_prediction/src/training/lightgbm/train.py \
  --config src/year_prediction/config/lightgbm_full.json \
  --output /tmp/year-prediction-lightgbm-smoke \
  --max-rows-per-split 1000 \
  --num-iterations 20 \
  --early-stopping-rounds 5 \
  --num-tasks 2 \
  --overwrite
```

## Validation tuning

`lightgbm_tune.py` reads only train and validation rows. It records every trial in `trials.json` and the lowest-validation-MAE trial in `best.json`. Transfer the selected parameters into `lightgbm_full.json` before the final full run.
