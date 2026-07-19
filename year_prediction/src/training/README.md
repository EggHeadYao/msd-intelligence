# Year Prediction Training

The training layer keeps model mathematics separate from Spark execution and experiment configuration. Experiment labels such as D0 are configuration names, not trainer implementations.

## Files

- `objectives.py`: linear prediction, squared-loss point gradients, partial reduction, and Ridge L2 finalization.
- `optimizer.py`: gradient norms and finite checked gradient-descent updates.
- `distributed.py`: Spark execution and validation prediction helpers. The current direct full-batch path intentionally uses per-record partials followed by `reduce`.
- `training_data.py`: linear feature schema validation, train/validation isolation, dimension discovery, and RDD views.
- `model_io.py`: JSON checksums, atomic JSON writes, and immutable output directory handling.
- `train_constants.py`: training-label mean/median baselines evaluated on validation artists.
- `train_sgd.py`: reusable custom Spark Ridge trainer and model artifact writer.
- `train_mllib_reference.py`: legacy MLlib Ridge result used only as an external reference.

## Commands

```bash
spark-submit --master 'local[*]' --driver-memory 4g \
  p1team02/year_prediction/src/training/train_constants.py
```

```bash
spark-submit --master 'local[*]' --driver-memory 4g \
  p1team02/year_prediction/src/training/train_sgd.py \
  --config p1team02/year_prediction/config/experiment_b/d0_direct.json
```

```bash
spark-submit --master 'local[*]' --driver-memory 4g \
  p1team02/year_prediction/src/training/train_mllib_reference.py \
  --config p1team02/year_prediction/config/experiment_b/d0_direct.json
```

```bash
spark-submit --master 'local[*]' --driver-memory 4g \
  p1team02/year_prediction/src/evaluation/validate_model.py \
  --model parquets/year_prediction/models/&lt;model_id&gt;
```

Use a new `--model-id` for another immutable run. `--overwrite` is available only for deliberate local reruns.
The D0 configuration starts with zero feature weights and the training-label mean as the unregularized intercept, so optimization begins at the Mean baseline rather than at year 1922.

## Custom Ridge Outputs

- `model.json`: weights, intercept, objective, feature dimension, target contract, and feature metadata fingerprint.
- `history.json`: per-iteration objective, gradient norm, validation MAE, row count, and separate gradient, update, validation, and total times.
- `metrics.json`: final train/validation metrics, objective, gradient norm, iteration count, and stop reason.
- `run_metadata.json`: exact configuration, Spark identity, input counts, checksums, and timing boundaries.
- `validation_predictions.parquet`: raw normalized prediction, raw year prediction, and the clipped prediction used by the primary metrics.
