# Year Prediction Training

The training layer turns validated feature views into model-ready inputs, trains the two model families, and persists reproducible artifacts.

## Shared Files

- `target.py`: the fixed `[1922, 2011]` year normalization contract.
- `model_io.py`: JSON loading, checksums, atomic JSON writes, and immutable output handling.
- `train_constants.py`: mean and median sanity-check baselines.

Run the constant baselines before fitting a learned model:

```bash
spark-submit --master 'local[4]' --driver-memory 3g \
  p1team02/year_prediction/src/training/train_constants.py
```

## Model Families

- `ridge/`: T90 preprocessing, custom Spark SGD Ridge, and its validation tools.
- `lightgbm/`: full-tabular loading, strongest-model training, and feature audit.

Each model directory documents its own commands, input contract, and outputs.
