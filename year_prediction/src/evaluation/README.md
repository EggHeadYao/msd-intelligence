# Year Prediction Evaluation

This directory checks regression metrics and the reproducibility of saved model artifacts.

- `metrics.py`: accumulates clipped and unclipped year-prediction MAE, RMSE, signed error, and out-of-range rates.
- `validate_ridge.py`: reloads a trained custom Ridge model, recomputes train and validation results, and verifies every saved validation prediction.

`validate_ridge.py` validates artifact consistency. It does not retrain the model, select hyperparameters, or evaluate the held-out test split.

```bash
spark-submit --master 'local[*]' --driver-memory 4g \
  p1team02/year_prediction/src/evaluation/validate_ridge.py \
  --model parquets/year_prediction/models/<model_id>
```
