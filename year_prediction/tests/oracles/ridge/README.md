# Ridge Gradient Oracle

The oracle defines the correctness contract used by the year-prediction Ridge trainer. It operates on a small deterministic fixture and does not train a model on the MSD dataset.

## Contract

- The objective is mean squared error plus `l2 * sum(weight^2)`.
- The gradient is averaged by the number of samples.
- The intercept is stored separately and is not L2-regularized.
- Labels in the fixture use the same `[0, 1]` scale planned for year training.
- The analytic gradient must match an independently computed central finite-difference gradient.
- Spark must reproduce the local loss, gradient, count, and one parameter update for every configured partition count.

## Files

- `ridge_math.py`: local loss, analytic gradient, finite-difference gradient, update, and comparison functions.
- `gradient_oracle.py`: Spark partition aggregation and local-versus-Spark validation.
- `tests/fixtures/ridge_oracle.json`: deterministic inputs, tolerances, and golden outputs.
- `tests/test_ridge_math.py`: local unit tests for the mathematical contract.
- `tests/test_gradient_oracle.py`: Spark integration test across multiple partition counts.

## Run

Run the local unit tests:

```bash
python3 -m unittest p1team02/year_prediction/tests/test_ridge_math.py
```

Run the Spark oracle:

```bash
spark-submit --master 'local[*]' \
  p1team02/year_prediction/src/training/gradient_oracle.py
```

A successful Spark run prints a JSON object with `"status": "valid"`.
