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

- `reference.py`: independent local loss, gradient, finite-difference, update, and comparison functions.
- `spark_oracle.py`: Spark partition aggregation and local-versus-Spark validation.
- `fixture.json`: deterministic inputs, tolerances, and golden outputs.
- `test_reference.py`: local unit tests for the mathematical contract.
- `test_spark.py`: Spark integration test across multiple partition counts.

## Run

Run the local unit tests:

```bash
python3 -m unittest discover \
  -s src/year_prediction/tests/oracles/ridge \
  -p 'test_reference.py'
```

Run the Spark oracle:

```bash
spark-submit --master 'local[*]' \
  src/year_prediction/tests/oracles/ridge/spark_oracle.py
```

A successful Spark run prints a JSON object with `"status": "valid"`.
