# Year prediction training

All scalable model paths use PySpark. Input rows stay in Spark DataFrames or RDDs, preprocessing is fitted on training rows, and partition results are reduced before optimizer updates.

## Model families

- `ridge`: custom Spark SGD Ridge on the T90 view.
- `lightgbm`: SynapseML distributed Huber LightGBM on 594 tabular features.
- `ordinal_moe`: distributed analytic-gradient Ordinal-MoE on the same contract.

Each model directory documents Linux `spark-submit` commands and artifact contracts. Normal training may use train and validation only. The fixed test split is reserved for scripts under `src/evaluation`.

Run tests from the repository root:

```bash
python3 -m unittest discover -s src/year_prediction/tests -p 'test_*.py'
```

Run SynapseML integration tests through `spark-submit` with the package supplied to the driver and executors.
