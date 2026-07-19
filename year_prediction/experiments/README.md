# Year Prediction Experiments

- `run_key_contracts.sh`: sequentially trains K0, K1, K2, and K3 with the same Ridge optimizer and local Spark resources, then validates every model artifact.
- `run_ridge_learning_rates.sh`: sequentially trains and validates the five K0 Ridge learning-rate candidates with fixed `L2=0.001` and 100 iterations.

The script uses six of the local machine's eight CPU cores and 4 GiB of driver memory. Existing model directories are never overwritten.

Run it from any directory after all four feature contracts have been built:

```bash
p1team02/year_prediction/experiments/run_key_contracts.sh
```

Run the Ridge learning-rate sweep after K0 has been built:

```bash
p1team02/year_prediction/experiments/run_ridge_learning_rates.sh
```
