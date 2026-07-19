# Year Prediction Experiments

- `run_key_contracts.sh`: sequentially trains K0, K1, K2, and K3 with the same Ridge optimizer and local Spark resources, then validates every model artifact.

The script uses six of the local machine's eight CPU cores and 4 GiB of driver memory. Existing model directories are never overwritten.

Run it from any directory after all four feature contracts have been built:

```bash
p1team02/year_prediction/experiments/run_key_contracts.sh
```
