# RFF Huber SGD

This trainer maps standardized T90 vectors to 512 deterministic RBF random Fourier features and applies Huber loss. The Huber threshold is configured in years and converted to normalized target units before gradient calculation.

```bash
spark-submit --master spark://spark-master:7077 \
  src/year_prediction/src/training/rff_huber/train.py \
  --config src/year_prediction/config/rff_huber_t90.json
```
