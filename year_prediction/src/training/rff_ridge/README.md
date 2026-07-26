# T90 plus RFF SGD Ridge

This trainer concatenates the 90 standardized T90 values with deterministic RBF
random Fourier features. It optimizes squared loss plus L2 regularization. The
checked-in configurations compare 256, 512, and 1024 RFF dimensions while
holding the remaining training parameters fixed.

```bash
for dimension in 256 512 1024; do
  spark-submit --master 'local[2]' --driver-memory 2g \
    --conf spark.hadoop.fs.defaultFS=file:/// \
    p1team02/year_prediction/src/training/rff_ridge/train.py \
    --config "p1team02/year_prediction/config/rff_ridge_t90_d${dimension}.json"
done
```

Each run writes an immutable model bundle under
`parquets/year_prediction/models/rff-ridge-t90-d<dimension>/`.
