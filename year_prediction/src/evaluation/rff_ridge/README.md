# T90 plus RFF SGD Ridge Evaluation

```bash
for dimension in 256 512 1024; do
  spark-submit --master 'local[2]' --driver-memory 2g \
    --conf spark.hadoop.fs.defaultFS=file:/// \
    p1team02/year_prediction/src/evaluation/rff_ridge/evaluate.py \
    --model-root "parquets/year_prediction/models/rff-ridge-t90-d${dimension}" \
    --input parquets/year_prediction/training/t90/vectors.parquet \
    --output parquets/year_prediction/results/experiment_a/rff_ridge
done
```

Each run writes test artifacts under
`parquets/year_prediction/results/experiment_a/rff_ridge/rff-ridge-t90-d<dimension>/test/`.
