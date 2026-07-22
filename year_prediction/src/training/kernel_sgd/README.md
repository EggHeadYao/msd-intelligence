# Kernel SGD Core

This package implements reusable PySpark gradient descent for transformed T90
features. Spark RDD partitions compute loss and gradient sums. The driver merges
those statistics, applies one parameter update, and evaluates validation rows at
configured intervals. The intercept is not regularized.

PCA is fitted on training rows only. RFF uses a persisted deterministic frequency
matrix and phase vector. Training never reads the test split. Each run writes
`model.json`, `transform.json`, `transform.npz`, `history.json`, `metrics.json`,
`run_metadata.json`, and validation predictions.

