# Kernel SGD Integration Tests

The three pipeline tests build a partitioned synthetic T90 dataset, train one
model, reload its persisted transform, evaluate the locked test split, and check
the model, metric, and prediction artifacts.

