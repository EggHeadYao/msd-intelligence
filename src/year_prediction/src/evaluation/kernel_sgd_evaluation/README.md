# Kernel SGD Test Evaluation

The evaluator reloads the linear parameters and persisted transform, reads only
the test split, reproduces PCA or RFF features in Spark workers, clips predictions
to 1922 through 2011, and writes predictions, metrics, and runtime metadata.
