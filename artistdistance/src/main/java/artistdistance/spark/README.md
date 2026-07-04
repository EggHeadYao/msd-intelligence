# Spark BFS

- `SparkBfsConfig`: Shared Spark configuration keys.
- `SparkBfsFormat`: Interface for format-specific Spark input and output.
- `SparkBfsRunner`: Runs the shared frontier-based BFS using only Spark DataFrames.
- `SparkBfsSession`: Creates Spark sessions with shared local defaults.
- `avro/`: Avro Spark implementation.
- `parquet/`: Parquet Spark implementation.
