# Parquet MapReduce BFS

- `ParquetMapReduceBfs`: CLI entry point for Parquet input and output.
- `ParquetBfsFormat`: Configures Parquet input/output classes and schemas for each job.
- `ParquetBfsInitMapper`: Converts adjacency rows into initial BFS vertices.
- `ParquetBfsIterationMapper`: Keeps each vertex and expands frontier vertices to candidate messages.
- `ParquetBfsIterationReducer`: Merges vertex and candidate messages for one BFS iteration.
- `ParquetBfsFinalMapper`: Converts final BFS vertices into artist-distance output records.
