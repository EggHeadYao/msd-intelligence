# MapReduce BFS

- `BfsMessage`: Writable value used between the iteration mapper and reducer.
- `BfsCounter`: Shared Hadoop counters used by Avro and Parquet jobs.
- `BfsIterationStep`: Shared reducer logic for one BFS iteration.
- `MapReduceBfsConfig`: Shared configuration keys for source id, max iterations, and reducers.
- `MapReduceBfsFormat`: Interface for Avro and Parquet job configuration.
- `MapReduceBfsRunner`: Runs init, iteration, and final MapReduce jobs for one format.
- `avro/`: Avro MapReduce implementation.
- `parquet/`: Parquet MapReduce implementation.
