# MapReduce BFS

- `BfsMessage`: Writable value used between the iteration mapper and reducer.
- `BfsCounter`: Shared Hadoop counters used by Avro and Parquet jobs.
- `BfsIterationStep`: Shared reducer logic for one BFS iteration.
- `MapReduceBfsConfig`: Shared configuration keys for source id, max iterations, and reducers.
- `MapReduceBfsFormat`: Interface for Avro and Parquet job configuration.
- `MapReduceBfsRunner`: Runs init, iteration, and final MapReduce jobs for one format.
- `avro/`: Avro MapReduce implementation.
- `parquet/`: Parquet MapReduce implementation.

Sample usage: `ParquetMapReduceBfs <adjacency.parquet> <source_artist_id> <output-dir>`

```bash
mvn exec:java -Dexec.mainClass=artistdistance.mapreduce.parquet.ParquetMapReduceBfs -Dexec.args="-Dmapreduce.framework.name=local ../data/artistdistance-output/parquet/adjacency.parquet ARGUACZ1187FB3F35C ../data/artistdistance-bfs-parquet-mapreduce"
```

