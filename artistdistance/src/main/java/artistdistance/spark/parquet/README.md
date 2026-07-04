# Parquet Spark BFS

- `ParquetSparkBfs`: CLI entry point for running DataFrame BFS with Parquet input and output.
- `ParquetSparkBfsFormat`: Reads Parquet adjacency data and writes Parquet artist-distance results.

Sample usage: `ParquetSparkBfs <adjacency.parquet> <source_artist_id> <output-dir>`

```bash
MAVEN_OPTS="-Xmx4g" mvn exec:java -Dexec.mainClass=artistdistance.spark.parquet.ParquetSparkBfs -Dexec.args="../data/artistdistance-output/parquet/adjacency.parquet ARGUACZ1187FB3F35C ../data/artistdistance-bfs-parquet-spark"
```
