# Parquet Spark BFS

- `ParquetSparkBfs`: CLI entry point for running DataFrame BFS with Parquet input and output.
- `ParquetSparkBfsFormat`: Reads Parquet adjacency data and writes Parquet artist-distance results.

Sample usage: `ParquetSparkBfs <adjacency.parquet> <source_artist_id> <output-dir>`

```bash
MAVEN_OPTS="-Xmx4g" mvn exec:java -Dexec.mainClass=artistdistance.spark.parquet.ParquetSparkBfs -Dexec.args="../data/artistdistance-output/parquet/adjacency.parquet ARGUACZ1187FB3F35C ../data/artistdistance-bfs-parquet-spark"
```

Sample output:

```
[4.0K]  data/artistdistance-bfs-parquet-spark/
├── [869K]  part-00000-51b4c5e7-cad7-467f-82d7-3a2022fb6da2-c000.snappy.parquet
├── [6.8K]  .part-00000-51b4c5e7-cad7-467f-82d7-3a2022fb6da2-c000.snappy.parquet.crc
├── [   0]  _SUCCESS
└── [   8]  ._SUCCESS.crc
```
