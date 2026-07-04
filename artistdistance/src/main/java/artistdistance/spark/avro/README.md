# Avro Spark BFS

- `AvroSparkBfs`: CLI entry point for running DataFrame BFS with Avro input and output.
- `AvroSparkBfsFormat`: Reads Avro adjacency data and writes Avro artist-distance results.

Sample usage: `AvroSparkBfs <adjacency.avro> <source_artist_id> <output-dir>`

```bash
MAVEN_OPTS="-Xmx4g" mvn exec:java -Dexec.mainClass=artistdistance.spark.avro.AvroSparkBfs -Dexec.args="../data/artistdistance-output/avro/adjacency.avro ARGUACZ1187FB3F35C ../data/artistdistance-bfs-avro-spark"
```
