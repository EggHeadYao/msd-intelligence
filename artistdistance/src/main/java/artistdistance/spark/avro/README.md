# Avro Spark BFS

- `AvroSparkBfs`: CLI entry point for running DataFrame BFS with Avro input and output.
- `AvroSparkBfsFormat`: Reads Avro adjacency data and writes Avro artist-distance results.

Sample usage: `AvroSparkBfs <adjacency.avro> <source_artist_id> <output-dir>`

```bash
MAVEN_OPTS="-Xmx4g" mvn exec:java -Dexec.mainClass=artistdistance.spark.avro.AvroSparkBfs -Dexec.args="../data/artistdistance-output/avro/adjacency.avro ARGUACZ1187FB3F35C ../data/artistdistance-bfs-avro-spark"
```

Sample output:

```
[4.0K]  data/artistdistance-bfs-avro-spark/
├── [1.1M]  part-00000-a4202a1c-c2ce-4e08-864c-2f4a59d837b7-c000.snappy.avro
├── [9.2K]  .part-00000-a4202a1c-c2ce-4e08-864c-2f4a59d837b7-c000.snappy.avro.crc
├── [   0]  _SUCCESS
└── [   8]  ._SUCCESS.crc
```
