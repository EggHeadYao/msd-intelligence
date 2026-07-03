# Avro MapReduce BFS

Sample usage: `AvroMapReduceBfs <adjacency.avro> <source_artist_id> <output-dir>`

```bash
mvn exec:java -Dexec.mainClass=artistdistance.mapreduce.avro.AvroMapReduceBfs -Dexec.args="-Dmapreduce.framework.name=local ../data/artistdistance-output/avro/adjacency.avro ARGUACZ1187FB3F35C ../data/artistdistance-bfs-avro-mapreduce"
```

