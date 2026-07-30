# Validate

- `AvroRecords`: Reads Avro records from a file or directory.
- `ParquetRecords`: Reads one Parquet file or all Parquet part files in a directory.
- `ReferenceBfs`: Computes independent in-memory BFS results from adjacency rows.
- `AvroBfsOutputVerifier`: Compares Avro BFS output against the reference BFS result.
- `ParquetBfsOutputVerifier`: Compares Parquet BFS output against the reference BFS result.

## Avro Output Verifier

Sample usage: `AvroBfsOutputVerifier <adjacency.avro> <final-dir> <source_id>`

```bash
mvn -q exec:java -Dexec.mainClass=artistdistance.validate.AvroBfsOutputVerifier -Dexec.args="../data/artistdistance-output/avro/adjacency.avro ../data/artistdistance-bfs-avro-spark/ ARGUACZ1187FB3F35C"
```

## Parquet Output Verifier

Sample usage: `ParquetBfsOutputVerifier <adjacency.parquet> <final-dir> <source_id>`

```bash
mvn -q exec:java -Dexec.mainClass=artistdistance.validate.ParquetBfsOutputVerifier -Dexec.args="../data/artistdistance-output/parquet/adjacency.parquet ../data/artistdistance-bfs-parquet-spark/ ARGUACZ1187FB3F35C"
```

## Sample Output

```
expected_total=44745
reachable=43228
unreachable=1517
max_distance=10
mismatches=0
```
