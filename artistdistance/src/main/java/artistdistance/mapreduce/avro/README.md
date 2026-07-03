# Avro MapReduce BFS

Sample usage: `AvroMapReduceBfs <adjacency.avro> <source_artist_id> <output-dir>`

```bash
mvn exec:java -Dexec.mainClass=artistdistance.mapreduce.avro.AvroMapReduceBfs -Dexec.args="-Dmapreduce.framework.name=local ../data/artistdistance-output/avro/adjacency.avro ARGUACZ1187FB3F35C ../data/artistdistance-bfs-avro-mapreduce"
```

Sample output:

```
[4.0K]  data/artistdistance-bfs-avro-mapreduce/
├── [4.0K]  final
│   ├── [445K]  part-m-00000.avro
│   ├── [3.5K]  .part-m-00000.avro.crc
│   ├── [444K]  part-m-00001.avro
│   ├── [3.5K]  .part-m-00001.avro.crc
│   ├── [441K]  part-m-00002.avro
│   ├── [3.5K]  .part-m-00002.avro.crc
│   ├── [435K]  part-m-00003.avro
│   ├── [3.4K]  .part-m-00003.avro.crc
│   ├── [   0]  _SUCCESS
│   └── [   8]  ._SUCCESS.crc
├── [4.0K]  iter-0000
│   ├── [ 32M]  part-m-00000.avro
│   ├── [257K]  .part-m-00000.avro.crc
│   ├── [8.9M]  part-m-00001.avro
│   ├── [ 71K]  .part-m-00001.avro.crc
│   ├── [   0]  _SUCCESS
│   └── [   8]  ._SUCCESS.crc
├── [4.0K]  iter-0001
│   ├── [ 10M]  part-r-00000.avro
│   ├── [ 82K]  .part-r-00000.avro.crc
│   ├── [ 10M]  part-r-00001.avro
│   ├── [ 82K]  .part-r-00001.avro.crc
│   ├── [ 10M]  part-r-00002.avro
│   ├── [ 81K]  .part-r-00002.avro.crc
│   ├── [ 10M]  part-r-00003.avro
│   ├── [ 83K]  .part-r-00003.avro.crc
│   ├── [   0]  _SUCCESS
│   └── [   8]  ._SUCCESS.crc
├── [4.0K]  iter-0002
│   ├── [ 10M]  part-r-00000.avro
│   ├── [ 82K]  .part-r-00000.avro.crc
│   ├── [ 10M]  part-r-00001.avro
│   ├── [ 82K]  .part-r-00001.avro.crc
│   ├── [ 10M]  part-r-00002.avro
│   ├── [ 81K]  .part-r-00002.avro.crc
│   ├── [ 10M]  part-r-00003.avro
│   ├── [ 83K]  .part-r-00003.avro.crc
│   ├── [   0]  _SUCCESS
│   └── [   8]  ._SUCCESS.crc
├── [4.0K]  iter-0003
│   ├── [ 10M]  part-r-00000.avro
│   ├── [ 82K]  .part-r-00000.avro.crc
│   ├── [ 10M]  part-r-00001.avro
│   ├── [ 83K]  .part-r-00001.avro.crc
│   ├── [ 10M]  part-r-00002.avro
│   ├── [ 81K]  .part-r-00002.avro.crc
│   ├── [ 10M]  part-r-00003.avro
│   ├── [ 83K]  .part-r-00003.avro.crc
│   ├── [   0]  _SUCCESS
│   └── [   8]  ._SUCCESS.crc
├── [4.0K]  iter-0004
│   ├── [ 10M]  part-r-00000.avro
│   ├── [ 83K]  .part-r-00000.avro.crc
│   ├── [ 10M]  part-r-00001.avro
│   ├── [ 83K]  .part-r-00001.avro.crc
│   ├── [ 10M]  part-r-00002.avro
│   ├── [ 82K]  .part-r-00002.avro.crc
│   ├── [ 11M]  part-r-00003.avro
│   ├── [ 84K]  .part-r-00003.avro.crc
│   ├── [   0]  _SUCCESS
│   └── [   8]  ._SUCCESS.crc
├── [4.0K]  iter-0005
│   ├── [ 10M]  part-r-00000.avro
│   ├── [ 84K]  .part-r-00000.avro.crc
│   ├── [ 10M]  part-r-00001.avro
│   ├── [ 84K]  .part-r-00001.avro.crc
│   ├── [ 10M]  part-r-00002.avro
│   ├── [ 82K]  .part-r-00002.avro.crc
│   ├── [ 11M]  part-r-00003.avro
│   ├── [ 85K]  .part-r-00003.avro.crc
│   ├── [   0]  _SUCCESS
│   └── [   8]  ._SUCCESS.crc
├── [4.0K]  iter-0006
│   ├── [ 10M]  part-r-00000.avro
│   ├── [ 84K]  .part-r-00000.avro.crc
│   ├── [ 10M]  part-r-00001.avro
│   ├── [ 84K]  .part-r-00001.avro.crc
│   ├── [ 10M]  part-r-00002.avro
│   ├── [ 82K]  .part-r-00002.avro.crc
│   ├── [ 11M]  part-r-00003.avro
│   ├── [ 85K]  .part-r-00003.avro.crc
│   ├── [   0]  _SUCCESS
│   └── [   8]  ._SUCCESS.crc
├── [4.0K]  iter-0007
│   ├── [ 10M]  part-r-00000.avro
│   ├── [ 84K]  .part-r-00000.avro.crc
│   ├── [ 11M]  part-r-00001.avro
│   ├── [ 84K]  .part-r-00001.avro.crc
│   ├── [ 10M]  part-r-00002.avro
│   ├── [ 82K]  .part-r-00002.avro.crc
│   ├── [ 11M]  part-r-00003.avro
│   ├── [ 85K]  .part-r-00003.avro.crc
│   ├── [   0]  _SUCCESS
│   └── [   8]  ._SUCCESS.crc
├── [4.0K]  iter-0008
│   ├── [ 10M]  part-r-00000.avro
│   ├── [ 84K]  .part-r-00000.avro.crc
│   ├── [ 11M]  part-r-00001.avro
│   ├── [ 84K]  .part-r-00001.avro.crc
│   ├── [ 10M]  part-r-00002.avro
│   ├── [ 82K]  .part-r-00002.avro.crc
│   ├── [ 11M]  part-r-00003.avro
│   ├── [ 85K]  .part-r-00003.avro.crc
│   ├── [   0]  _SUCCESS
│   └── [   8]  ._SUCCESS.crc
