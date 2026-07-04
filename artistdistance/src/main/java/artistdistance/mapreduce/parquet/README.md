# Parquet MapReduce BFS

- `ParquetMapReduceBfs`: CLI entry point for running BFS with Parquet input and output.
- `ParquetBfsFormat`: Configures Parquet input/output formats, schemas, mappers, and reducers.
- `ParquetBfsInitMapper`: Converts `Adjacency` records into initial `BfsVertex` records.
- `ParquetBfsIterationMapper`: Emits each vertex and expands frontier vertices to candidate messages.
- `ParquetBfsIterationReducer`: Merges vertex and candidate messages for one BFS iteration.
- `ParquetBfsFinalMapper`: Converts final `BfsVertex` records into `ArtistDistance` records.

Sample usage: `ParquetMapReduceBfs <adjacency.parquet> <source_artist_id> <output-dir>`

```bash
mvn exec:java -Dexec.mainClass=artistdistance.mapreduce.parquet.ParquetMapReduceBfs -Dexec.args="-Dmapreduce.framework.name=local ../data/artistdistance-output/parquet/adjacency.parquet ARGUACZ1187FB3F35C ../data/artistdistance-bfs-parquet-mapreduce"
```

Sample output:

```
[4.0K]  data/artistdistance-bfs-parquet-mapreduce/
├── [4.0K]  final
│   ├── [ 558]  _common_metadata
│   ├── [  16]  ._common_metadata.crc
│   ├── [2.2K]  _metadata
│   ├── [  28]  ._metadata.crc
│   ├── [380K]  part-m-00000.parquet
│   ├── [3.0K]  .part-m-00000.parquet.crc
│   ├── [380K]  part-m-00001.parquet
│   ├── [3.0K]  .part-m-00001.parquet.crc
│   ├── [378K]  part-m-00002.parquet
│   ├── [3.0K]  .part-m-00002.parquet.crc
│   ├── [372K]  part-m-00003.parquet
│   ├── [2.9K]  .part-m-00003.parquet.crc
│   ├── [   0]  _SUCCESS
│   └── [   8]  ._SUCCESS.crc
├── [4.0K]  iter-0000
│   ├── [ 911]  _common_metadata
│   ├── [  16]  ._common_metadata.crc
│   ├── [1.7K]  _metadata
│   ├── [  24]  ._metadata.crc
│   ├── [6.3M]  part-m-00000.parquet
│   ├── [ 50K]  .part-m-00000.parquet.crc
│   ├── [   0]  _SUCCESS
│   └── [   8]  ._SUCCESS.crc
├── [4.0K]  iter-0001
│   ├── [ 911]  _common_metadata
│   ├── [  16]  ._common_metadata.crc
│   ├── [4.3K]  _metadata
│   ├── [  44]  ._metadata.crc
│   ├── [2.2M]  part-r-00000.parquet
│   ├── [ 17K]  .part-r-00000.parquet.crc
│   ├── [2.2M]  part-r-00001.parquet
│   ├── [ 18K]  .part-r-00001.parquet.crc
│   ├── [2.2M]  part-r-00002.parquet
│   ├── [ 17K]  .part-r-00002.parquet.crc
│   ├── [2.2M]  part-r-00003.parquet
│   ├── [ 18K]  .part-r-00003.parquet.crc
│   ├── [   0]  _SUCCESS
│   └── [   8]  ._SUCCESS.crc
├── [4.0K]  iter-0002
│   ├── [ 911]  _common_metadata
│   ├── [  16]  ._common_metadata.crc
│   ├── [4.2K]  _metadata
│   ├── [  44]  ._metadata.crc
│   ├── [2.2M]  part-r-00000.parquet
│   ├── [ 17K]  .part-r-00000.parquet.crc
│   ├── [2.2M]  part-r-00001.parquet
│   ├── [ 18K]  .part-r-00001.parquet.crc
│   ├── [2.2M]  part-r-00002.parquet
│   ├── [ 17K]  .part-r-00002.parquet.crc
│   ├── [2.2M]  part-r-00003.parquet
│   ├── [ 18K]  .part-r-00003.parquet.crc
│   ├── [   0]  _SUCCESS
│   └── [   8]  ._SUCCESS.crc
├── [4.0K]  iter-0003
│   ├── [ 911]  _common_metadata
│   ├── [  16]  ._common_metadata.crc
│   ├── [4.2K]  _metadata
│   ├── [  44]  ._metadata.crc
│   ├── [2.2M]  part-r-00000.parquet
│   ├── [ 18K]  .part-r-00000.parquet.crc
│   ├── [2.2M]  part-r-00001.parquet
│   ├── [ 18K]  .part-r-00001.parquet.crc
│   ├── [2.2M]  part-r-00002.parquet
│   ├── [ 17K]  .part-r-00002.parquet.crc
│   ├── [2.2M]  part-r-00003.parquet
│   ├── [ 18K]  .part-r-00003.parquet.crc
│   ├── [   0]  _SUCCESS
│   └── [   8]  ._SUCCESS.crc
├── [4.0K]  iter-0004
│   ├── [ 911]  _common_metadata
│   ├── [  16]  ._common_metadata.crc
│   ├── [4.2K]  _metadata
│   ├── [  44]  ._metadata.crc
│   ├── [2.3M]  part-r-00000.parquet
│   ├── [ 18K]  .part-r-00000.parquet.crc
│   ├── [2.3M]  part-r-00001.parquet
│   ├── [ 18K]  .part-r-00001.parquet.crc
│   ├── [2.2M]  part-r-00002.parquet
│   ├── [ 18K]  .part-r-00002.parquet.crc
│   ├── [2.3M]  part-r-00003.parquet
│   ├── [ 18K]  .part-r-00003.parquet.crc
│   ├── [   0]  _SUCCESS
│   └── [   8]  ._SUCCESS.crc
├── [4.0K]  iter-0005
│   ├── [ 911]  _common_metadata
│   ├── [  16]  ._common_metadata.crc
│   ├── [4.2K]  _metadata
│   ├── [  44]  ._metadata.crc
│   ├── [2.3M]  part-r-00000.parquet
│   ├── [ 18K]  .part-r-00000.parquet.crc
│   ├── [2.3M]  part-r-00001.parquet
│   ├── [ 18K]  .part-r-00001.parquet.crc
│   ├── [2.3M]  part-r-00002.parquet
│   ├── [ 18K]  .part-r-00002.parquet.crc
│   ├── [2.3M]  part-r-00003.parquet
│   ├── [ 19K]  .part-r-00003.parquet.crc
│   ├── [   0]  _SUCCESS
│   └── [   8]  ._SUCCESS.crc
├── [4.0K]  iter-0006
│   ├── [ 911]  _common_metadata
│   ├── [  16]  ._common_metadata.crc
│   ├── [4.2K]  _metadata
│   ├── [  44]  ._metadata.crc
│   ├── [2.3M]  part-r-00000.parquet
│   ├── [ 19K]  .part-r-00000.parquet.crc
│   ├── [2.3M]  part-r-00001.parquet
│   ├── [ 19K]  .part-r-00001.parquet.crc
│   ├── [2.3M]  part-r-00002.parquet
│   ├── [ 18K]  .part-r-00002.parquet.crc
│   ├── [2.3M]  part-r-00003.parquet
│   ├── [ 19K]  .part-r-00003.parquet.crc
│   ├── [   0]  _SUCCESS
│   └── [   8]  ._SUCCESS.crc
├── [4.0K]  iter-0007
│   ├── [ 911]  _common_metadata
│   ├── [  16]  ._common_metadata.crc
│   ├── [4.2K]  _metadata
│   ├── [  44]  ._metadata.crc
│   ├── [2.3M]  part-r-00000.parquet
│   ├── [ 19K]  .part-r-00000.parquet.crc
│   ├── [2.3M]  part-r-00001.parquet
│   ├── [ 19K]  .part-r-00001.parquet.crc
│   ├── [2.3M]  part-r-00002.parquet
│   ├── [ 18K]  .part-r-00002.parquet.crc
│   ├── [2.3M]  part-r-00003.parquet
│   ├── [ 19K]  .part-r-00003.parquet.crc
│   ├── [   0]  _SUCCESS
│   └── [   8]  ._SUCCESS.crc
├── [4.0K]  iter-0008
│   ├── [ 911]  _common_metadata
│   ├── [  16]  ._common_metadata.crc
│   ├── [4.2K]  _metadata
│   ├── [  44]  ._metadata.crc
│   ├── [2.3M]  part-r-00000.parquet
│   ├── [ 19K]  .part-r-00000.parquet.crc
│   ├── [2.3M]  part-r-00001.parquet
│   ├── [ 19K]  .part-r-00001.parquet.crc
│   ├── [2.3M]  part-r-00002.parquet
│   ├── [ 18K]  .part-r-00002.parquet.crc
│   ├── [2.3M]  part-r-00003.parquet
│   ├── [ 19K]  .part-r-00003.parquet.crc
│   ├── [   0]  _SUCCESS
│   └── [   8]  ._SUCCESS.crc
├── [4.0K]  iter-0009
│   ├── [ 911]  _common_metadata
│   ├── [  16]  ._common_metadata.crc
│   ├── [4.2K]  _metadata
│   ├── [  44]  ._metadata.crc
│   ├── [2.3M]  part-r-00000.parquet
│   ├── [ 19K]  .part-r-00000.parquet.crc
│   ├── [2.3M]  part-r-00001.parquet
│   ├── [ 19K]  .part-r-00001.parquet.crc
│   ├── [2.3M]  part-r-00002.parquet
│   ├── [ 18K]  .part-r-00002.parquet.crc
│   ├── [2.3M]  part-r-00003.parquet
│   ├── [ 19K]  .part-r-00003.parquet.crc
│   ├── [   0]  _SUCCESS
│   └── [   8]  ._SUCCESS.crc
├── [4.0K]  iter-0010
│   ├── [ 911]  _common_metadata
│   ├── [  16]  ._common_metadata.crc
│   ├── [4.2K]  _metadata
│   ├── [  44]  ._metadata.crc
│   ├── [2.3M]  part-r-00000.parquet
│   ├── [ 19K]  .part-r-00000.parquet.crc
│   ├── [2.3M]  part-r-00001.parquet
│   ├── [ 19K]  .part-r-00001.parquet.crc
│   ├── [2.3M]  part-r-00002.parquet
│   ├── [ 18K]  .part-r-00002.parquet.crc
│   ├── [2.3M]  part-r-00003.parquet
│   ├── [ 19K]  .part-r-00003.parquet.crc
│   ├── [   0]  _SUCCESS
│   └── [   8]  ._SUCCESS.crc
└── [4.0K]  iter-0011
    ├── [ 911]  _common_metadata
    ├── [  16]  ._common_metadata.crc
    ├── [4.2K]  _metadata
    ├── [  44]  ._metadata.crc
    ├── [2.3M]  part-r-00000.parquet
    ├── [ 19K]  .part-r-00000.parquet.crc
    ├── [2.3M]  part-r-00001.parquet
    ├── [ 19K]  .part-r-00001.parquet.crc
    ├── [2.3M]  part-r-00002.parquet
    ├── [ 18K]  .part-r-00002.parquet.crc
    ├── [2.3M]  part-r-00003.parquet
    ├── [ 19K]  .part-r-00003.parquet.crc
    ├── [   0]  _SUCCESS
    └── [   8]  ._SUCCESS.crc
```
