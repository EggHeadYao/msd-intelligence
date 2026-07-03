# Convert

- `ArtistGraphConverter.java`: CLI entry point that runs the full SQLite to Avro and Parquet conversion.
- `SqliteArtistGraphReader.java`: Reads directed artist and similarity records from `artist_similarity.db`.
- `AvroGraphWriter.java`: Writes edge and adjacency records to Avro files.
- `ParquetGraphWriter.java`: Writes edge and adjacency records to Parquet files.

Sample usage: `ArtistGraphConverter <artist_similarity.db> <output-dir>`.

```bash
mvn exec:java -Dexec.mainClass=artistdistance.convert.ArtistGraphConverter -Dexec.args="../../msd/AdditionalFiles/artist_similarity.db ../data/artistdistance-output"
```

Sample output:

```
[4.0K]  data/
└── [4.0K]  artistdistance-output
    ├── [4.0K]  avro
    │   ├── [ 41M]  adjacency.avro
    │   └── [ 80M]  edges.avro
    └── [4.0K]  parquet
        ├── [6.3M]  adjacency.parquet
        ├── [ 50K]  .adjacency.parquet.crc
        ├── [6.2M]  edges.parquet
        └── [ 50K]  .edges.parquet.crc
```
