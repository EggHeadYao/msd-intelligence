package artistdistance.convert;

import artistdistance.schema.Adjacency;
import artistdistance.schema.Edge;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.apache.avro.specific.SpecificData;
import org.apache.hadoop.conf.Configuration;
import org.apache.parquet.avro.AvroParquetWriter;
import org.apache.parquet.hadoop.ParquetFileWriter;
import org.apache.parquet.hadoop.ParquetWriter;
import org.apache.parquet.hadoop.util.HadoopOutputFile;

public final class ParquetGraphWriter {
  public void write(Path outputDir, SqliteArtistGraphReader reader, List<Adjacency> adjacency) throws Exception {
    Files.createDirectories(outputDir);
    writeEdges(outputDir.resolve("edges.parquet"), reader);
    writeAdjacency(outputDir.resolve("adjacency.parquet"), adjacency);
  }

  private void writeEdges(Path file, SqliteArtistGraphReader reader) throws Exception {
    try (ParquetWriter<Edge> writer = writer(file, Edge.getClassSchema())) {
      reader.forEachEdge((src, dst) -> writer.write(new Edge(src, dst)));
    }
  }

  private void writeAdjacency(Path file, List<Adjacency> adjacency) throws Exception {
    try (ParquetWriter<Adjacency> writer = writer(file, Adjacency.getClassSchema())) {
      for (Adjacency row : adjacency) {
        writer.write(row);
      }
    }
  }

  private static <T> ParquetWriter<T> writer(Path file, org.apache.avro.Schema schema) throws Exception {
    Configuration conf = new Configuration();
    org.apache.hadoop.fs.Path path = new org.apache.hadoop.fs.Path(file.toUri());
    return AvroParquetWriter.<T>builder(HadoopOutputFile.fromPath(path, conf))
        .withSchema(schema)
        .withDataModel(SpecificData.get())
        .withConf(conf)
        .withWriteMode(ParquetFileWriter.Mode.OVERWRITE)
        .build();
  }
}
