package artistdistance.convert;

import artistdistance.schema.Adjacency;
import artistdistance.schema.Edge;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.apache.avro.file.DataFileWriter;
import org.apache.avro.specific.SpecificDatumWriter;

public final class AvroGraphWriter {
  public void write(Path outputDir, SqliteArtistGraphReader reader, List<Adjacency> adjacency) throws Exception {
    Files.createDirectories(outputDir);
    writeEdges(outputDir.resolve("edges.avro"), reader);
    writeAdjacency(outputDir.resolve("adjacency.avro"), adjacency);
  }

  private void writeEdges(Path file, SqliteArtistGraphReader reader) throws Exception {
    try (DataFileWriter<Edge> writer = new DataFileWriter<>(new SpecificDatumWriter<>(Edge.class))) {
      writer.create(Edge.getClassSchema(), file.toFile());
      reader.forEachEdge((src, dst) -> writer.append(new Edge(src, dst)));
    }
  }

  private void writeAdjacency(Path file, List<Adjacency> adjacency) throws Exception {
    try (DataFileWriter<Adjacency> writer = new DataFileWriter<>(new SpecificDatumWriter<>(Adjacency.class))) {
      writer.create(Adjacency.getClassSchema(), file.toFile());
      for (Adjacency row : adjacency) {
        writer.append(row);
      }
    }
  }
}
