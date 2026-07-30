package artistdistance.validate;

import artistdistance.schema.Adjacency;
import artistdistance.schema.ArtistDistance;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import org.apache.avro.generic.GenericRecord;
import org.apache.avro.specific.SpecificData;
import org.apache.avro.specific.SpecificRecordBase;
import org.apache.parquet.avro.AvroParquetReader;
import org.apache.parquet.hadoop.ParquetReader;

public final class ParquetRecords {
  private ParquetRecords() {}

  public static List<Adjacency> readAdjacency(Path path) throws IOException {
    List<Adjacency> rows = new ArrayList<>();
    for (Object row : readObjects(path)) {
      rows.add(toAdjacency(row));
    }
    return rows;
  }

  public static List<ArtistDistance> readArtistDistances(Path path) throws IOException {
    List<ArtistDistance> rows = new ArrayList<>();
    for (Object row : readObjects(path)) {
      rows.add(toArtistDistance(row));
    }
    return rows;
  }

  private static List<Object> readObjects(Path path) throws IOException {
    List<Object> rows = new ArrayList<>();
    for (Path file : parquetFiles(path)) {
      try (ParquetReader<Object> reader =
          AvroParquetReader.builder(new org.apache.hadoop.fs.Path(file.toUri()))
              .withDataModel(SpecificData.get())
              .build()) {
        Object row;
        while ((row = reader.read()) != null) {
          rows.add(copy(row));
        }
      }
    }
    return rows;
  }

  private static Object copy(Object row) {
    if (row instanceof SpecificRecordBase specific) {
      return SpecificData.get().deepCopy(specific.getSchema(), specific);
    }
    return row;
  }

  private static Adjacency toAdjacency(Object row) {
    if (row instanceof Adjacency adjacency) {
      return adjacency;
    }
    GenericRecord record = (GenericRecord) row;
    List<String> neighbors = strings(record.get("neighbors"));
    Object degree = record.hasField("degree") ? record.get("degree") : neighbors.size();
    return new Adjacency(string(record.get("id")), neighbors, ((Number) degree).intValue());
  }

  private static ArtistDistance toArtistDistance(Object row) {
    if (row instanceof ArtistDistance distance) {
      return distance;
    }
    GenericRecord record = (GenericRecord) row;
    Object distance = record.get("distance");
    return new ArtistDistance(
        string(record.get("id")),
        distance == null ? null : ((Number) distance).intValue(),
        nullableString(record.get("parent")));
  }

  private static List<String> strings(Object value) {
    List<String> rows = new ArrayList<>();
    for (Object row : (Iterable<?>) value) {
      rows.add(string(row));
    }
    return rows;
  }

  private static String string(Object value) {
    return value.toString();
  }

  private static String nullableString(Object value) {
    return value == null ? null : value.toString();
  }

  private static List<Path> parquetFiles(Path path) throws IOException {
    if (Files.isRegularFile(path)) {
      return List.of(path);
    }
    try (var files = Files.list(path)) {
      return files
          .filter(Files::isRegularFile)
          .filter(file -> file.getFileName().toString().endsWith(".parquet"))
          .sorted(Comparator.comparing(Path::toString))
          .toList();
    }
  }
}
