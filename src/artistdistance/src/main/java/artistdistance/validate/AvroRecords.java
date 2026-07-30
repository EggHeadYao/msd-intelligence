package artistdistance.validate;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import org.apache.avro.file.DataFileReader;
import org.apache.avro.specific.SpecificData;
import org.apache.avro.specific.SpecificDatumReader;
import org.apache.avro.specific.SpecificRecordBase;

public final class AvroRecords {
  private AvroRecords() {}

  public static <T extends SpecificRecordBase> List<T> read(Path path, Class<T> type)
      throws IOException {
    List<T> rows = new ArrayList<>();
    for (Path file : avroFiles(path)) {
      try (DataFileReader<T> reader =
          new DataFileReader<>(file.toFile(), new SpecificDatumReader<>(type))) {
        for (T row : reader) {
          rows.add(SpecificData.get().deepCopy(row.getSchema(), row));
        }
      }
    }
    return rows;
  }

  private static List<Path> avroFiles(Path path) throws IOException {
    if (Files.isRegularFile(path)) {
      return List.of(path);
    }
    try (var files = Files.list(path)) {
      return files
          .filter(Files::isRegularFile)
          .filter(file -> file.getFileName().toString().endsWith(".avro"))
          .sorted(Comparator.comparing(Path::toString))
          .toList();
    }
  }
}
