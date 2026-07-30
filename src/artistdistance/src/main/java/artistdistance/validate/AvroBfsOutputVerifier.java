package artistdistance.validate;

import artistdistance.schema.Adjacency;
import artistdistance.schema.ArtistDistance;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;
import java.util.TreeSet;

public final class AvroBfsOutputVerifier {
  private AvroBfsOutputVerifier() {}

  public static void main(String[] args) throws Exception {
    if (args.length != 3) {
      System.err.println("Usage: AvroBfsOutputVerifier <adjacency.avro> <final-dir> <source_id>");
      System.exit(1);
    }
    ReferenceBfs.Result expected =
        ReferenceBfs.compute(AvroRecords.read(Path.of(args[0]), Adjacency.class), args[2]);
    Map<String, ArtistDistance> actual = readActual(Path.of(args[1]));
    var errors = new ArrayList<String>();
    long reachable = 0;
    int maxDistance = -1;
    for (String id : new TreeSet<>(expected.ids())) {
      ArtistDistance row = actual.remove(id);
      if (row == null) {
        errors.add("missing id=" + id);
        continue;
      }
      Integer distance = expected.distances().get(id);
      String parent = expected.parents().get(id);
      if (!Objects.equals(distance, row.getDistance()) || !Objects.equals(parent, row.getParent())) {
        errors.add("mismatch id=" + id + " expected=(" + distance + "," + parent
            + ") actual=(" + row.getDistance() + "," + row.getParent() + ")");
      }
      if (row.getDistance() != null) {
        reachable++;
        maxDistance = Math.max(maxDistance, row.getDistance());
      }
    }
    for (String id : actual.keySet()) {
      errors.add("extra id=" + id);
    }
    printSummary(expected.ids().size(), reachable, maxDistance, errors);
  }

  private static Map<String, ArtistDistance> readActual(Path path) throws Exception {
    Map<String, ArtistDistance> rows = new TreeMap<>();
    for (ArtistDistance row : AvroRecords.read(path, ArtistDistance.class)) {
      if (rows.put(row.getId(), row) != null) {
        throw new IllegalArgumentException("Duplicate output id: " + row.getId());
      }
    }
    return rows;
  }

  private static void printSummary(long total, long reachable, int maxDistance, ArrayList<String> errors) {
    System.out.println("expected_total=" + total);
    System.out.println("reachable=" + reachable);
    System.out.println("unreachable=" + (total - reachable));
    System.out.println("max_distance=" + maxDistance);
    System.out.println("mismatches=" + errors.size());
    errors.stream().limit(20).forEach(System.out::println);
    if (!errors.isEmpty()) {
      System.exit(2);
    }
  }
}
