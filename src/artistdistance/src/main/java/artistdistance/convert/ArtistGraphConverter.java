package artistdistance.convert;

import artistdistance.schema.Adjacency;
import java.nio.file.Path;
import java.util.List;

public final class ArtistGraphConverter {
  private ArtistGraphConverter() {}

  public static void main(String[] args) throws Exception {
    if (args.length != 2) {
      System.err.println("Usage: ArtistGraphConverter <artist_similarity.db> <output-dir>");
      System.exit(1);
    }

    SqliteArtistGraphReader reader = new SqliteArtistGraphReader(Path.of(args[0]));
    List<Adjacency> adjacency = reader.readAdjacency();
    Path outputDir = Path.of(args[1]);
    new AvroGraphWriter().write(outputDir.resolve("avro"), reader, adjacency);
    new ParquetGraphWriter().write(outputDir.resolve("parquet"), reader, adjacency);

    long edges = reader.countEdges();
    long degreeSum = adjacency.stream().mapToLong(Adjacency::getDegree).sum();
    if (degreeSum != edges) {
      throw new IllegalStateException("Degree sum " + degreeSum + " differs from edge count " + edges);
    }
    System.out.println("artists=" + adjacency.size());
    System.out.println("edges=" + edges);
    System.out.println("degree_sum=" + degreeSum);
  }
}
