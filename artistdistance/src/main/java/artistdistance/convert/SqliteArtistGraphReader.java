package artistdistance.convert;

import artistdistance.schema.Adjacency;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public final class SqliteArtistGraphReader {
  @FunctionalInterface
  public interface EdgeHandler {
    void accept(String src, String dst) throws Exception;
  }

  private final String url;

  public SqliteArtistGraphReader(Path dbPath) {
    this.url = "jdbc:sqlite:" + dbPath.toAbsolutePath().toUri() + "?mode=ro&immutable=1";
  }

  public List<Adjacency> readAdjacency() throws Exception {
    Map<String, List<String>> graph = new TreeMap<>();
    try (Connection connection = connect()) {
      try (var statement = connection.createStatement();
           var rows = statement.executeQuery("select artist_id from artists order by artist_id")) {
        while (rows.next()) {
          graph.put(rows.getString(1), new ArrayList<>());
        }
      }
      try (var statement = connection.createStatement();
           var rows = statement.executeQuery("select target, similar from similarity order by target, similar")) {
        while (rows.next()) {
          graph.computeIfAbsent(rows.getString(1), key -> new ArrayList<>()).add(rows.getString(2));
        }
      }
    }
    List<Adjacency> adjacency = new ArrayList<>(graph.size());
    for (var entry : graph.entrySet()) {
      List<String> neighbors = entry.getValue();
      adjacency.add(new Adjacency(entry.getKey(), neighbors, neighbors.size()));
    }
    return adjacency;
  }

  public void forEachEdge(EdgeHandler handler) throws Exception {
    try (Connection connection = connect();
         var statement = connection.createStatement();
         var rows = statement.executeQuery("select target, similar from similarity order by target, similar")) {
      while (rows.next()) {
        handler.accept(rows.getString(1), rows.getString(2));
      }
    }
  }

  public long countEdges() throws SQLException {
    return count("select count(*) from similarity");
  }

  private long count(String sql) throws SQLException {
    try (Connection connection = connect();
         var statement = connection.createStatement();
         var rows = statement.executeQuery(sql)) {
      return rows.getLong(1);
    }
  }

  private Connection connect() throws SQLException {
    return DriverManager.getConnection(url);
  }
}
