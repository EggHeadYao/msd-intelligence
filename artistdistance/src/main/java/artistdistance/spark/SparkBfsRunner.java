package artistdistance.spark;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.explode;
import static org.apache.spark.sql.functions.lit;
import static org.apache.spark.sql.functions.min;
import static org.apache.spark.sql.functions.struct;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.storage.StorageLevel;

public final class SparkBfsRunner {
  public void run(
      SparkSession spark, SparkBfsFormat format, String input, String sourceId, String output) {
    Dataset<Row> adjacency =
        format
            .readAdjacency(spark, input)
            .select(col("id").alias("src"), col("neighbors"))
            .persist(StorageLevel.MEMORY_AND_DISK());
    if (adjacency.filter(col("src").equalTo(lit(sourceId))).limit(1).count() == 0) {
      throw new IllegalArgumentException("Source artist not found: " + sourceId);
    }

    Dataset<Row> frontier = checkpoint(initialFrontier(adjacency, sourceId));
    Dataset<Row> visited = frontier;
    long frontierCount = frontier.count();
    int iteration = 0;
    int maxIterations = Integer.parseInt(spark.conf().get(SparkBfsConfig.MAX_ITERATIONS, "1000"));

    while (frontierCount > 0 && iteration < maxIterations) {
      Dataset<Row> nextFrontier = checkpoint(bestUnvisitedCandidates(adjacency, visited, frontier));
      frontierCount = nextFrontier.count();
      if (frontierCount == 0) {
        nextFrontier.unpersist();
        break;
      }

      Dataset<Row> previousVisited = visited;
      visited = checkpoint(visited.unionByName(nextFrontier));
      previousVisited.unpersist();
      frontier.unpersist();
      frontier = nextFrontier;
      iteration++;
    }

    if (iteration >= maxIterations && frontierCount > 0) {
      throw new IllegalStateException("BFS reached max iterations without convergence");
    }

    format.writeArtistDistances(finalDistances(adjacency, visited), output);
    adjacency.unpersist();
    visited.unpersist();
    frontier.unpersist();
  }

  private static Dataset<Row> initialFrontier(Dataset<Row> adjacency, String sourceId) {
    return adjacency
        .filter(col("src").equalTo(lit(sourceId)))
        .select(
            col("src").alias("id"),
            lit(0).cast("int").alias("distance"),
            lit(null).cast("string").alias("parent"));
  }

  private static Dataset<Row> bestUnvisitedCandidates(
      Dataset<Row> adjacency, Dataset<Row> visited, Dataset<Row> frontier) {
    Dataset<Row> f =
        frontier
            .select(col("id").alias("frontier_id"), col("distance").alias("frontier_distance"))
            .alias("f");
    Dataset<Row> a = adjacency.alias("a");
    Dataset<Row> candidates =
        f.join(a, f.col("frontier_id").equalTo(a.col("src")))
            .select(
                explode(a.col("neighbors")).alias("id"),
                f.col("frontier_distance").plus(1).cast("int").alias("distance"),
                f.col("frontier_id").alias("parent"));

    Dataset<Row> v = visited.select(col("id").alias("visited_id")).alias("v");
    Dataset<Row> unvisited =
        candidates
            .alias("c")
            .join(v, col("c.id").equalTo(col("v.visited_id")), "left_anti")
            .select(col("c.id"), col("c.distance"), col("c.parent"));

    return unvisited
        .groupBy(col("id"))
        .agg(min(struct(col("distance"), col("parent"))).alias("best"))
        .select(
            col("id"),
            col("best.distance").cast("int").alias("distance"),
            col("best.parent").cast("string").alias("parent"));
  }

  private static Dataset<Row> finalDistances(Dataset<Row> adjacency, Dataset<Row> visited) {
    Dataset<Row> ids = adjacency.select(col("src").alias("artist_id")).alias("a");
    Dataset<Row> v =
        visited
            .select(col("id").alias("visited_id"), col("distance"), col("parent"))
            .alias("v");
    return ids.join(v, ids.col("artist_id").equalTo(v.col("visited_id")), "left_outer")
        .select(
            ids.col("artist_id").alias("id"),
            v.col("distance").cast("int").alias("distance"),
            v.col("parent").cast("string").alias("parent"));
  }

  private static Dataset<Row> checkpoint(Dataset<Row> rows) {
    return rows.localCheckpoint(true);
  }
}
