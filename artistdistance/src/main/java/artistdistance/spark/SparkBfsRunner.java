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

  private static Dataset<Row> checkpoint(Dataset<Row> rows) {
    return rows.localCheckpoint(true);
  }
}
