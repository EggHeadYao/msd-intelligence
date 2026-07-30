package artistdistance.spark.parquet;

import artistdistance.spark.SparkBfsRunner;
import artistdistance.spark.SparkBfsSession;
import org.apache.spark.sql.SparkSession;

public final class ParquetSparkBfs {
  private ParquetSparkBfs() {}

  public static void main(String[] args) {
    if (args.length != 3) {
      System.err.println("Usage: ParquetSparkBfs <adjacency.parquet> <source_artist_id> <output-dir>");
      System.exit(1);
    }
    SparkSession spark = SparkBfsSession.create("artist-distance-parquet-spark-bfs");
    try {
      new SparkBfsRunner().run(spark, new ParquetSparkBfsFormat(), args[0], args[1], args[2]);
    } finally {
      spark.stop();
    }
  }
}
