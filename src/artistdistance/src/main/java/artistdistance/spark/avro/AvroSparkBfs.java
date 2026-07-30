package artistdistance.spark.avro;

import artistdistance.spark.SparkBfsRunner;
import artistdistance.spark.SparkBfsSession;
import org.apache.spark.sql.SparkSession;

public final class AvroSparkBfs {
  private AvroSparkBfs() {}

  public static void main(String[] args) {
    if (args.length != 3) {
      System.err.println("Usage: AvroSparkBfs <adjacency.avro> <source_artist_id> <output-dir>");
      System.exit(1);
    }
    SparkSession spark = SparkBfsSession.create("artist-distance-avro-spark-bfs");
    try {
      new SparkBfsRunner().run(spark, new AvroSparkBfsFormat(), args[0], args[1], args[2]);
    } finally {
      spark.stop();
    }
  }
}
