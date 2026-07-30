package artistdistance.spark;

public final class SparkBfsConfig {
  public static final String MAX_ITERATIONS = "artistdistance.spark.bfs.max.iterations";
  public static final String MASTER = "spark.master";
  public static final String SHUFFLE_PARTITIONS = "spark.sql.shuffle.partitions";

  private SparkBfsConfig() {}
}
