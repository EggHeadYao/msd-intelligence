package artistdistance.spark;

import org.apache.spark.sql.SparkSession;

public final class SparkBfsSession {
  private SparkBfsSession() {}

  public static SparkSession create(String appName) {
    SparkSession.Builder builder = SparkSession.builder().appName(appName);
    String master = System.getProperty(SparkBfsConfig.MASTER, "local[4]");
    if (!master.isBlank()) {
      builder.master(master);
    }
    if (System.getProperty(SparkBfsConfig.SHUFFLE_PARTITIONS) == null) {
      builder.config(SparkBfsConfig.SHUFFLE_PARTITIONS, "32");
    }
    SparkSession spark = builder.getOrCreate();
    spark.sparkContext().setLogLevel("WARN");
    return spark;
  }
}
