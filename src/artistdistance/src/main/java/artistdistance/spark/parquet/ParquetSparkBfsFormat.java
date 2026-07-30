package artistdistance.spark.parquet;

import artistdistance.spark.SparkBfsFormat;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;

public final class ParquetSparkBfsFormat implements SparkBfsFormat {
  @Override
  public String name() {
    return "parquet";
  }

  @Override
  public Dataset<Row> readAdjacency(SparkSession spark, String input) {
    return spark.read().parquet(input);
  }

  @Override
  public void writeArtistDistances(Dataset<Row> distances, String output) {
    distances.write().mode(SaveMode.ErrorIfExists).parquet(output);
  }
}
