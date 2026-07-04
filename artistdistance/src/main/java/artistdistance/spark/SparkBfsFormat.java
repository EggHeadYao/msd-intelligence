package artistdistance.spark;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public interface SparkBfsFormat {
  String name();

  Dataset<Row> readAdjacency(SparkSession spark, String input);

  void writeArtistDistances(Dataset<Row> distances, String output);
}
