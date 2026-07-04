package artistdistance.spark.avro;

import artistdistance.spark.SparkBfsFormat;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.SaveMode;

public final class AvroSparkBfsFormat implements SparkBfsFormat {
  @Override
  public String name() {
    return "avro";
  }

  @Override
  public Dataset<Row> readAdjacency(SparkSession spark, String input) {
    return spark.read().format("avro").load(input);
  }

  @Override
  public void writeArtistDistances(Dataset<Row> distances, String output) {
    distances
        .write()
        .format("avro")
        .option("recordName", "ArtistDistance")
        .option("recordNamespace", "artistdistance.schema")
        .mode(SaveMode.ErrorIfExists)
        .save(output);
  }
}
