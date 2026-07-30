package artistdistance.mapreduce.parquet;

import artistdistance.mapreduce.MapReduceBfsFormat;
import artistdistance.schema.Adjacency;
import artistdistance.schema.ArtistDistance;
import artistdistance.schema.BfsVertex;
import java.io.IOException;
import org.apache.hadoop.mapreduce.Job;
import org.apache.parquet.avro.AvroParquetInputFormat;
import org.apache.parquet.avro.AvroParquetOutputFormat;
import org.apache.parquet.avro.SpecificDataSupplier;

public final class ParquetBfsFormat implements MapReduceBfsFormat {
  @Override
  public String name() {
    return "parquet";
  }

  @Override
  public void configureInitJob(Job job) throws IOException {
    job.setMapperClass(ParquetBfsInitMapper.class);
    setInput(job, Adjacency.getClassSchema());
    setOutput(job, BfsVertex.getClassSchema(), BfsVertex.class);
  }

  @Override
  public void configureIterationJob(Job job) throws IOException {
    job.setMapperClass(ParquetBfsIterationMapper.class);
    job.setReducerClass(ParquetBfsIterationReducer.class);
    setInput(job, BfsVertex.getClassSchema());
    setOutput(job, BfsVertex.getClassSchema(), BfsVertex.class);
  }

  @Override
  public void configureFinalJob(Job job) throws IOException {
    job.setMapperClass(ParquetBfsFinalMapper.class);
    setInput(job, BfsVertex.getClassSchema());
    setOutput(job, ArtistDistance.getClassSchema(), ArtistDistance.class);
  }

  private static void setInput(Job job, org.apache.avro.Schema schema) {
    job.setInputFormatClass(AvroParquetInputFormat.class);
    AvroParquetInputFormat.setAvroReadSchema(job, schema);
    AvroParquetInputFormat.setAvroDataSupplier(job, SpecificDataSupplier.class);
  }

  private static void setOutput(
      Job job, org.apache.avro.Schema schema, Class<?> outputValueClass) {
    job.setOutputFormatClass(AvroParquetOutputFormat.class);
    job.setOutputKeyClass(Void.class);
    job.setOutputValueClass(outputValueClass);
    AvroParquetOutputFormat.setSchema(job, schema);
    AvroParquetOutputFormat.setAvroDataSupplier(job, SpecificDataSupplier.class);
  }
}
