package artistdistance.mapreduce.avro;

import artistdistance.mapreduce.MapReduceBfsFormat;
import artistdistance.schema.Adjacency;
import artistdistance.schema.ArtistDistance;
import artistdistance.schema.BfsVertex;
import java.io.IOException;
import org.apache.avro.mapreduce.AvroJob;
import org.apache.avro.mapreduce.AvroKeyInputFormat;
import org.apache.avro.mapreduce.AvroKeyOutputFormat;
import org.apache.hadoop.mapreduce.Job;

public final class AvroBfsFormat implements MapReduceBfsFormat {
  @Override
  public String name() {
    return "avro";
  }

  @Override
  public void configureInitJob(Job job) throws IOException {
    job.setMapperClass(AvroBfsInitMapper.class);
    job.setInputFormatClass(AvroKeyInputFormat.class);
    job.setOutputFormatClass(AvroKeyOutputFormat.class);
    AvroJob.setInputKeySchema(job, Adjacency.getClassSchema());
    AvroJob.setOutputKeySchema(job, BfsVertex.getClassSchema());
  }

  @Override
  public void configureIterationJob(Job job) throws IOException {
    job.setMapperClass(AvroBfsIterationMapper.class);
    job.setReducerClass(AvroBfsIterationReducer.class);
    job.setInputFormatClass(AvroKeyInputFormat.class);
    job.setOutputFormatClass(AvroKeyOutputFormat.class);
    AvroJob.setInputKeySchema(job, BfsVertex.getClassSchema());
    AvroJob.setOutputKeySchema(job, BfsVertex.getClassSchema());
  }

  @Override
  public void configureFinalJob(Job job) throws IOException {
    job.setMapperClass(AvroBfsFinalMapper.class);
    job.setInputFormatClass(AvroKeyInputFormat.class);
    job.setOutputFormatClass(AvroKeyOutputFormat.class);
    AvroJob.setInputKeySchema(job, BfsVertex.getClassSchema());
    AvroJob.setOutputKeySchema(job, ArtistDistance.getClassSchema());
  }
}
