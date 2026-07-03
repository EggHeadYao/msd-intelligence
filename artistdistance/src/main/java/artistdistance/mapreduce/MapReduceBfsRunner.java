package artistdistance.mapreduce;

import artistdistance.bfs.BfsPaths;
import artistdistance.schema.Adjacency;
import artistdistance.schema.ArtistDistance;
import artistdistance.schema.BfsVertex;
import org.apache.avro.mapreduce.AvroJob;
import org.apache.avro.mapreduce.AvroKeyInputFormat;
import org.apache.avro.mapreduce.AvroKeyOutputFormat;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public final class MapReduceBfsRunner {
  private static final String MAX_ITERATIONS = "artistdistance.bfs.max.iterations";
  private static final String REDUCERS = "artistdistance.bfs.reducers";

  public void runAvro(Configuration conf, String input, String sourceId, String output)
      throws Exception {
    conf.set(BfsInitMapper.SOURCE_ID, sourceId);
    long sourceFound = runInit(conf, input, BfsPaths.initial(output));
    if (sourceFound == 0) {
      throw new IllegalArgumentException("Source artist not found: " + sourceId);
    }
    int iteration = 0;
    while (iteration < conf.getInt(MAX_ITERATIONS, 1000)) {
      String in = BfsPaths.iteration(output, iteration);
      String out = BfsPaths.iteration(output, iteration + 1);
      long discovered = runIteration(conf, in, out, iteration + 1);
      iteration++;
      if (discovered == 0) {
        runFinal(conf, out, BfsPaths.finalOutput(output));
        return;
      }
    }
    throw new IllegalStateException("BFS reached max iterations without convergence");
  }

  private long runInit(Configuration conf, String input, String output) throws Exception {
    Job job = Job.getInstance(conf, "artist-distance-avro-bfs-init");
    job.setJarByClass(MapReduceBfsRunner.class);
    job.setMapperClass(BfsInitMapper.class);
    job.setNumReduceTasks(0);
    job.setInputFormatClass(AvroKeyInputFormat.class);
    job.setOutputFormatClass(AvroKeyOutputFormat.class);
    AvroJob.setInputKeySchema(job, Adjacency.getClassSchema());
    AvroJob.setOutputKeySchema(job, BfsVertex.getClassSchema());
    FileInputFormat.addInputPath(job, new Path(input));
    FileOutputFormat.setOutputPath(job, new Path(output));
    if (!job.waitForCompletion(true)) {
      throw new IllegalStateException("BFS init job failed");
    }
    return job.getCounters().findCounter(BfsInitMapper.Counter.SOURCE_FOUND).getValue();
  }

  private long runIteration(Configuration conf, String input, String output, int iteration)
      throws Exception {
    Job job = Job.getInstance(conf, "artist-distance-avro-bfs-iter-" + iteration);
    job.setJarByClass(MapReduceBfsRunner.class);
    job.setMapperClass(BfsIterationMapper.class);
    job.setReducerClass(BfsIterationReducer.class);
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(BfsMessage.class);
    job.setNumReduceTasks(Math.max(1, conf.getInt(REDUCERS, 4)));
    job.setInputFormatClass(AvroKeyInputFormat.class);
    job.setOutputFormatClass(AvroKeyOutputFormat.class);
    AvroJob.setInputKeySchema(job, BfsVertex.getClassSchema());
    AvroJob.setOutputKeySchema(job, BfsVertex.getClassSchema());
    FileInputFormat.addInputPath(job, new Path(input));
    FileOutputFormat.setOutputPath(job, new Path(output));
    if (!job.waitForCompletion(true)) {
      throw new IllegalStateException("BFS iteration job failed: " + iteration);
    }
    return job.getCounters().findCounter(BfsIterationReducer.Counter.DISCOVERED).getValue();
  }

  private void runFinal(Configuration conf, String input, String output) throws Exception {
    Job job = Job.getInstance(conf, "artist-distance-avro-bfs-final");
    job.setJarByClass(MapReduceBfsRunner.class);
    job.setMapperClass(BfsFinalMapper.class);
    job.setNumReduceTasks(0);
    job.setInputFormatClass(AvroKeyInputFormat.class);
    job.setOutputFormatClass(AvroKeyOutputFormat.class);
    AvroJob.setInputKeySchema(job, BfsVertex.getClassSchema());
    AvroJob.setOutputKeySchema(job, ArtistDistance.getClassSchema());
    FileInputFormat.addInputPath(job, new Path(input));
    FileOutputFormat.setOutputPath(job, new Path(output));
    if (!job.waitForCompletion(true)) {
      throw new IllegalStateException("BFS final job failed");
    }
  }
}
