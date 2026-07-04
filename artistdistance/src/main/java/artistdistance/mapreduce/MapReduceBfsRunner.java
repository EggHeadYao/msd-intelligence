package artistdistance.mapreduce;

import artistdistance.bfs.BfsPaths;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public final class MapReduceBfsRunner {
  public void run(
      Configuration conf, MapReduceBfsFormat format, String input, String sourceId, String output)
      throws Exception {
    conf.set(MapReduceBfsConfig.SOURCE_ID, sourceId);
    long sourceFound = runInit(conf, format, input, BfsPaths.initial(output));
    if (sourceFound == 0) {
      throw new IllegalArgumentException("Source artist not found: " + sourceId);
    }
    int iteration = 0;
    while (iteration < conf.getInt(MapReduceBfsConfig.MAX_ITERATIONS, 1000)) {
      String in = BfsPaths.iteration(output, iteration);
      String out = BfsPaths.iteration(output, iteration + 1);
      long discovered = runIteration(conf, format, in, out, iteration + 1);
      iteration++;
      if (discovered == 0) {
        runFinal(conf, format, out, BfsPaths.finalOutput(output));
        return;
      }
    }
    throw new IllegalStateException("BFS reached max iterations without convergence");
  }

  private long runInit(Configuration conf, MapReduceBfsFormat format, String input, String output)
      throws Exception {
    Job job = Job.getInstance(conf, "artist-distance-" + format.name() + "-bfs-init");
    job.setJarByClass(MapReduceBfsRunner.class);
    job.setNumReduceTasks(0);
    format.configureInitJob(job);
    FileInputFormat.addInputPath(job, new Path(input));
    FileOutputFormat.setOutputPath(job, new Path(output));
    if (!job.waitForCompletion(true)) {
      throw new IllegalStateException("BFS init job failed");
    }
    return job.getCounters().findCounter(BfsCounter.SOURCE_FOUND).getValue();
  }

  private long runIteration(
      Configuration conf, MapReduceBfsFormat format, String input, String output, int iteration)
      throws Exception {
    Job job = Job.getInstance(conf, "artist-distance-" + format.name() + "-bfs-iter-" + iteration);
    job.setJarByClass(MapReduceBfsRunner.class);
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(BfsMessage.class);
    job.setNumReduceTasks(Math.max(1, conf.getInt(MapReduceBfsConfig.REDUCERS, 4)));
    format.configureIterationJob(job);
    FileInputFormat.addInputPath(job, new Path(input));
    FileOutputFormat.setOutputPath(job, new Path(output));
    if (!job.waitForCompletion(true)) {
      throw new IllegalStateException("BFS iteration job failed: " + iteration);
    }
    return job.getCounters().findCounter(BfsCounter.DISCOVERED).getValue();
  }

  private void runFinal(Configuration conf, MapReduceBfsFormat format, String input, String output)
      throws Exception {
    Job job = Job.getInstance(conf, "artist-distance-" + format.name() + "-bfs-final");
    job.setJarByClass(MapReduceBfsRunner.class);
    job.setNumReduceTasks(0);
    format.configureFinalJob(job);
    FileInputFormat.addInputPath(job, new Path(input));
    FileOutputFormat.setOutputPath(job, new Path(output));
    if (!job.waitForCompletion(true)) {
      throw new IllegalStateException("BFS final job failed");
    }
  }
}
