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
}
