package artistdistance.mapreduce.avro;

import artistdistance.mapreduce.MapReduceBfsRunner;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public final class AvroMapReduceBfs extends Configured implements Tool {
  @Override
  public int run(String[] args) throws Exception {
    if (args.length != 3) {
      System.err.println("Usage: AvroMapReduceBfs <adjacency.avro> <source_artist_id> <output-dir>");
      return 1;
    }
    new MapReduceBfsRunner().run(getConf(), new AvroBfsFormat(), args[0], args[1], args[2]);
    return 0;
  }

  public static void main(String[] args) throws Exception {
    int exitCode = ToolRunner.run(new AvroMapReduceBfs(), args);
    System.exit(exitCode);
  }
}
