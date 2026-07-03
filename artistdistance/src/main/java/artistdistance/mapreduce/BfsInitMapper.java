package artistdistance.mapreduce;

import artistdistance.bfs.BfsRules;
import artistdistance.schema.Adjacency;
import artistdistance.schema.BfsVertex;
import java.io.IOException;
import org.apache.avro.mapred.AvroKey;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Mapper;

public final class BfsInitMapper
    extends Mapper<AvroKey<Adjacency>, NullWritable, AvroKey<BfsVertex>, NullWritable> {
  public static final String SOURCE_ID = "artistdistance.bfs.source";

  public enum Counter {
    SOURCE_FOUND
  }

  private String sourceId;

  @Override
  protected void setup(Context context) {
    sourceId = context.getConfiguration().get(SOURCE_ID);
  }

  @Override
  protected void map(AvroKey<Adjacency> key, NullWritable value, Context context)
      throws IOException, InterruptedException {
    if (key.datum().getId().equals(sourceId)) {
      context.getCounter(Counter.SOURCE_FOUND).increment(1);
    }
    context.write(new AvroKey<>(BfsRules.initialVertex(key.datum(), sourceId)), NullWritable.get());
  }
}
