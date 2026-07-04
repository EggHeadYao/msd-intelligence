package artistdistance.mapreduce.avro;

import artistdistance.bfs.BfsRules;
import artistdistance.mapreduce.BfsCounter;
import artistdistance.mapreduce.MapReduceBfsConfig;
import artistdistance.schema.Adjacency;
import artistdistance.schema.BfsVertex;
import java.io.IOException;
import org.apache.avro.mapred.AvroKey;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Mapper;

public final class AvroBfsInitMapper
    extends Mapper<AvroKey<Adjacency>, NullWritable, AvroKey<BfsVertex>, NullWritable> {
  private String sourceId;

  @Override
  protected void setup(Context context) {
    sourceId = context.getConfiguration().get(MapReduceBfsConfig.SOURCE_ID);
  }

  @Override
  protected void map(AvroKey<Adjacency> key, NullWritable value, Context context)
      throws IOException, InterruptedException {
    Adjacency row = key.datum();
    if (row.getId().equals(sourceId)) {
      context.getCounter(BfsCounter.SOURCE_FOUND).increment(1);
    }
    context.write(new AvroKey<>(BfsRules.initialVertex(row, sourceId)), NullWritable.get());
  }
}
