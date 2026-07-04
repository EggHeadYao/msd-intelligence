package artistdistance.mapreduce.parquet;

import artistdistance.bfs.BfsRules;
import artistdistance.mapreduce.BfsCounter;
import artistdistance.mapreduce.MapReduceBfsConfig;
import artistdistance.schema.Adjacency;
import artistdistance.schema.BfsVertex;
import java.io.IOException;
import org.apache.hadoop.mapreduce.Mapper;

public final class ParquetBfsInitMapper extends Mapper<Void, Adjacency, Void, BfsVertex> {
  private String sourceId;

  @Override
  protected void setup(Context context) {
    sourceId = context.getConfiguration().get(MapReduceBfsConfig.SOURCE_ID);
  }

  @Override
  protected void map(Void key, Adjacency value, Context context)
      throws IOException, InterruptedException {
    if (value.getId().equals(sourceId)) {
      context.getCounter(BfsCounter.SOURCE_FOUND).increment(1);
    }
    context.write(null, BfsRules.initialVertex(value, sourceId));
  }
}
