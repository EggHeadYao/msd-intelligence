package artistdistance.mapreduce.parquet;

import artistdistance.mapreduce.BfsCounter;
import artistdistance.mapreduce.BfsIterationStep;
import artistdistance.mapreduce.BfsMessage;
import artistdistance.schema.BfsVertex;
import java.io.IOException;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public final class ParquetBfsIterationReducer extends Reducer<Text, BfsMessage, Void, BfsVertex> {
  @Override
  protected void reduce(Text key, Iterable<BfsMessage> values, Context context)
      throws IOException, InterruptedException {
    BfsIterationStep.Result result = BfsIterationStep.reduce(key.toString(), values);
    if (result.discovered()) {
      context.getCounter(BfsCounter.DISCOVERED).increment(1);
    }
    context.write(null, result.vertex());
  }
}
