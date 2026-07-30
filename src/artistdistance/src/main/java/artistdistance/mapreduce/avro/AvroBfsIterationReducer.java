package artistdistance.mapreduce.avro;

import artistdistance.mapreduce.BfsCounter;
import artistdistance.mapreduce.BfsIterationStep;
import artistdistance.mapreduce.BfsMessage;
import artistdistance.schema.BfsVertex;
import java.io.IOException;
import org.apache.avro.mapred.AvroKey;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public final class AvroBfsIterationReducer
    extends Reducer<Text, BfsMessage, AvroKey<BfsVertex>, NullWritable> {
  @Override
  protected void reduce(Text key, Iterable<BfsMessage> values, Context context)
      throws IOException, InterruptedException {
    BfsIterationStep.Result result = BfsIterationStep.reduce(key.toString(), values);
    if (result.discovered()) {
      context.getCounter(BfsCounter.DISCOVERED).increment(1);
    }
    context.write(new AvroKey<>(result.vertex()), NullWritable.get());
  }
}
