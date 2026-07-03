package artistdistance.mapreduce;

import artistdistance.bfs.BfsRules;
import artistdistance.schema.BfsStatus;
import artistdistance.schema.BfsVertex;
import java.io.IOException;
import org.apache.avro.mapred.AvroKey;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public final class BfsIterationReducer
    extends Reducer<Text, BfsMessage, AvroKey<BfsVertex>, NullWritable> {
  public enum Counter {
    DISCOVERED
  }

  @Override
  protected void reduce(Text key, Iterable<BfsMessage> values, Context context)
      throws IOException, InterruptedException {
    BfsVertex vertex = null;
    Integer bestDistance = null;
    String bestParent = null;
    for (BfsMessage value : values) {
      if (value.isVertex()) {
        vertex = value.toVertex(key.toString());
      } else if (BfsRules.isBetterCandidate(
          bestDistance, bestParent, value.getDistance(), value.getParent())) {
        bestDistance = value.getDistance();
        bestParent = value.getParent();
      }
    }
    if (vertex == null) {
      throw new IOException("Missing vertex message for id: " + key);
    }
    boolean newlyDiscovered = vertex.getStatus() == BfsStatus.unvisited && bestDistance != null;
    if (newlyDiscovered) {
      vertex.setDistance(bestDistance);
      vertex.setParent(bestParent);
      context.getCounter(Counter.DISCOVERED).increment(1);
    }
    vertex.setStatus(BfsRules.nextStatus(vertex.getStatus(), newlyDiscovered));
    context.write(new AvroKey<>(vertex), NullWritable.get());
  }
}
