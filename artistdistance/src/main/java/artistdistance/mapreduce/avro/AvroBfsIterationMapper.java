package artistdistance.mapreduce.avro;

import artistdistance.bfs.BfsRules;
import artistdistance.mapreduce.BfsMessage;
import artistdistance.schema.BfsVertex;
import java.io.IOException;
import org.apache.avro.mapred.AvroKey;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public final class AvroBfsIterationMapper
    extends Mapper<AvroKey<BfsVertex>, NullWritable, Text, BfsMessage> {
  @Override
  protected void map(AvroKey<BfsVertex> key, NullWritable value, Context context)
      throws IOException, InterruptedException {
    BfsVertex vertex = key.datum();
    String id = vertex.getId();
    context.write(new Text(id), BfsMessage.vertex(vertex));
    if (!BfsRules.isFrontier(vertex)) {
      return;
    }
    int distance = BfsRules.candidateDistance(vertex);
    for (String neighbor : vertex.getNeighbors()) {
      context.write(new Text(neighbor), BfsMessage.candidate(distance, id));
    }
  }
}
