package artistdistance.mapreduce.parquet;

import artistdistance.bfs.BfsRules;
import artistdistance.mapreduce.BfsMessage;
import artistdistance.schema.BfsVertex;
import java.io.IOException;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public final class ParquetBfsIterationMapper extends Mapper<Void, BfsVertex, Text, BfsMessage> {
  @Override
  protected void map(Void key, BfsVertex vertex, Context context)
      throws IOException, InterruptedException {
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
