package artistdistance.mapreduce;

import artistdistance.bfs.BfsRules;
import artistdistance.schema.BfsStatus;
import artistdistance.schema.BfsVertex;
import java.io.IOException;

public final class BfsIterationStep {
  private BfsIterationStep() {}

  public record Result(BfsVertex vertex, boolean discovered) {}

  public static Result reduce(String id, Iterable<BfsMessage> values) throws IOException {
    BfsVertex vertex = null;
    Integer bestDistance = null;
    String bestParent = null;
    for (BfsMessage value : values) {
      if (value.isVertex()) {
        vertex = value.toVertex(id);
      } else if (BfsRules.isBetterCandidate(
          bestDistance, bestParent, value.getDistance(), value.getParent())) {
        bestDistance = value.getDistance();
        bestParent = value.getParent();
      }
    }
    if (vertex == null) {
      throw new IOException("Missing vertex message for id: " + id);
    }
    boolean discovered = vertex.getStatus() == BfsStatus.unvisited && bestDistance != null;
    if (discovered) {
      vertex.setDistance(bestDistance);
      vertex.setParent(bestParent);
    }
    vertex.setStatus(BfsRules.nextStatus(vertex.getStatus(), discovered));
    return new Result(vertex, discovered);
  }
}
