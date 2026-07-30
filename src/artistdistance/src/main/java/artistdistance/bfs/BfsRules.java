package artistdistance.bfs;

import artistdistance.schema.Adjacency;
import artistdistance.schema.ArtistDistance;
import artistdistance.schema.BfsStatus;
import artistdistance.schema.BfsVertex;
import java.util.ArrayList;
import java.util.Objects;

public final class BfsRules {
  private BfsRules() {}

  public static BfsVertex initialVertex(Adjacency row, String sourceId) {
    boolean source = Objects.equals(row.getId(), sourceId);
    return new BfsVertex(
        row.getId(),
        new ArrayList<>(row.getNeighbors()),
        row.getDegree(),
        source ? 0 : null,
        null,
        source ? BfsStatus.frontier : BfsStatus.unvisited);
  }

  public static boolean isFrontier(BfsVertex vertex) {
    return vertex.getStatus() == BfsStatus.frontier;
  }

  public static int candidateDistance(BfsVertex frontier) {
    Integer distance = frontier.getDistance();
    if (distance == null) {
      throw new IllegalArgumentException("Frontier vertex has no distance: " + frontier.getId());
    }
    return distance + 1;
  }

  public static boolean isBetterCandidate(
      Integer currentDistance, String currentParent, int candidateDistance, String candidateParent) {
    if (currentDistance == null) {
      return true;
    }
    if (candidateDistance != currentDistance) {
      return candidateDistance < currentDistance;
    }
    return candidateParent != null
        && (currentParent == null || candidateParent.compareTo(currentParent) < 0);
  }

  public static BfsStatus nextStatus(BfsStatus current, boolean newlyDiscovered) {
    if (current == BfsStatus.frontier) {
      return BfsStatus.visited;
    }
    if (current == BfsStatus.unvisited && newlyDiscovered) {
      return BfsStatus.frontier;
    }
    return current;
  }

  public static ArtistDistance toArtistDistance(BfsVertex vertex) {
    return new ArtistDistance(vertex.getId(), vertex.getDistance(), vertex.getParent());
  }
}
