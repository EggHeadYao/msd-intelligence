package artistdistance.validate;

import artistdistance.schema.Adjacency;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

public final class ReferenceBfs {
  private ReferenceBfs() {}

  public record Result(
      Set<String> ids, Map<String, Integer> distances, Map<String, String> parents) {}

  public static Result compute(List<Adjacency> rows, String sourceId) {
    Map<String, List<String>> graph = new TreeMap<>();
    for (Adjacency row : rows) {
      if (graph.put(row.getId(), List.copyOf(row.getNeighbors())) != null) {
        throw new IllegalArgumentException("Duplicate adjacency id: " + row.getId());
      }
    }
    if (!graph.containsKey(sourceId)) {
      throw new IllegalArgumentException("Source artist not found: " + sourceId);
    }
    Map<String, Integer> distances = new HashMap<>();
    Map<String, String> parents = new HashMap<>();
    Set<String> frontier = new TreeSet<>();
    distances.put(sourceId, 0);
    frontier.add(sourceId);
    int nextDistance = 1;
    while (!frontier.isEmpty()) {
      Map<String, String> candidates = new TreeMap<>();
      for (String parent : frontier) {
        for (String neighbor : graph.getOrDefault(parent, List.of())) {
          if (distances.containsKey(neighbor)) {
            continue;
          }
          String bestParent = candidates.get(neighbor);
          if (bestParent == null || parent.compareTo(bestParent) < 0) {
            candidates.put(neighbor, parent);
          }
        }
      }
      frontier = new TreeSet<>();
      for (var candidate : candidates.entrySet()) {
        if (!distances.containsKey(candidate.getKey())) {
          distances.put(candidate.getKey(), nextDistance);
          parents.put(candidate.getKey(), candidate.getValue());
          frontier.add(candidate.getKey());
        }
      }
      nextDistance++;
    }
    Set<String> ids = new HashSet<>(graph.keySet());
    ids.addAll(distances.keySet());
    return new Result(ids, distances, parents);
  }
}
