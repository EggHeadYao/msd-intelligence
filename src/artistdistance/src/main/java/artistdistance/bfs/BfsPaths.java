package artistdistance.bfs;

public final class BfsPaths {
  private BfsPaths() {}

  public static String initial(String baseDir) {
    return iteration(baseDir, 0);
  }

  public static String iteration(String baseDir, int iteration) {
    if (iteration < 0) {
      throw new IllegalArgumentException("Iteration must be non-negative: " + iteration);
    }
    return child(baseDir, String.format("iter-%04d", iteration));
  }

  public static String finalOutput(String baseDir) {
    return child(baseDir, "final");
  }

  private static String child(String parent, String child) {
    return parent.endsWith("/") ? parent + child : parent + "/" + child;
  }
}
