package artistdistance.mapreduce;

import artistdistance.schema.BfsStatus;
import artistdistance.schema.BfsVertex;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.hadoop.io.Writable;

public final class BfsMessage implements Writable {
  private boolean vertex;
  private List<String> neighbors = List.of();
  private int degree;
  private Integer distance;
  private String parent;
  private BfsStatus status;

  public static BfsMessage vertex(BfsVertex value) {
    BfsMessage message = new BfsMessage();
    message.vertex = true;
    message.neighbors = new ArrayList<>(value.getNeighbors());
    message.degree = value.getDegree();
    message.distance = value.getDistance();
    message.parent = value.getParent();
    message.status = value.getStatus();
    return message;
  }

  public static BfsMessage candidate(int distance, String parent) {
    BfsMessage message = new BfsMessage();
    message.distance = distance;
    message.parent = parent;
    return message;
  }

  public boolean isVertex() {
    return vertex;
  }

  public int getDistance() {
    return distance;
  }

  public String getParent() {
    return parent;
  }

  public BfsVertex toVertex(String id) {
    return new BfsVertex(id, new ArrayList<>(neighbors), degree, distance, parent, status);
  }

}
