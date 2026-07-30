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

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeBoolean(vertex);
    out.writeBoolean(distance != null);
    if (distance != null) {
      out.writeInt(distance);
    }
    out.writeBoolean(parent != null);
    if (parent != null) {
      out.writeUTF(parent);
    }
    if (vertex) {
      out.writeInt(degree);
      out.writeInt(neighbors.size());
      for (String neighbor : neighbors) {
        out.writeUTF(neighbor);
      }
      out.writeUTF(status.name());
    }
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    vertex = in.readBoolean();
    distance = in.readBoolean() ? in.readInt() : null;
    parent = in.readBoolean() ? in.readUTF() : null;
    if (vertex) {
      degree = in.readInt();
      int size = in.readInt();
      neighbors = new ArrayList<>(size);
      for (int i = 0; i < size; i++) {
        neighbors.add(in.readUTF());
      }
      status = BfsStatus.valueOf(in.readUTF());
    } else {
      degree = 0;
      neighbors = List.of();
      status = null;
    }
  }
}
