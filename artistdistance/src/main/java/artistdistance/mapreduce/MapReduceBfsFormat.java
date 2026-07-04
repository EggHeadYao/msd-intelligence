package artistdistance.mapreduce;

import java.io.IOException;
import org.apache.hadoop.mapreduce.Job;

public interface MapReduceBfsFormat {
  String name();

  void configureInitJob(Job job) throws IOException;

  void configureIterationJob(Job job) throws IOException;

  void configureFinalJob(Job job) throws IOException;
}
