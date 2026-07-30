package artistdistance.mapreduce.parquet;

import artistdistance.bfs.BfsRules;
import artistdistance.schema.ArtistDistance;
import artistdistance.schema.BfsVertex;
import java.io.IOException;
import org.apache.hadoop.mapreduce.Mapper;

public final class ParquetBfsFinalMapper extends Mapper<Void, BfsVertex, Void, ArtistDistance> {
  @Override
  protected void map(Void key, BfsVertex value, Context context)
      throws IOException, InterruptedException {
    context.write(null, BfsRules.toArtistDistance(value));
  }
}
