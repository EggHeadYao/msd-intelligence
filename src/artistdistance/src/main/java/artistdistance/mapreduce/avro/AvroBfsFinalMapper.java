package artistdistance.mapreduce.avro;

import artistdistance.bfs.BfsRules;
import artistdistance.schema.ArtistDistance;
import artistdistance.schema.BfsVertex;
import java.io.IOException;
import org.apache.avro.mapred.AvroKey;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Mapper;

public final class AvroBfsFinalMapper
    extends Mapper<AvroKey<BfsVertex>, NullWritable, AvroKey<ArtistDistance>, NullWritable> {
  @Override
  protected void map(AvroKey<BfsVertex> key, NullWritable value, Context context)
      throws IOException, InterruptedException {
    context.write(new AvroKey<>(BfsRules.toArtistDistance(key.datum())), NullWritable.get());
  }
}
