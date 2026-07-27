SELECT
  track_id,
  title,
  artist_id,
  artist_name,
  duration
FROM dfs.msd.`songs_scalar.parquet`
WHERE duration > 0
  AND artist_name IS NOT NULL
  AND artist_name <> ''
ORDER BY duration DESC, track_id ASC
LIMIT 1;
