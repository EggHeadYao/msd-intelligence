SELECT
  release_7digitalid AS album_id,
  MIN(release) AS album_name,
  COUNT(DISTINCT track_id) AS track_count
FROM dfs.msd.`songs_scalar.parquet`
WHERE release_7digitalid IS NOT NULL
  AND release_7digitalid > 0
  AND release IS NOT NULL
  AND release <> ''
GROUP BY release_7digitalid
ORDER BY track_count DESC, album_id ASC
LIMIT 1;
