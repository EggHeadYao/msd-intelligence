SELECT
  s.track_id,
  s.title,
  s.artist_name,
  s.song_hotttnesss,
  s.duration,
  COALESCE(a.energy, CAST(0.0 AS DOUBLE)) AS energy,
  a.energy IS NOT NULL AS energy_available,
  s.tempo
FROM dfs.msd.`songs_scalar.parquet` AS s
LEFT JOIN dfs.msd.`audio_features/features_*.parquet` AS a
  ON s.track_id = a.track_id
WHERE s.song_hotttnesss IS NOT NULL
  AND s.duration > 0
  AND s.tempo > 0
ORDER BY
  s.song_hotttnesss DESC,
  s.duration ASC,
  COALESCE(a.energy, CAST(0.0 AS DOUBLE)) DESC,
  s.tempo ASC,
  s.track_id ASC
LIMIT 1;
