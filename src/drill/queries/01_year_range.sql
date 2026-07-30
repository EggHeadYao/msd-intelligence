SELECT
  MIN(year) AS oldest_year,
  MAX(year) AS youngest_year
FROM dfs.msd.`songs_scalar.parquet`
WHERE year > 0;
