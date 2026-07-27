#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: summarize_results.sh <results.csv> [summary.csv] [comparisons.csv]" >&2
  exit 1
fi

input_csv="$1"
summary_csv="${2:-$(dirname "${input_csv}")/summary.csv}"
comparisons_csv="${3:-$(dirname "${input_csv}")/comparisons.csv}"
expected_header="run_id,order_index,source_id,engine,format,wall_seconds,yarn_seconds,memory_seconds,vcore_seconds,application_count,expected_total,reachable,unreachable,max_distance,verified"

[[ -f "${input_csv}" ]] || { echo "results do not exist: ${input_csv}" >&2; exit 1; }
[[ "$(head -n 1 "${input_csv}")" == "${expected_header}" ]] || { echo "unexpected results schema: ${input_csv}" >&2; exit 1; }

mkdir -p "$(dirname "${summary_csv}")" "$(dirname "${comparisons_csv}")"
echo "engine,format,runs,median_wall_seconds,min_wall_seconds,max_wall_seconds,wall_iqr_seconds,median_yarn_seconds,median_memory_seconds,median_vcore_seconds,expected_total,reachable,unreachable,max_distance,all_verified" > "${summary_csv}"

all_outcomes="$(awk -F, 'NR > 1 && $15 == "true" { print $11 "," $12 "," $13 "," $14 }' "${input_csv}" | sort -u)"
[[ "$(printf '%s\n' "${all_outcomes}" | wc -l)" -eq 1 ]] || { echo "BFS outcomes differ between combinations" >&2; exit 1; }

column_stats() {
  local engine="$1"
  local format="$2"
  local column="$3"
  awk -F, -v engine="${engine}" -v format="${format}" -v column="${column}" \
    'NR > 1 && $4 == engine && $5 == format && $15 == "true" { print $column }' "${input_csv}" | \
    sort -n | \
    awk '
      function quantile(p, position, lower, fraction) {
        position = (count - 1) * p + 1
        lower = int(position)
        fraction = position - lower
        if (lower >= count) return value[count]
        return value[lower] + fraction * (value[lower + 1] - value[lower])
      }
      { value[++count] = $1 }
      END {
        if (count == 0) exit 1
        printf "%.3f %.3f %.3f %.3f %.3f", quantile(0.5), value[1], value[count], quantile(0.25), quantile(0.75)
      }
    '
}

declare -A median_wall
for engine in mapreduce spark; do
  for format in avro parquet; do
    runs="$(awk -F, -v engine="${engine}" -v format="${format}" 'NR > 1 && $4 == engine && $5 == format && $15 == "true" { count++ } END { print count + 0 }' "${input_csv}")"
    [[ "${runs}" -gt 0 ]] || { echo "no verified rows for ${engine}+${format}" >&2; exit 1; }

    read -r wall_median wall_min wall_max wall_q1 wall_q3 <<< "$(column_stats "${engine}" "${format}" 6)"
    read -r yarn_median _ <<< "$(column_stats "${engine}" "${format}" 7)"
    read -r memory_median _ <<< "$(column_stats "${engine}" "${format}" 8)"
    read -r vcore_median _ <<< "$(column_stats "${engine}" "${format}" 9)"
    wall_iqr="$(awk -v q1="${wall_q1}" -v q3="${wall_q3}" 'BEGIN { printf "%.3f", q3 - q1 }')"

    outcome="$(awk -F, -v engine="${engine}" -v format="${format}" 'NR > 1 && $4 == engine && $5 == format && $15 == "true" { print $11 "," $12 "," $13 "," $14 }' "${input_csv}" | sort -u)"
    [[ "$(printf '%s\n' "${outcome}" | wc -l)" -eq 1 ]] || { echo "inconsistent BFS outcomes for ${engine}+${format}" >&2; exit 1; }
    failures="$(awk -F, -v engine="${engine}" -v format="${format}" 'NR > 1 && $4 == engine && $5 == format && $15 != "true" { count++ } END { print count + 0 }' "${input_csv}")"
    all_verified=true
    [[ "${failures}" -eq 0 ]] || all_verified=false

    median_wall["${engine}-${format}"]="${wall_median}"
    echo "${engine},${format},${runs},${wall_median},${wall_min},${wall_max},${wall_iqr},${yarn_median},${memory_median},${vcore_median},${outcome},${all_verified}" >> "${summary_csv}"
  done
done

speedup() {
  awk -v baseline="$1" -v candidate="$2" 'BEGIN { printf "%.3f", baseline / candidate }'
}

echo "comparison,baseline,candidate,wall_speedup" > "${comparisons_csv}"
echo "engine_avro,mapreduce-avro,spark-avro,$(speedup "${median_wall[mapreduce-avro]}" "${median_wall[spark-avro]}")" >> "${comparisons_csv}"
echo "engine_parquet,mapreduce-parquet,spark-parquet,$(speedup "${median_wall[mapreduce-parquet]}" "${median_wall[spark-parquet]}")" >> "${comparisons_csv}"
echo "format_mapreduce,mapreduce-avro,mapreduce-parquet,$(speedup "${median_wall[mapreduce-avro]}" "${median_wall[mapreduce-parquet]}")" >> "${comparisons_csv}"
echo "format_spark,spark-avro,spark-parquet,$(speedup "${median_wall[spark-avro]}" "${median_wall[spark-parquet]}")" >> "${comparisons_csv}"

echo "Wrote ${summary_csv} and ${comparisons_csv}"
