#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: run_experiment.sh [repetitions]" >&2
  exit 1
fi

repetitions="${1:-5}"
[[ "${repetitions}" =~ ^[1-9][0-9]*$ ]] || { echo "repetitions must be a positive integer" >&2; exit 1; }

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
workspace_dir="$(cd "${project_dir}/../.." && pwd)"
data_output="${project_dir}/../data/artistdistance-output"
database="${ARTIST_SIMILARITY_DB:-${workspace_dir}/msd/AdditionalFiles/artist_similarity.db}"
results_csv="${RESULTS_CSV:-${project_dir}/experiments/results.csv}"
summary_csv="${SUMMARY_CSV:-${project_dir}/experiments/summary.csv}"
comparisons_csv="${COMPARISONS_CSV:-${project_dir}/experiments/comparisons.csv}"
source_id="${SOURCE_ID:-ARGUACZ1187FB3F35C}"
hdfs_root="${HDFS_BENCHMARK_ROOT:-/user/${USER}/artistdistance-benchmark}"
results_header="run_id,order_index,source_id,engine,format,wall_seconds,yarn_seconds,memory_seconds,vcore_seconds,application_count,expected_total,reachable,unreachable,max_distance,verified"

cd "${project_dir}"
echo "Building and testing artistdistance..."
mvn -q package dependency:copy-dependencies -DincludeScope=runtime

avro_input="${data_output}/avro/adjacency.avro"
parquet_input="${data_output}/parquet/adjacency.parquet"
rebuild_inputs="${REBUILD_INPUTS:-false}"
[[ "${rebuild_inputs}" == "true" || "${rebuild_inputs}" == "false" ]] || { echo "REBUILD_INPUTS must be true or false" >&2; exit 1; }

convert_inputs() {
  [[ -f "${database}" ]] || { echo "database does not exist: ${database}" >&2; exit 1; }
  echo "Converting the full artist graph to Avro and Parquet..."
  MAVEN_OPTS="${MAVEN_OPTS:--Xmx4g}" mvn -q exec:java \
    -Dexec.mainClass=artistdistance.convert.ArtistGraphConverter \
    -Dexec.args="${database} ${data_output}"
}

if [[ "${rebuild_inputs}" == "true" ]]; then
  convert_inputs
elif [[ ! -e "${avro_input}" || ! -e "${parquet_input}" ]]; then
  if [[ -e "${avro_input}" || -e "${parquet_input}" ]]; then
    echo "only one benchmark input exists; remove or restore ${data_output} before continuing" >&2
    exit 1
  fi
  convert_inputs
fi

if [[ -f "${results_csv}" ]]; then
  [[ "$(head -n 1 "${results_csv}")" == "${results_header}" ]] || {
    echo "unexpected results schema: ${results_csv}" >&2
    exit 1
  }
  if ! awk -F, -v source="${source_id}" 'NR > 1 && $3 != source { exit 1 }' "${results_csv}"; then
    echo "results already contain a different source_id: ${results_csv}" >&2
    exit 1
  fi
fi

completed() {
  local run_id="$1"
  local engine="$2"
  local format="$3"
  [[ -f "${results_csv}" ]] && awk -F, -v run="${run_id}" -v source="${source_id}" -v engine="${engine}" -v format="${format}" \
    'NR > 1 && $1 == run && $3 == source && $4 == engine && $5 == format && $15 == "true" { found = 1 } END { exit !found }' \
    "${results_csv}"
}

combinations=(
  "mapreduce avro"
  "mapreduce parquet"
  "spark avro"
  "spark parquet"
)

echo "Running ${repetitions} repetitions with source ${source_id}."
declare -A refreshed_hdfs_input=(
  [avro]=false
  [parquet]=false
)
for run_id in $(seq 1 "${repetitions}"); do
  start_index="$(((run_id - 1) % ${#combinations[@]}))"
  for offset in $(seq 0 "$((${#combinations[@]} - 1))"); do
    combination_index="$(((start_index + offset) % ${#combinations[@]}))"
    read -r engine format <<< "${combinations[${combination_index}]}"
    order_index="$((offset + 1))"

    if completed "${run_id}" "${engine}" "${format}"; then
      echo "Skipping completed run ${run_id}: ${engine}+${format}"
      continue
    fi

    echo "Run ${run_id}/${repetitions}, order ${order_index}/4: ${engine}+${format}"
    refresh_input=true
    [[ "${refreshed_hdfs_input[${format}]}" == "false" ]] || refresh_input=false
    RESULTS_CSV="${results_csv}" \
      SOURCE_ID="${source_id}" \
      ORDER_INDEX="${order_index}" \
      KEEP_HDFS_INPUT=true \
      REFRESH_HDFS_INPUT="${refresh_input}" \
      "${script_dir}/run_one.sh" "${engine}" "${format}" "${run_id}"
    refreshed_hdfs_input["${format}"]=true
  done
done

expected_completed="$((repetitions * ${#combinations[@]}))"
actual_completed="$(awk -F, '$15 == "true" { count++ } END { print count + 0 }' "${results_csv}")"
if [[ "${actual_completed}" -ne "${expected_completed}" ]]; then
  echo "expected ${expected_completed} verified runs, found ${actual_completed}" >&2
  exit 1
fi

"${script_dir}/summarize_results.sh" "${results_csv}" "${summary_csv}" "${comparisons_csv}"

if [[ "${CLEANUP_HDFS_INPUT_AFTER_EXPERIMENT:-true}" == "true" ]]; then
  hdfs dfs -rm -r -f "${hdfs_root}/input" >/dev/null 2>&1 || true
fi

echo "Experiment complete: ${results_csv}"
echo "Summary: ${summary_csv}"
echo "Comparisons: ${comparisons_csv}"
