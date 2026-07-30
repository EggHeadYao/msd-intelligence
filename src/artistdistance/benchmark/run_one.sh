#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: run_one.sh <mapreduce|spark> <avro|parquet> [run_id]" >&2
  exit 1
fi

engine="$1"
format="$2"
run_id="${3:-1}"

case "${engine}" in
  mapreduce|spark) ;;
  *) echo "engine must be mapreduce or spark: ${engine}" >&2; exit 1 ;;
esac

case "${format}" in
  avro|parquet) ;;
  *) echo "format must be avro or parquet: ${format}" >&2; exit 1 ;;
esac

[[ "${run_id}" =~ ^[1-9][0-9]*$ ]] || { echo "run_id must be a positive integer" >&2; exit 1; }

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
data_dir="$(cd "${project_dir}/.." && pwd)/data"
results_csv="${RESULTS_CSV:-${project_dir}/experiments/results.csv}"
hdfs_root="${HDFS_BENCHMARK_ROOT:-/user/${USER}/artistdistance-benchmark}"
source_id="${SOURCE_ID:-ARGUACZ1187FB3F35C}"
order_index="${ORDER_INDEX:-1}"
reducers="${MAPREDUCE_REDUCERS:-4}"
restart_cluster="${RESTART_CLUSTER:-true}"
keep_hdfs_input="${KEEP_HDFS_INPUT:-false}"
refresh_hdfs_input="${REFRESH_HDFS_INPUT:-false}"
jar_path="${project_dir}/target/artistdistance-1.0-SNAPSHOT.jar"
deps_dir="${project_dir}/target/dependency"
spark_avro_jar="${HOME}/.m2/repository/org/apache/spark/spark-avro_2.13/4.1.2/spark-avro_2.13-4.1.2.jar"
results_header="run_id,order_index,source_id,engine,format,wall_seconds,yarn_seconds,memory_seconds,vcore_seconds,application_count,expected_total,reachable,unreachable,max_distance,verified"

[[ "${order_index}" =~ ^[1-4]$ ]] || { echo "ORDER_INDEX must be between 1 and 4" >&2; exit 1; }
[[ "${restart_cluster}" == "true" || "${restart_cluster}" == "false" ]] || { echo "RESTART_CLUSTER must be true or false" >&2; exit 1; }
[[ "${keep_hdfs_input}" == "true" || "${keep_hdfs_input}" == "false" ]] || { echo "KEEP_HDFS_INPUT must be true or false" >&2; exit 1; }
[[ "${refresh_hdfs_input}" == "true" || "${refresh_hdfs_input}" == "false" ]] || { echo "REFRESH_HDFS_INPUT must be true or false" >&2; exit 1; }

mkdir -p "$(dirname "${results_csv}")"
if [[ ! -f "${results_csv}" ]]; then
  echo "${results_header}" > "${results_csv}"
elif [[ "$(head -n 1 "${results_csv}")" != "${results_header}" ]]; then
  echo "unexpected results schema: ${results_csv}" >&2
  exit 1
fi

local_input="${data_dir}/artistdistance-output/${format}/adjacency.${format}"
input_path="${hdfs_root}/input/${format}/adjacency.${format}"
format_class="${format^}"

if [[ "${engine}" == "mapreduce" ]]; then
  main="artistdistance.mapreduce.${format}.${format_class}MapReduceBfs"
else
  main="artistdistance.spark.${format}.${format_class}SparkBfs"
fi
verifier="artistdistance.validate.${format_class}BfsOutputVerifier"

output_name="$(date -u +"%Y%m%dT%H%M%S")-${engine}-${format}-run${run_id}-${BASHPID}"
output_path="${hdfs_root}/output/${output_name}"
local_output_dir="${project_dir}/experiments/output/${output_name}"
command_output_file="$(mktemp)"
wall_time_file="$(mktemp)"
verify_output_file="$(mktemp)"
trap 'rm -f "${command_output_file}" "${wall_time_file}" "${verify_output_file}"' EXIT

[[ -e "${local_input}" ]] || { echo "local input does not exist: ${local_input}" >&2; exit 1; }

cd "${project_dir}"
if [[ ! -f "${jar_path}" || ! -d "${deps_dir}" ]]; then
  mvn -q package dependency:copy-dependencies -DincludeScope=runtime >/dev/null
fi
runtime_classpath="${jar_path}:$(find "${deps_dir}" -name '*.jar' | sort | paste -sd: -)"

running_yarn_nodes() {
  local listing
  listing="$(yarn node -list -states RUNNING 2>/dev/null || true)"
  printf '%s\n' "${listing}" | awk -F: '/Total Nodes/ { gsub(/[[:space:]]/, "", $2); print $2; exit }'
}

if [[ "${restart_cluster}" == "true" ]]; then
  running_before="$(running_yarn_nodes)"
  expected_nodes="${YARN_EXPECTED_RUNNING_NODES:-${running_before:-1}}"
  [[ "${expected_nodes}" =~ ^[1-9][0-9]*$ ]] || { echo "YARN_EXPECTED_RUNNING_NODES must be a positive integer" >&2; exit 1; }
  stop-all.sh >/dev/null
  start-all.sh >/dev/null
  hdfs dfsadmin -safemode wait >/dev/null 2>&1

  deadline="$((SECONDS + ${YARN_START_TIMEOUT_SECONDS:-180}))"
  while [[ "$(running_yarn_nodes)" -lt "${expected_nodes}" ]]; do
    if (( SECONDS >= deadline )); then
      echo "YARN did not reach ${expected_nodes} running nodes" >&2
      exit 1
    fi
    sleep 2
  done
fi

hdfs dfs -mkdir -p "$(dirname "${input_path}")" "${hdfs_root}/output" >/dev/null 2>&1
input_uploaded=false
if [[ "${refresh_hdfs_input}" == "true" ]]; then
  hdfs dfs -rm -f "${input_path}" >/dev/null 2>&1 || true
fi
if ! hdfs dfs -test -e "${input_path}" >/dev/null 2>&1; then
  hdfs dfs -put "${local_input}" "${input_path}" >/dev/null 2>&1
  input_uploaded=true
fi
if hdfs dfs -test -e "${output_path}" >/dev/null 2>&1; then
  echo "output path already exists: ${output_path}" >&2
  exit 1
fi

if [[ "${engine}" == "mapreduce" ]]; then
  libjars="$(find "${deps_dir}" -name '*.jar' | sort | paste -sd, -)"
  set +e
  LC_ALL=C /usr/bin/time -f '%e' -o "${wall_time_file}" \
    env HADOOP_CLASSPATH="${runtime_classpath}" \
    hadoop jar "${jar_path}" "${main}" \
      -Dmapreduce.framework.name=yarn \
      -Dartistdistance.bfs.reducers="${reducers}" \
      -libjars "${libjars}" \
      "${input_path}" "${source_id}" "${output_path}" \
      >"${command_output_file}" 2>&1
  command_status=$?
  set -e
else
  spark_jars=()
  if [[ "${format}" == "avro" ]]; then
    [[ -f "${spark_avro_jar}" ]] || { echo "spark avro jar does not exist: ${spark_avro_jar}" >&2; exit 1; }
    spark_jars=(--jars "${spark_avro_jar}")
  fi
  set +e
  LC_ALL=C /usr/bin/time -f '%e' -o "${wall_time_file}" \
    spark-submit \
      --master yarn \
      --deploy-mode client \
      --driver-java-options "-Dspark.master=yarn" \
      --conf "spark.sql.shuffle.partitions=${SPARK_SHUFFLE_PARTITIONS:-32}" \
      "${spark_jars[@]}" \
      --class "${main}" \
      "${jar_path}" \
      "${input_path}" "${source_id}" "${output_path}" \
      >"${command_output_file}" 2>&1
  command_status=$?
  set -e
fi

wall_seconds="$(tr -d '[:space:]' < "${wall_time_file}")"
if [[ "${command_status}" -ne 0 ]]; then
  cat "${command_output_file}" >&2
  exit "${command_status}"
fi

mapfile -t app_ids < <(grep -o 'application_[0-9_]*' "${command_output_file}" | sort -u)
mapfile -t job_ids < <(grep -o 'job_[0-9_]*' "${command_output_file}" | sort -u)
if [[ "${#app_ids[@]}" -eq 0 ]]; then
  cat "${command_output_file}" >&2
  echo "no YARN application id found" >&2
  exit 1
fi

elapsed_ms=0
memory_seconds=0
vcore_seconds=0
for app_id in "${app_ids[@]}"; do
  status="$(yarn application -status "${app_id}" 2>/dev/null)"
  start_ms="$(printf '%s\n' "${status}" | awk -F: '/Start-Time/ { gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit }')"
  finish_ms="$(printf '%s\n' "${status}" | awk -F: '/Finish-Time/ { gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit }')"
  final_state="$(printf '%s\n' "${status}" | awk -F: '/Final-State/ { gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit }')"
  allocation="$(printf '%s\n' "${status}" | sed -n 's/.*: *\([0-9][0-9]*\) MB-seconds, *\([0-9][0-9]*\) vcore-seconds.*/\1 \2/p' | head -n 1)"
  if [[ "${final_state}" != "SUCCEEDED" || -z "${start_ms}" || -z "${finish_ms}" || "${finish_ms}" -le 0 ]]; then
    cat "${command_output_file}" >&2
    echo "cannot read a successful completed status for ${app_id}" >&2
    exit 1
  fi
  elapsed_ms="$((elapsed_ms + finish_ms - start_ms))"
  if [[ -n "${allocation}" ]]; then
    read -r app_memory_seconds app_vcore_seconds <<< "${allocation}"
    memory_seconds="$((memory_seconds + app_memory_seconds))"
    vcore_seconds="$((vcore_seconds + app_vcore_seconds))"
  fi
done

yarn_seconds="$(awk -v ms="${elapsed_ms}" 'BEGIN { printf "%.3f", ms / 1000 }')"

mkdir -p "$(dirname "${local_output_dir}")"
hdfs dfs -copyToLocal "${output_path}" "${local_output_dir}" >/dev/null 2>&1
verify_input="${local_output_dir}"
if [[ -d "${local_output_dir}/final" ]]; then
  verify_input="${local_output_dir}/final"
fi

set +e
java -cp "${runtime_classpath}" "${verifier}" "${local_input}" "${verify_input}" "${source_id}" \
  >"${verify_output_file}" 2>&1
verify_status=$?
set -e

metric_value() {
  awk -F= -v key="$1" '$1 == key { print $2; exit }' "${verify_output_file}"
}

expected_total="$(metric_value expected_total)"
reachable="$(metric_value reachable)"
unreachable="$(metric_value unreachable)"
max_distance="$(metric_value max_distance)"
mismatches="$(metric_value mismatches)"
if [[ "${verify_status}" -eq 0 && "${mismatches:-1}" -eq 0 ]]; then
  verified=true
else
  verified=false
fi

echo "${run_id},${order_index},${source_id},${engine},${format},${wall_seconds},${yarn_seconds},${memory_seconds},${vcore_seconds},${#app_ids[@]},${expected_total},${reachable},${unreachable},${max_distance},${verified}" >> "${results_csv}"

if [[ "${verified}" == "true" ]]; then
  rm -rf "${local_output_dir}"
  rmdir "$(dirname "${local_output_dir}")" 2>/dev/null || true
  hdfs dfs -rm -r -f "${output_path}" >/dev/null 2>&1 || true
  if [[ "${input_uploaded}" == "true" && "${keep_hdfs_input}" != "true" ]]; then
    hdfs dfs -rm -f "${input_path}" >/dev/null 2>&1 || true
    hdfs dfs -rmdir "$(dirname "${input_path}")" "${hdfs_root}/input" >/dev/null 2>&1 || true
  fi
  for app_id in "${app_ids[@]}"; do
    hdfs dfs -rm -r -f "/user/${USER}/.sparkStaging/${app_id}" >/dev/null 2>&1 || true
  done
  for job_id in "${job_ids[@]}"; do
    hdfs dfs -rm -r -f "/tmp/hadoop-yarn/staging/${USER}/.staging/${job_id}" >/dev/null 2>&1 || true
    hdfs dfs -rm -r -f "/user/${USER}/.staging/${job_id}" >/dev/null 2>&1 || true
  done
else
  cat "${verify_output_file}" >&2
  if [[ "${verify_status}" -eq 0 ]]; then
    exit 1
  fi
  exit "${verify_status}"
fi

echo "${engine}+${format}: wall=${wall_seconds}s yarn=${yarn_seconds}s verified=${verified}"
