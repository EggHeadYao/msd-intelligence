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

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
data_dir="$(cd "${project_dir}/.." && pwd)/data"
results_csv="${project_dir}/experiments/results.csv"
hdfs_root="${HDFS_BENCHMARK_ROOT:-/user/${USER}/artistdistance-benchmark}"
source_id="${SOURCE_ID:-ARGUACZ1187FB3F35C}"
reducers="${MAPREDUCE_REDUCERS:-4}"
jar_path="${project_dir}/target/artistdistance-1.0-SNAPSHOT.jar"
deps_dir="${project_dir}/target/dependency"
spark_avro_jar="${HOME}/.m2/repository/org/apache/spark/spark-avro_2.13/4.1.2/spark-avro_2.13-4.1.2.jar"

mkdir -p "$(dirname "${results_csv}")"

results_header="run_id,source_id,engine,format,elapsed_seconds,verified"
if [[ ! -f "${results_csv}" ]]; then
  echo "${results_header}" > "${results_csv}"
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

output_name="$(date -u +"%Y%m%dT%H%M%SZ")-${engine}-${format}-run${run_id}"
output_path="${hdfs_root}/output/${output_name}"
local_output_dir="${project_dir}/experiments/output/${output_name}"
output_file="$(mktemp)"
verify_output_file="$(mktemp)"
trap 'rm -f "${output_file}" "${verify_output_file}"' EXIT

[[ -e "${local_input}" ]] || { echo "local input does not exist: ${local_input}" >&2; exit 1; }

cd "${project_dir}"

if [[ ! -f "${jar_path}" || ! -d "${deps_dir}" ]]; then
  mvn -q package dependency:copy-dependencies -DincludeScope=runtime >/dev/null
fi
runtime_classpath="${jar_path}:$(find "${deps_dir}" -name '*.jar' | sort | paste -sd: -)"

stop-all.sh >/dev/null
start-all.sh >/dev/null
hdfs dfsadmin -safemode wait >/dev/null 2>&1

hdfs dfs -mkdir -p "$(dirname "${input_path}")" "${hdfs_root}/output" >/dev/null 2>&1
input_uploaded=false
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
  HADOOP_CLASSPATH="${runtime_classpath}" hadoop jar "${jar_path}" "${main}" \
    -Dmapreduce.framework.name=yarn \
    -Dartistdistance.bfs.reducers="${reducers}" \
    -libjars "${libjars}" \
    "${input_path}" "${source_id}" "${output_path}" \
    >"${output_file}" 2>&1
  command_status=$?
  set -e
else
  spark_jars=()
  if [[ "${format}" == "avro" ]]; then
    [[ -f "${spark_avro_jar}" ]] || { echo "spark avro jar does not exist: ${spark_avro_jar}" >&2; exit 1; }
    spark_jars=(--jars "${spark_avro_jar}")
  fi
  set +e
  spark-submit \
    --master yarn \
    --deploy-mode client \
    --driver-java-options "-Dspark.master=yarn" \
    --conf "spark.sql.shuffle.partitions=${SPARK_SHUFFLE_PARTITIONS:-32}" \
    "${spark_jars[@]}" \
    --class "${main}" \
    "${jar_path}" \
    "${input_path}" "${source_id}" "${output_path}" \
    >"${output_file}" 2>&1
  command_status=$?
  set -e
fi

if [[ "${command_status}" -ne 0 ]]; then
  cat "${output_file}" >&2
  exit "${command_status}"
fi

mapfile -t app_ids < <(grep -o 'application_[0-9_]*' "${output_file}" | sort -u)
mapfile -t job_ids < <(grep -o 'job_[0-9_]*' "${output_file}" | sort -u)
if [[ "${#app_ids[@]}" -eq 0 ]]; then
  cat "${output_file}" >&2
  echo "no YARN application id found" >&2
  exit 1
fi

elapsed_ms=0
for app_id in "${app_ids[@]}"; do
  status="$(yarn application -status "${app_id}" 2>/dev/null)"
  start_ms="$(printf "%s\n" "${status}" | awk -F: '/Start-Time/ { gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit }')"
  finish_ms="$(printf "%s\n" "${status}" | awk -F: '/Finish-Time/ { gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit }')"
  final_state="$(printf "%s\n" "${status}" | awk -F: '/Final-State/ { gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit }')"
  if [[ "${final_state}" != "SUCCEEDED" ]]; then
    cat "${output_file}" >&2
    echo "${app_id} final state is ${final_state}" >&2
    exit 1
  fi
  if [[ -z "${start_ms}" || -z "${finish_ms}" || "${finish_ms}" -le 0 ]]; then
    cat "${output_file}" >&2
    echo "cannot read completed YARN timing for ${app_id}" >&2
    exit 1
  fi
  elapsed_ms="$((elapsed_ms + finish_ms - start_ms))"
done

elapsed_seconds="$(awk -v ms="${elapsed_ms}" 'BEGIN { printf "%.3f", ms / 1000 }')"

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

if [[ "${verify_status}" -eq 0 ]]; then
  verified=true
else
  verified=false
fi

echo "${run_id},${source_id},${engine},${format},${elapsed_seconds},${verified}" >> "${results_csv}"

if [[ "${verified}" == "true" ]]; then
  rm -rf "${local_output_dir}"
  rmdir "$(dirname "${local_output_dir}")" 2>/dev/null || true
  hdfs dfs -rm -r -f "${output_path}" >/dev/null 2>&1 || true
  if [[ "${input_uploaded}" == "true" ]]; then
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
  exit "${verify_status}"
fi

echo "${engine}+${format}: ${elapsed_seconds}s verified=${verified}"
