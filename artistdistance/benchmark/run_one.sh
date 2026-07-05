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

results_header="run_id,source_id,engine,format,elapsed_seconds"
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

output_path="${hdfs_root}/output/$(date -u +"%Y%m%dT%H%M%SZ")-${engine}-${format}-run${run_id}"
output_file="$(mktemp)"
trap 'rm -f "${output_file}"' EXIT

[[ -e "${local_input}" ]] || { echo "local input does not exist: ${local_input}" >&2; exit 1; }

cd "${project_dir}"

if [[ ! -f "${jar_path}" || ! -d "${deps_dir}" ]]; then
  mvn -q package dependency:copy-dependencies -DincludeScope=runtime >/dev/null
fi
