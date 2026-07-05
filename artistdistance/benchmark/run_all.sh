#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: run_all.sh [repetitions]" >&2
  exit 1
fi

repetitions="${1:-1}"
[[ "${repetitions}" =~ ^[1-9][0-9]*$ ]] || { echo "repetitions must be a positive integer" >&2; exit 1; }

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for run_id in $(seq 1 "${repetitions}"); do
  "${script_dir}/run_one.sh" mapreduce avro "${run_id}"
  "${script_dir}/run_one.sh" mapreduce parquet "${run_id}"
  "${script_dir}/run_one.sh" spark avro "${run_id}"
  "${script_dir}/run_one.sh" spark parquet "${run_id}"
done
