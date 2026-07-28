#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MERLIN_ROOT=${MERLIN_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
MERLIN_PYTHON=${MERLIN_PYTHON:-/home/zjk/.venvs/merlin-faiss/bin/python}
MERLIN_SPARK_SUBMIT=${MERLIN_SPARK_SUBMIT:-/opt/spark/bin/spark-submit}
MERLIN_RANKER_ROOT=${MERLIN_RANKER_ROOT:-$MERLIN_ROOT/parquets_new/merlin/ranker}

MIN_FREE_GB=${MIN_FREE_GB:-8}
VALIDATION_MIN_FREE_GB=${VALIDATION_MIN_FREE_GB:-4}
MODEL_MIN_FREE_GB=${MODEL_MIN_FREE_GB:-2}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
VALIDATION_SPARK_CORES=${VALIDATION_SPARK_CORES:-6}
MODEL_SPARK_CORES=${MODEL_SPARK_CORES:-4}
SHUFFLE_PARTITIONS=${SHUFFLE_PARTITIONS:-64}
VALIDATION_DRIVER_MEMORY=${VALIDATION_DRIVER_MEMORY:-5g}
MODEL_DRIVER_MEMORY=${MODEL_DRIVER_MEMORY:-5500m}
MAX_BLOCK_SIZE_MB=${MAX_BLOCK_SIZE_MB:-8}

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$MERLIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_CORETYPE=${OPENBLAS_CORETYPE:-generic}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export OMP_NUM_THREADS
export PYSPARK_PYTHON="$MERLIN_PYTHON"
export PYSPARK_DRIVER_PYTHON="$MERLIN_PYTHON"
export SPARK_LOCAL_IP=${SPARK_LOCAL_IP:-127.0.0.1}
export JAVA_TOOL_OPTIONS=${JAVA_TOOL_OPTIONS:--XX:UseAVX=0 -XX:UseSSE=2 -XX:-TieredCompilation}

STEPS=(
  split
  recall-contract
  weak-labels
  set-a-candidates
  tuning-pairs
  tuning-features
  set-b-candidates
  validation-groups
  validation-features
  tune-model
  retrain-data
  ablation-data
  full-model
  ablation-model
  development-protocol
  development-candidates
  development-groups
  development-features
  development-evaluation
)

FROM_STEP=${FROM_STEP:-${STEPS[0]}}
TO_STEP=${TO_STEP:-${STEPS[${#STEPS[@]}-1]}}
DRY_RUN=0
TEMP_ENTRY_DIR=

usage() {
  cat <<'EOF'
Usage: merlin/inference/scripts/run_c3_pipeline.sh [options]

Run the canonical C3 training and development-evaluation pipeline. Completed
artifacts are skipped, and streamed retraining resumes from its checkpoint.

Options:
  --from STEP     Start at STEP (earlier steps are not checked).
  --to STEP       Stop after STEP.
  --dry-run       Print commands without running them.
  --list-steps    Print valid step names and exit.
  -h, --help      Show this help.

Common overrides are environment variables: MERLIN_ROOT, MERLIN_PYTHON,
MIN_FREE_GB, VALIDATION_MIN_FREE_GB, MODEL_MIN_FREE_GB, OMP_NUM_THREADS,
VALIDATION_SPARK_CORES, MODEL_SPARK_CORES, and SHUFFLE_PARTITIONS.
EOF
}

die() {
  printf 'c3_pipeline_error: %s\n' "$*" >&2
  exit 1
}

step_index() {
  local wanted=$1 index
  for index in "${!STEPS[@]}"; do
    if [[ ${STEPS[$index]} == "$wanted" ]]; then
      printf '%s\n' "$index"
      return 0
    fi
  done
  return 1
}

while (($#)); do
  case $1 in
    --from)
      (($# >= 2)) || die '--from requires a step'
      FROM_STEP=$2
      shift 2
      ;;
    --to)
      (($# >= 2)) || die '--to requires a step'
      TO_STEP=$2
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --list-steps)
      printf '%s\n' "${STEPS[@]}"
      exit 0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

FROM_INDEX=$(step_index "$FROM_STEP") || die "unknown --from step: $FROM_STEP"
TO_INDEX=$(step_index "$TO_STEP") || die "unknown --to step: $TO_STEP"
((FROM_INDEX <= TO_INDEX)) || die '--from must not be later than --to'

[[ -d $MERLIN_ROOT/merlin/inference ]] || die "invalid MERLIN_ROOT: $MERLIN_ROOT"
[[ -x $MERLIN_PYTHON ]] || die "Python is not executable: $MERLIN_PYTHON"
[[ -x $MERLIN_SPARK_SUBMIT ]] || die "spark-submit is not executable: $MERLIN_SPARK_SUBMIT"
cd "$MERLIN_ROOT"
