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

cleanup() {
  if [[ -n $TEMP_ENTRY_DIR && -d $TEMP_ENTRY_DIR ]]; then
    rm -rf -- "$TEMP_ENTRY_DIR"
  fi
}
trap cleanup EXIT

if [[ -n ${MERLIN_SPARK_ENTRY:-} ]]; then
  [[ -f $MERLIN_SPARK_ENTRY ]] || die "Spark entry does not exist: $MERLIN_SPARK_ENTRY"
else
  TEMP_ENTRY_DIR=$(mktemp -d "${TMPDIR:-/tmp}/merlin-c3-entry.XXXXXX")
  MERLIN_SPARK_ENTRY=$TEMP_ENTRY_DIR/train_ranker.py
  printf '%s\n' \
    'from merlin.inference.scripts.ranker.train_ranker import main' \
    '' \
    'if __name__ == "__main__":' \
    '    main()' >"$MERLIN_SPARK_ENTRY"
fi

run_command() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if ((DRY_RUN == 0)); then
    "$@"
  fi
}

outputs_complete() {
  local output
  for output in "$@"; do
    [[ -e $output ]] || return 1
  done
}

run_step() {
  local name=$1 index
  shift
  index=$(step_index "$name") || die "internal unknown step: $name"
  if ((index < FROM_INDEX || index > TO_INDEX)); then
    return 0
  fi

  local -a outputs=()
  while (($#)) && [[ $1 != -- ]]; do
    outputs+=("$1")
    shift
  done
  (($#)) || die "step $name has no command separator"
  shift

  if outputs_complete "${outputs[@]}"; then
    printf 'c3_pipeline_skip step=%s reason=outputs_complete\n' "$name"
    return 0
  fi

  printf 'c3_pipeline_start step=%s\n' "$name"
  run_command "$@"
  if ((DRY_RUN == 0)) && ! outputs_complete "${outputs[@]}"; then
    die "step $name returned without all declared outputs"
  fi
  printf 'c3_pipeline_done step=%s\n' "$name"
}

spark_common=(
  --conf spark.eventLog.enabled=false
  --conf spark.ui.enabled=false
  --conf "spark.sql.shuffle.partitions=$SHUFFLE_PARTITIONS"
  --conf spark.hadoop.fs.defaultFS=file:///
  --conf spark.driver.bindAddress=127.0.0.1
)

split_assignments=$MERLIN_RANKER_ROOT/split_assignments.parquet
split_manifest=$MERLIN_RANKER_ROOT/split_manifest.json
candidate_policy=$MERLIN_RANKER_ROOT/candidate_policy_manifest.json
tag_idf=$MERLIN_RANKER_ROOT/tag_idf.json
weak_thresholds=$MERLIN_RANKER_ROOT/weak_label_thresholds.json
weak_positives=$MERLIN_RANKER_ROOT/weak_positives.parquet
weak_manifest=$MERLIN_RANKER_ROOT/weak_positives_manifest.json
set_a_pool=$MERLIN_RANKER_ROOT/candidate_pool.parquet
set_a_pool_manifest=$MERLIN_RANKER_ROOT/candidate_pool_manifest.json
set_b_pool=$MERLIN_RANKER_ROOT/set_b_candidate_pool.parquet
set_b_pool_manifest=$MERLIN_RANKER_ROOT/set_b_candidate_pool_manifest.json
tuning_root=$MERLIN_RANKER_ROOT/tuning
tuning_pairs=$tuning_root/training_pairs.parquet
tuning_pairs_manifest=$tuning_root/training_pairs_manifest.json
tuning_features=$tuning_root/raw_pair_features.parquet
tuning_features_manifest=$tuning_root/raw_pair_features_manifest.json
validation_thresholds=$MERLIN_RANKER_ROOT/validation_group_thresholds.json
validation_positives=$MERLIN_RANKER_ROOT/validation_group_positives.parquet
validation_pairs=$MERLIN_RANKER_ROOT/validation_pairs.parquet
validation_groups_manifest=$MERLIN_RANKER_ROOT/validation_groups_manifest.json
validation_features=$MERLIN_RANKER_ROOT/validation_raw_features.parquet
validation_features_manifest=$MERLIN_RANKER_ROOT/validation_raw_features_manifest.json
tuning_model=$MERLIN_RANKER_ROOT/tuning_model
training_pairs=$MERLIN_RANKER_ROOT/training_pairs.parquet
training_pairs_manifest=$MERLIN_RANKER_ROOT/training_pairs_manifest.json
training_features=$MERLIN_RANKER_ROOT/raw_pair_features.parquet
training_features_manifest=$MERLIN_RANKER_ROOT/raw_pair_features_manifest.json
ablation_pairs=$MERLIN_RANKER_ROOT/no_hard_neg_training_pairs.parquet
ablation_pairs_manifest=$MERLIN_RANKER_ROOT/no_hard_neg_training_pairs_manifest.json
ablation_features=$MERLIN_RANKER_ROOT/no_hard_neg_raw_pair_features.parquet
ablation_features_manifest=$MERLIN_RANKER_ROOT/no_hard_neg_raw_pair_features_manifest.json
ablation_model=$MERLIN_RANKER_ROOT/no_hard_neg_model
development_root=$MERLIN_RANKER_ROOT/development_evaluation
development_protocol=$development_root/protocol.json
development_pool=$development_root/candidate_pool.parquet
development_pool_manifest=$development_root/candidate_pool_manifest.json
development_positives=$development_root/validation_group_positives.parquet
development_pairs=$development_root/validation_pairs.parquet
development_groups_manifest=$development_root/validation_groups_manifest.json
development_features=$development_root/raw_pair_features.parquet
development_features_manifest=$development_root/raw_pair_features_manifest.json
development_report=$development_root/evaluation_report.json

run_step split "$split_assignments" "$split_manifest" -- \
  "$MERLIN_PYTHON" -m merlin.inference.scripts.ranker.build_split \
  --songs-metadata "$MERLIN_ROOT/parquets_new/prepared/songs_metadata.parquet" \
  --assignments "$split_assignments" \
  --manifest "$split_manifest"

run_step recall-contract "$candidate_policy" "$tag_idf" -- \
  "$MERLIN_PYTHON" -m merlin.inference.scripts.recall.build_recall_artifacts \
  --graph-edges "$MERLIN_ROOT/parquets_new/prepared/graph_edges.parquet" \
  --candidate-policy "$candidate_policy" \
  --tag-idf "$tag_idf"

run_step weak-labels "$weak_thresholds" "$weak_positives" "$weak_manifest" -- \
  "$MERLIN_PYTHON" -m merlin.inference.scripts.ranker.build_weak_labels \
  --split-assignments "$split_assignments" \
  --split-manifest "$split_manifest" \
  --thresholds "$weak_thresholds" \
  --positives "$weak_positives" \
  --manifest "$weak_manifest" \
  --query-splits set_a \
  --min-free-gb "$MIN_FREE_GB"

run_step set-a-candidates "$set_a_pool" "$set_a_pool_manifest" -- \
  "$MERLIN_PYTHON" -m merlin.inference.scripts.recall.export_candidates \
  --split-assignments "$split_assignments" \
  --split-manifest "$split_manifest" \
  --query-split set_a \
  --output "$set_a_pool" \
  --manifest "$set_a_pool_manifest" \
  --min-free-gb "$MIN_FREE_GB"

run_step tuning-pairs "$tuning_pairs" "$tuning_pairs_manifest" -- \
  "$MERLIN_PYTHON" -m merlin.inference.scripts.ranker.build_training_pairs \
  --candidate-pool "$set_a_pool" \
  --candidate-pool-manifest "$set_a_pool_manifest" \
  --weak-positives "$weak_positives" \
  --weak-positives-manifest "$weak_manifest" \
  --thresholds "$weak_thresholds" \
  --split-assignments "$split_assignments" \
  --split-manifest "$split_manifest" \
  --stage tuning \
  --scope formal \
  --output "$tuning_pairs" \
  --manifest "$tuning_pairs_manifest"

run_step tuning-features "$tuning_features" "$tuning_features_manifest" -- \
  "$MERLIN_PYTHON" -m merlin.inference.scripts.ranker.export_ranker_features \
  --pair-kind training \
  --pairs "$tuning_pairs" \
  --pairs-manifest "$tuning_pairs_manifest" \
  --output "$tuning_features" \
  --manifest "$tuning_features_manifest" \
  --stage tuning \
  --scope formal \
  --min-free-gb "$MIN_FREE_GB"

run_step set-b-candidates "$set_b_pool" "$set_b_pool_manifest" -- \
  "$MERLIN_PYTHON" -m merlin.inference.scripts.recall.export_candidates \
  --split-assignments "$split_assignments" \
  --split-manifest "$split_manifest" \
  --query-split set_b \
  --output "$set_b_pool" \
  --manifest "$set_b_pool_manifest" \
  --min-free-gb "$MIN_FREE_GB"

run_step validation-groups \
  "$validation_thresholds" "$validation_positives" "$validation_pairs" \
  "$validation_groups_manifest" -- \
  "$MERLIN_SPARK_SUBMIT" \
  --master "local[$VALIDATION_SPARK_CORES]" \
  --driver-memory "$VALIDATION_DRIVER_MEMORY" \
  "${spark_common[@]}" \
  "$MERLIN_ROOT/merlin/inference/scripts/ranker/build_validation_groups.py" \
  --candidate-pool "$set_b_pool" \
  --candidate-pool-manifest "$set_b_pool_manifest" \
  --thresholds "$validation_thresholds" \
  --positives "$validation_positives" \
  --validation-pairs "$validation_pairs" \
  --manifest "$validation_groups_manifest" \
  --apply-split set_b \
  --scope formal \
  --audio-pair-engine numpy \
  --audio-block-size 256 \
  --shuffle-partitions "$SHUFFLE_PARTITIONS" \
  --scratch-root "$MERLIN_RANKER_ROOT/.c3-scratch" \
  --min-free-gb "$VALIDATION_MIN_FREE_GB"

run_step validation-features "$validation_features" "$validation_features_manifest" -- \
  "$MERLIN_PYTHON" -m merlin.inference.scripts.ranker.export_ranker_features \
  --pair-kind validation \
  --pairs "$validation_pairs" \
  --pairs-manifest "$validation_groups_manifest" \
  --validation-positives "$validation_positives" \
  --validation-thresholds "$validation_thresholds" \
  --output "$validation_features" \
  --manifest "$validation_features_manifest" \
  --stage tuning \
  --scope formal \
  --min-free-gb "$MIN_FREE_GB"

run_step tune-model \
  "$tuning_model/training_manifest.json" \
  "$tuning_model/ranker_coefficients.json" \
  "$tuning_model/ranker_scaler.json" -- \
  "$MERLIN_SPARK_SUBMIT" \
  --master "local[$MODEL_SPARK_CORES]" \
  --driver-memory "$MODEL_DRIVER_MEMORY" \
  "${spark_common[@]}" \
  --conf spark.memory.fraction=0.40 \
  "$MERLIN_SPARK_ENTRY" \
  --train-features "$tuning_features" \
  --train-features-manifest "$tuning_features_manifest" \
  --validation-features "$validation_features" \
  --validation-features-manifest "$validation_features_manifest" \
  --output "$tuning_model" \
  --stage tuning \
  --scope formal \
  --scratch-root "$MERLIN_RANKER_ROOT/.c3-scratch" \
  --min-free-gb "$VALIDATION_MIN_FREE_GB" \
  --max-block-size-mb "$MAX_BLOCK_SIZE_MB" \
  --parent "audio_index_manifest=$MERLIN_ROOT/parquets_new/merlin/audio/index_audio_manifest.json" \
  --parent "graph_index_manifest=$MERLIN_ROOT/parquets_new/merlin/graph/index_graph_manifest.json" \
  --parent "candidate_policy_manifest=$candidate_policy" \
  --parent "tag_idf=$tag_idf" \
  --parent "validation_features_manifest=$validation_features_manifest"

run_step retrain-data \
  "$training_pairs" "$training_pairs_manifest" \
  "$training_features" "$training_features_manifest" -- \
  "$MERLIN_PYTHON" -m merlin.inference.scripts.ranker.build_training_pairs \
  --stage final_retrain \
  --scope formal \
  --split-assignments "$split_assignments" \
  --split-manifest "$split_manifest" \
  --thresholds "$weak_thresholds" \
  --output "$training_pairs" \
  --manifest "$training_pairs_manifest" \
  --features-output "$training_features" \
  --features-manifest "$training_features_manifest" \
  --batch-size 256 \
  --rows-per-file 250000 \
  --positive-neighbor-limit 1001 \
  --min-free-gb "$MIN_FREE_GB"

run_step ablation-data \
  "$ablation_pairs" "$ablation_pairs_manifest" \
  "$ablation_features" "$ablation_features_manifest" -- \
  "$MERLIN_PYTHON" -m merlin.inference.scripts.ranker.build_training_pairs \
  --stage final_retrain \
  --negative-mode random_only \
  --scope formal \
  --split-assignments "$split_assignments" \
  --split-manifest "$split_manifest" \
  --thresholds "$weak_thresholds" \
  --full-training-pairs "$training_pairs" \
  --full-training-pairs-manifest "$training_pairs_manifest" \
  --full-features "$training_features" \
  --full-features-manifest "$training_features_manifest" \
  --output "$ablation_pairs" \
  --manifest "$ablation_pairs_manifest" \
  --features-output "$ablation_features" \
  --features-manifest "$ablation_features_manifest" \
  --rows-per-file 250000 \
  --min-free-gb "$MIN_FREE_GB"

selected_reg() {
  local manifest=$tuning_model/training_manifest.json
  if ((DRY_RUN)); then
    printf '<selected_reg_param>'
    return 0
  fi
  [[ -f $manifest ]] || die "missing tuning manifest: $manifest"
  "$MERLIN_PYTHON" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["selected_reg_param"])' \
    "$manifest"
}

FULL_MODEL_INDEX=$(step_index full-model)
ABLATION_MODEL_INDEX=$(step_index ablation-model)
if ((FROM_INDEX <= ABLATION_MODEL_INDEX && TO_INDEX >= FULL_MODEL_INDEX)); then
  MERLIN_REG=$(selected_reg)
else
  MERLIN_REG=unused
fi

run_step full-model \
  "$MERLIN_RANKER_ROOT/training_manifest.json" \
  "$MERLIN_RANKER_ROOT/ranker_coefficients.json" \
  "$MERLIN_RANKER_ROOT/ranker_scaler.json" -- \
  "$MERLIN_SPARK_SUBMIT" \
  --master "local[$MODEL_SPARK_CORES]" \
  --driver-memory "$MODEL_DRIVER_MEMORY" \
  "${spark_common[@]}" \
  --conf spark.memory.fraction=0.40 \
  "$MERLIN_SPARK_ENTRY" \
  --train-features "$training_features" \
  --train-features-manifest "$training_features_manifest" \
  --output "$MERLIN_RANKER_ROOT" \
  --stage final_retrain \
  --training-variant full \
  --fixed-reg-param "$MERLIN_REG" \
  --frozen-scaler "$tuning_model/ranker_scaler.json" \
  --frozen-tuning-manifest "$tuning_model/training_manifest.json" \
  --scope formal \
  --scratch-root "$MERLIN_RANKER_ROOT/.c3-scratch" \
  --min-free-gb "$MODEL_MIN_FREE_GB" \
  --max-block-size-mb "$MAX_BLOCK_SIZE_MB" \
  --parent "audio_index_manifest=$MERLIN_ROOT/parquets_new/merlin/audio/index_audio_manifest.json" \
  --parent "graph_index_manifest=$MERLIN_ROOT/parquets_new/merlin/graph/index_graph_manifest.json" \
  --parent "candidate_policy_manifest=$candidate_policy" \
  --parent "tag_idf=$tag_idf" \
  --parent "training_pairs_manifest=$training_pairs_manifest"

run_step ablation-model \
  "$ablation_model/training_manifest.json" \
  "$ablation_model/ranker_coefficients.json" \
  "$ablation_model/ranker_scaler.json" -- \
  "$MERLIN_SPARK_SUBMIT" \
  --master "local[$MODEL_SPARK_CORES]" \
  --driver-memory "$MODEL_DRIVER_MEMORY" \
  "${spark_common[@]}" \
  --conf spark.memory.fraction=0.40 \
  "$MERLIN_SPARK_ENTRY" \
  --train-features "$ablation_features" \
  --train-features-manifest "$ablation_features_manifest" \
  --training-pairs-manifest "$ablation_pairs_manifest" \
  --base-train-features "$training_features" \
  --base-train-features-manifest "$training_features_manifest" \
  --output "$ablation_model" \
  --stage final_retrain \
  --training-variant no_hard_neg \
  --fixed-reg-param "$MERLIN_REG" \
  --frozen-scaler "$tuning_model/ranker_scaler.json" \
  --frozen-tuning-manifest "$tuning_model/training_manifest.json" \
  --scope formal \
  --scratch-root "$MERLIN_RANKER_ROOT/.c3-scratch" \
  --min-free-gb "$MODEL_MIN_FREE_GB" \
  --max-block-size-mb "$MAX_BLOCK_SIZE_MB" \
  --parent "audio_index_manifest=$MERLIN_ROOT/parquets_new/merlin/audio/index_audio_manifest.json" \
  --parent "graph_index_manifest=$MERLIN_ROOT/parquets_new/merlin/graph/index_graph_manifest.json" \
  --parent "candidate_policy_manifest=$candidate_policy" \
  --parent "tag_idf=$tag_idf"
