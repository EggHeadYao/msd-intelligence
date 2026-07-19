#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
readonly CONFIG_DIR="$WORKSPACE_ROOT/p1team02/year_prediction/config/tuning/key_contracts"
readonly TRAINER="$WORKSPACE_ROOT/p1team02/year_prediction/src/training/train_sgd.py"
readonly VALIDATOR="$WORKSPACE_ROOT/p1team02/year_prediction/src/evaluation/validate_model.py"
readonly MASTER='local[6]'
readonly DRIVER_MEMORY='4g'
readonly CONTRACTS=(k0 k1 k2 k3)

cd "$WORKSPACE_ROOT"

for contract in "${CONTRACTS[@]}"; do
  config="$CONFIG_DIR/$contract.json"
  input="$(jq -er '.input' "$config")"
  metadata="$(jq -er '.feature_metadata' "$config")"
  model_id="$(jq -er '.model_id' "$config")"
  output_root="$(jq -er '.output_root' "$config")"

  [[ -d "$WORKSPACE_ROOT/$input" ]] || {
    printf 'missing feature input: %s\n' "$WORKSPACE_ROOT/$input" >&2
    exit 1
  }
  [[ -f "$WORKSPACE_ROOT/$metadata" ]] || {
    printf 'missing feature metadata: %s\n' "$WORKSPACE_ROOT/$metadata" >&2
    exit 1
  }
  [[ ! -e "$WORKSPACE_ROOT/$output_root/$model_id" ]] || {
    printf 'model output already exists: %s\n' "$WORKSPACE_ROOT/$output_root/$model_id" >&2
    exit 1
  }
done

for contract in "${CONTRACTS[@]}"; do
  config="$CONFIG_DIR/$contract.json"
  model_id="$(jq -er '.model_id' "$config")"
  output_root="$(jq -er '.output_root' "$config")"
  model="$WORKSPACE_ROOT/$output_root/$model_id"

  printf 'training contract=%s master=%s driver_memory=%s\n' \
    "$contract" "$MASTER" "$DRIVER_MEMORY"
  spark-submit \
    --master "$MASTER" \
    --driver-memory "$DRIVER_MEMORY" \
    "$TRAINER" \
    --config "$config"

  spark-submit \
    --master "$MASTER" \
    --driver-memory "$DRIVER_MEMORY" \
    "$VALIDATOR" \
    --model "$model"
done
