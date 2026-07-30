#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DRILL_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPOSITORY_DIR=$(cd -- "$DRILL_DIR/.." && pwd)
WORKSPACE_DIR=$(cd -- "$REPOSITORY_DIR/.." && pwd)

DRILL_HOME=${DRILL_HOME:-/usr/local/drill}
DRILL_DATA_DIR=${1:-$WORKSPACE_DIR/parquets/year_prediction/raw}
DRILL_LOG_DIR=${DRILL_LOG_DIR:-/tmp/p1team02-drill}

SCALAR_FILE=$DRILL_DATA_DIR/songs_scalar.parquet
AUDIO_DIR=$DRILL_DATA_DIR/audio_features

test -x "$DRILL_HOME/bin/drill-embedded"
test -f "$SCALAR_FILE"
test -d "$AUDIO_DIR"

export DRILL_CONF_DIR=$DRILL_DIR/conf
export DRILL_DATA_DIR
export DRILL_LOG_DIR

mkdir -p "$DRILL_DIR/results" "$DRILL_LOG_DIR"

for query in "$DRILL_DIR"/queries/*.sql; do
  name=$(basename "$query" .sql)
  output=$DRILL_DIR/results/$name.csv
  "$DRILL_HOME/bin/drill-embedded" \
    --silent=true \
    --showHeader=true \
    --outputformat=csv \
    --force=false \
    -f "$query" > "$output"
  echo "Wrote $output"
done
