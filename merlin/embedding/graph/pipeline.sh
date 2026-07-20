#!/bin/bash
# MERLIN C2 Complete Pipeline: Meta-path walks → Word2Vec → FAISS

set -e

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$PROJECT_ROOT"

INPUT_DIR="$PROJECT_ROOT/merlin/parquets/prepared"
WALKS_DIR="$PROJECT_ROOT/merlin/parquets/walks"
EMBEDDINGS_DIR="$PROJECT_ROOT/merlin/parquets/embeddings"
INDICES_DIR="$PROJECT_ROOT/merlin/parquets/indices"

# Parameters
WALKS_PER_SONG=${1:-3}
WALK_LENGTH=${2:-40}
VECTOR_SIZE=${3:-128}
ITERATIONS=${4:-10}

echo "=========================================="
echo "MERLIN C2 Pipeline"
echo "=========================================="
echo "Input: $INPUT_DIR"
echo "Walks: r=$WALKS_PER_SONG, L=$WALK_LENGTH"
echo "Embeddings: vector_size=$VECTOR_SIZE, iterations=$ITERATIONS"
echo ""

# Phase 1: Generate meta-path walks
echo "Phase 1: Generating meta-path walks..."
mkdir -p "$WALKS_DIR"
PYTHONPATH="$PROJECT_ROOT" spark-submit \
  --driver-memory 8g --executor-memory 4g \
  merlin/embedding/graph/main.py \
  --input "$INPUT_DIR" \
  --output "$WALKS_DIR" \
  --walks "$WALKS_PER_SONG" \
  --length "$WALK_LENGTH"

echo ""
echo "Phase 2: Training Word2Vec on walks..."
mkdir -p "$EMBEDDINGS_DIR"
PYTHONPATH="$PROJECT_ROOT" spark-submit \
  --driver-memory 8g --executor-memory 4g \
  merlin/embedding/graph/train_word2vec.py \
  --input "$WALKS_DIR" \
  --output "$EMBEDDINGS_DIR" \
  --vector-size "$VECTOR_SIZE" \
  --num-iterations "$ITERATIONS"

echo ""
echo "Phase 3: Building FAISS index..."
mkdir -p "$INDICES_DIR"
python3 merlin/embedding/graph/build_faiss.py \
  --embeddings "$EMBEDDINGS_DIR/song_embeddings_graph.parquet" \
  --output "$INDICES_DIR"

echo ""
echo "=========================================="
echo "[OK] C2 Pipeline Complete!"
echo "=========================================="
echo "Outputs:"
echo "  - Walks:      $WALKS_DIR"
echo "  - Embeddings: $EMBEDDINGS_DIR"
echo "  - Indices:    $INDICES_DIR"
echo ""
