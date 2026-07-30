#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YEAR_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
NATIVE_ROOT="$YEAR_ROOT/.synapseml-native"
SOURCE="$NATIVE_ROOT/src"
BUILD="$NATIVE_ROOT/build"
LIB="$NATIVE_ROOT/lib"
LIGHTGBM_COMMIT=0957ab72f7e46649dc0e48daf15d1c2a9381c6a3

if [[ ! -d "$SOURCE/.git" ]]; then
  git clone --filter=blob:none --no-checkout \
    https://github.com/microsoft/LightGBM.git "$SOURCE"
fi

git -C "$SOURCE" fetch origin "$LIGHTGBM_COMMIT" --depth 1
git -C "$SOURCE" checkout --detach "$LIGHTGBM_COMMIT"
git -C "$SOURCE" submodule update --init \
  external_libs/compute external_libs/fast_double_parser external_libs/fmt

cmake -U CMAKE_LIBRARY_OUTPUT_DIRECTORY \
  -S "$SOURCE" -B "$BUILD" \
  -DUSE_SWIG=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS=-I/usr/include/eigen3
cmake --build "$BUILD" --parallel "${JOBS:-$(nproc)}"

mkdir -p "$LIB"
cp "$SOURCE/lib_lightgbm.so" "$SOURCE/lib_lightgbm_swig.so" "$LIB/"
file "$LIB/lib_lightgbm.so" "$LIB/lib_lightgbm_swig.so"
