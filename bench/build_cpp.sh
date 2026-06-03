#!/usr/bin/env bash
# Build the original nla-group/snn C++ extension (snnomp) for benchmarking.
#
# Requirements: a C++ compiler with OpenMP, a CBLAS implementation (cblas.h +
# libcblas), and the project's venv (for pybind11 + matching Python headers).
#
# Usage:  uv run bash bench/build_cpp.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SRC="$ROOT/snn_orig/snnpy/snnpy.cpp"
OUT="$HERE/_cpp/snnomp.so"

if [[ ! -f "$SRC" ]]; then
  echo "Cloning upstream reference repo into $ROOT/snn_orig ..."
  git clone --depth 1 https://github.com/nla-group/snn.git "$ROOT/snn_orig"
fi

mkdir -p "$HERE/_cpp"
INCLUDES="$(python -m pybind11 --includes)"

echo "Compiling $SRC -> $OUT"
g++ -O3 -fopenmp -shared -std=c++17 -fPIC \
    $INCLUDES "$SRC" -o "$OUT" -lcblas -fopenmp

echo "Done: $OUT"
