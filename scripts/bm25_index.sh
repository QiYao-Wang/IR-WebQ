#!/usr/bin/env bash
set -euo pipefail

# 路径配置
ROOT="/home/yangzhihao/data1_link/UCAS_hmw_term1/IR/IR-WebQ"
JAVA_HOME="$ROOT/configs/jdk/graalvm-jdk-24.0.2+11.1"
PYTHON_BIN="python"

INPUT_DIR="$ROOT/datasets"
INDEX_OUT="$ROOT/outputs/lucene/bm25-webq"

export JAVA_HOME
export PATH="$JAVA_HOME/bin:$PATH"

echo "Building BM25 index to: $INDEX_OUT"
$PYTHON_BIN -m pyserini.index.lucene \
  --collection JsonCollection \
  --input "$INPUT_DIR" \
  --index "$INDEX_OUT" \
  --generator DefaultLuceneDocumentGenerator \
  --fields text \
  --threads 8 \
  --storePositions --storeDocvectors --storeRaw

echo "Done. bm25_index_path = $INDEX_OUT"

