#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x "$ROOT_DIR/pgolf/bin/torchrun" ]]; then
  TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT_DIR/pgolf/bin/torchrun}"
else
  TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
fi

: "${RUN_ID:=laptop_smoke}"
: "${ENABLE_COMPILE:=0}"
: "${VAL_MAX_TOKENS:=131072}"
: "${ITERATIONS:=2}"
: "${WARMUP_STEPS:=0}"
: "${TRAIN_BATCH_TOKENS:=8192}"
: "${VAL_BATCH_SIZE:=8192}"
: "${VAL_LOSS_EVERY:=0}"
: "${TRAIN_LOG_EVERY:=1}"
: "${DATA_PATH:=./data/datasets/fineweb10B_sp1024}"
: "${TOKENIZER_PATH:=./data/tokenizers/fineweb_1024_bpe.model}"
: "${VOCAB_SIZE:=1024}"

exec env \
  RUN_ID="$RUN_ID" \
  ENABLE_COMPILE="$ENABLE_COMPILE" \
  VAL_MAX_TOKENS="$VAL_MAX_TOKENS" \
  ITERATIONS="$ITERATIONS" \
  WARMUP_STEPS="$WARMUP_STEPS" \
  TRAIN_BATCH_TOKENS="$TRAIN_BATCH_TOKENS" \
  VAL_BATCH_SIZE="$VAL_BATCH_SIZE" \
  VAL_LOSS_EVERY="$VAL_LOSS_EVERY" \
  TRAIN_LOG_EVERY="$TRAIN_LOG_EVERY" \
  DATA_PATH="$DATA_PATH" \
  TOKENIZER_PATH="$TOKENIZER_PATH" \
  VOCAB_SIZE="$VOCAB_SIZE" \
  "$TORCHRUN_BIN" --standalone --nproc_per_node=1 ./train_gpt.py
