#!/bin/bash
# SuomiChat: Train a Finnish GPT from scratch.
#
# This is the main training pipeline for SuomiChat. It downloads Finnish data,
# trains a tokenizer, pretrains a GPT, fine-tunes on Finnish instructions,
# and launches a web UI.
#
# Usage:
#   # Full pipeline (adjust --depth for model size):
#   bash runs/suomichat.sh
#
# Hardware: designed for a single GPU with 24-48 GB VRAM.
# Time: depth 12 → ~8h, depth 16 → ~24h, depth 20 → ~48h
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Set SUOMICHAT_BASE_DIR to control where data/checkpoints go.
# Default: ~/.cache/suomichat
export SUOMICHAT_BASE_DIR="${SUOMICHAT_BASE_DIR:-$HOME/.cache/suomichat}"

DEPTH="${SUOMICHAT_DEPTH:-12}"
NUM_SHARDS="${SUOMICHAT_NUM_SHARDS:-30}"

echo "============================================"
echo " SuomiChat — Finnish LLM Training Pipeline"
echo " depth=$DEPTH, shards=$NUM_SHARDS"
echo " data dir: $SUOMICHAT_BASE_DIR"
echo "============================================"

# Step 1: Download and prepare Finnish pretraining data
echo ">>> Step 1: Preparing Finnish data ($NUM_SHARDS shards)..."
python -m scripts.prepare_data --num-shards "$NUM_SHARDS"

# Step 2: Generate Finnish identity conversations
echo ">>> Step 2: Generating Finnish identity data..."
python scripts/gen_identity.py

# Step 3: Train Finnish BPE tokenizer
echo ">>> Step 3: Training Finnish tokenizer (vocab=32768)..."
python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768
python -m scripts.tok_eval || echo "(tok_eval may warn on non-English samples; continuing)"

# Step 4: Pretrain
echo ">>> Step 4: Pretraining (depth=$DEPTH)..."
python -m scripts.base_train \
    --depth="$DEPTH" \
    --device-batch-size=16 \
    --run=dummy \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1

# Step 5: Finnish SFT
echo ">>> Step 5: Finnish supervised fine-tuning..."
python -m scripts.chat_sft \
    --device-batch-size=8 \
    --max-seq-len=2048 \
    --run=dummy

# Step 6: Launch web UI
echo "============================================"
echo " Training complete!"
echo " To test: python -m scripts.chat_cli"
echo " Web UI:  python -m scripts.chat_web"
echo "============================================"
