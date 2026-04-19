#!/bin/bash
# SuomiChat: Train a Finnish GPT from scratch.
#
# Full pipeline: download Finnish data → train tokenizer → pre-tokenize →
# pretrain GPT → Finnish SFT → ready to chat.
#
# Usage:
#   bash runs/suomichat.sh                          # defaults: depth=12, 30 shards
#   SUOMICHAT_DEPTH=20 SUOMICHAT_NUM_SHARDS=139 bash runs/suomichat.sh  # d20 speed-optimized
#
# Hardware: single GPU with 24-48 GB VRAM.
# Time estimates (single RTX 6000 Ada / A6000):
#   depth 12, 30 shards:   ~8 hours total
#   depth 20, 139 shards:  ~3.5 days total
#   depth 20 on 8xH100:    ~2 hours total
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SUOMICHAT_BASE_DIR="${SUOMICHAT_BASE_DIR:-$HOME/.cache/suomichat}"

DEPTH="${SUOMICHAT_DEPTH:-12}"
NUM_SHARDS="${SUOMICHAT_NUM_SHARDS:-30}"
TOK_WORKERS="${SUOMICHAT_TOK_WORKERS:-4}"

echo "============================================"
echo " SuomiChat — Finnish LLM Training Pipeline"
echo " depth=$DEPTH, shards=$NUM_SHARDS"
echo " data dir: $SUOMICHAT_BASE_DIR"
echo "============================================"

# Step 1: Download and prepare Finnish pretraining data
echo ">>> Step 1/6: Preparing Finnish data ($NUM_SHARDS shards)..."
python -m scripts.prepare_data --num-shards "$NUM_SHARDS"

# Step 2: Generate Finnish identity conversations
echo ">>> Step 2/6: Generating Finnish identity data..."
python scripts/gen_identity.py

# Step 3: Train Finnish BPE tokenizer on the full mix
echo ">>> Step 3/6: Training Finnish tokenizer (vocab=32768, 10B chars)..."
python -m scripts.tok_train --vocab-size=32768
python -m scripts.tok_eval || echo "(tok_eval may warn on non-Finnish samples; continuing)"

# Step 4: Pre-tokenize all shards (one-time CPU cost, speeds up training)
echo ">>> Step 4/6: Pre-tokenizing shards ($TOK_WORKERS workers)..."
python -m scripts.tokenize_shards --workers "$TOK_WORKERS"

# Step 5: Pretrain
echo ">>> Step 5/6: Pretraining (depth=$DEPTH)..."
python -m scripts.base_train \
    --depth="$DEPTH" \
    --device-batch-size=16 \
    --target-param-data-ratio=8 \
    --run=dummy \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1

# Step 6: Finnish SFT
echo ">>> Step 6/6: Finnish supervised fine-tuning..."
python -m scripts.chat_sft \
    --device-batch-size=8 \
    --max-seq-len=2048 \
    --run=dummy

echo "============================================"
echo " SuomiChat training complete!"
echo ""
echo " Test:    python -m scripts.chat_cli"
echo " Web UI:  python -m scripts.chat_web"
echo " Eval:    python scripts/run_finbench.py"
echo "============================================"
