#!/bin/bash
# SuomiChat: Train a Finnish GPT from scratch.
#
# Full pipeline: download Finnish data → train tokenizer → pre-tokenize →
# pretrain GPT → Finnish SFT → ready to chat.
#
# Single dial: --depth (or SUOMICHAT_DEPTH env). Hardware is auto-detected
# and batch-size / FP8 / nproc are derived from (depth, hw) — see
# suomichat/hardware.py for the table. Override with SUOMICHAT_BS / SUOMICHAT_NPROC.
#
# Usage:
#   bash runs/suomichat.sh                        # default depth=12
#   SUOMICHAT_DEPTH=6 bash runs/suomichat.sh      # smoke test
#   SUOMICHAT_DEPTH=20 SUOMICHAT_NUM_SHARDS=139 bash runs/suomichat.sh
#
# Time estimates (single RTX 6000 Ada, 48GB):
#   depth 6,  8 shards:    ~30 minutes
#   depth 12, 30 shards:   ~6 hours
#   depth 20, 139 shards:  multi-day (use Modal H100 instead)
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SUOMICHAT_BASE_DIR="${SUOMICHAT_BASE_DIR:-$HOME/.cache/suomichat}"

DEPTH="${SUOMICHAT_DEPTH:-12}"
NUM_SHARDS="${SUOMICHAT_NUM_SHARDS:-30}"
TOK_WORKERS="${SUOMICHAT_TOK_WORKERS:-4}"

# Resolve hardware-aware training config: BS, FP8, MAX_SEQ_LEN, NPROC, RATIO, HW_SUMMARY
eval "$(python -m suomichat.hardware --depth "$DEPTH" --shell)"

echo "============================================"
echo " SuomiChat — Finnish LLM Training Pipeline"
echo " depth=$DEPTH, shards=$NUM_SHARDS, bs=$BS, nproc=$NPROC, fp8=$FP8"
echo " hw: $HW_SUMMARY"
echo " data dir: $SUOMICHAT_BASE_DIR"
echo "============================================"

# Step 1: Download and prepare Finnish pretraining data
echo ">>> Step 1/6: Preparing Finnish data ($NUM_SHARDS shards)..."
python -m scripts.prepare_data --num-shards "$NUM_SHARDS"

# Step 2: Generate Finnish identity conversations
echo ">>> Step 2/6: Generating Finnish identity data..."
python scripts/gen_identity.py

# Step 3: Train Finnish BPE tokenizer on the full mix
echo ">>> Step 3/6: Training Finnish tokenizer (vocab=32768)..."
python -m scripts.tok_train --vocab-size=32768
python -m scripts.tok_eval || echo "(tok_eval may warn on non-Finnish samples; continuing)"

# Step 4: Pre-tokenize all shards (helps multi-GPU runs; cheap one-time on single GPU)
echo ">>> Step 4/6: Pre-tokenizing shards ($TOK_WORKERS workers)..."
python -m scripts.tokenize_shards --workers "$TOK_WORKERS"

# Step 5: Pretrain
echo ">>> Step 5/6: Pretraining (depth=$DEPTH, bs=$BS, nproc=$NPROC)..."
PRETRAIN_FLAGS=(
    --depth="$DEPTH"
    --device-batch-size="$BS"
    --target-param-data-ratio="$RATIO"
    --max-seq-len="$MAX_SEQ_LEN"
    --run=dummy
    --sample-every=-1
    --save-every=-1
)
[ "$FP8" = "1" ] && PRETRAIN_FLAGS+=(--fp8)

if [ "$NPROC" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- "${PRETRAIN_FLAGS[@]}"
else
    python -m scripts.base_train "${PRETRAIN_FLAGS[@]}"
fi

# Step 6: Finnish SFT
echo ">>> Step 6/6: Finnish supervised fine-tuning..."
SFT_FLAGS=(
    --device-batch-size="$BS"
    --max-seq-len="$MAX_SEQ_LEN"
    --load-optimizer=0
    --run=dummy
)
if [ "$NPROC" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$NPROC" -m scripts.chat_sft -- "${SFT_FLAGS[@]}"
else
    python -m scripts.chat_sft "${SFT_FLAGS[@]}"
fi

echo "============================================"
echo " SuomiChat training complete!"
echo ""
echo " Test:    python -m scripts.chat_cli"
echo " Web UI:  python -m scripts.chat_web"
echo " Eval:    python -m scripts.run_finbench"
echo "============================================"
