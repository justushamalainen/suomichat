#!/bin/bash
# SuomiChat on Nebius (bare-metal H100 VMs).
#
# Workflow:
#   1. Provision a VM with GPUs via `nebius` CLI or Nebius web console
#      (see nebius/README.md for details). Typical choice:
#        - 1× H100 80GB for d12-d20
#        - 8× H100 80GB or 8× H200 141GB for d20-d26
#   2. SSH in.
#   3. git clone <this repo> && cd suomichat
#   4. bash nebius/train.sh --depth 24
#
# The script auto-detects hardware via suomichat/hardware.py and picks
# batch size / FP8 / nproc. Same flow as the local runs/suomichat.sh,
# just with cloud-sized defaults and data upload helpers.

set -euo pipefail

cd "$(dirname "$0")/.."

# ---------------- CLI args ----------------
DEPTH=12
NUM_SHARDS=""        # empty = depth-appropriate default
SFT_FILE=""          # empty = use $SUOMICHAT_BASE_DIR/sft_train.jsonl if present
SKIP_DATA=0          # 1 if data is already present (e.g. from prior run)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --depth)        DEPTH="$2"; shift 2;;
        --num-shards)   NUM_SHARDS="$2"; shift 2;;
        --sft-file)     SFT_FILE="$2"; shift 2;;
        --skip-data)    SKIP_DATA=1; shift;;
        -h|--help)
            head -n 20 "$0" | tail -n 18 | sed 's/^# \?//'
            exit 0;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

# Depth-appropriate shard default for Chinchilla-ish training
if [[ -z "$NUM_SHARDS" ]]; then
    case "$DEPTH" in
        6)  NUM_SHARDS=8 ;;
        12) NUM_SHARDS=30 ;;
        16) NUM_SHARDS=60 ;;
        20) NUM_SHARDS=139 ;;
        24|26) NUM_SHARDS=241 ;;
        *)  NUM_SHARDS=30 ;;
    esac
fi

# ---------------- Environment ----------------
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SUOMICHAT_BASE_DIR="${SUOMICHAT_BASE_DIR:-$HOME/.cache/suomichat}"
mkdir -p "$SUOMICHAT_BASE_DIR"

# ---------------- Venv setup (one-time on fresh VM) ----------------
if [[ ! -d .venv ]]; then
    command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv venv
    uv sync --extra gpu --extra eval
fi
source .venv/bin/activate

# ---------------- Hardware autoconfig ----------------
eval "$(python -m suomichat.hardware --depth "$DEPTH" --shell)"
echo "============================================"
echo " SuomiChat on Nebius"
echo " depth=$DEPTH | bs=$BS | fp8=$FP8 | nproc=$NPROC | shards=$NUM_SHARDS"
echo " hw: $HW_SUMMARY"
echo " data: $SUOMICHAT_BASE_DIR"
echo "============================================"

# ---------------- Data pipeline ----------------
if [[ "$SKIP_DATA" -eq 0 ]]; then
    echo ">>> Step 1/6: Preparing Finnish data ($NUM_SHARDS shards)..."
    python -m scripts.prepare_data --num-shards "$NUM_SHARDS"

    echo ">>> Step 2/6: Generating Finnish identity data..."
    python scripts/gen_identity.py

    echo ">>> Step 3/6: Training Finnish tokenizer..."
    python -m scripts.tok_train --vocab-size=32768
    python -m scripts.tok_eval || echo "(tok_eval may warn; continuing)"

    echo ">>> Step 4/6: Pre-tokenizing shards..."
    python -m scripts.tokenize_shards --workers "${TOK_WORKERS:-8}"
else
    echo ">>> Steps 1-4 skipped (--skip-data)"
fi

# ---------------- SFT data (optional custom file) ----------------
if [[ -n "$SFT_FILE" && "$SFT_FILE" != "$SUOMICHAT_BASE_DIR/sft_train.jsonl" ]]; then
    ln -sf "$SFT_FILE" "$SUOMICHAT_BASE_DIR/sft_train.jsonl"
fi

# ---------------- Pretrain ----------------
echo ">>> Step 5/6: Pretraining (depth=$DEPTH, bs=$BS, nproc=$NPROC, fp8=$FP8)..."
PRETRAIN_FLAGS=(
    --depth="$DEPTH"
    --device-batch-size="$BS"
    --target-param-data-ratio="$RATIO"
    --max-seq-len="$MAX_SEQ_LEN"
    --run="${WANDB_RUN:-dummy}"
    --sample-every=-1
    --save-every=-1
)
[[ "$FP8" = "1" ]] && PRETRAIN_FLAGS+=(--fp8)

if [[ "$NPROC" -gt 1 ]]; then
    torchrun --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- "${PRETRAIN_FLAGS[@]}"
else
    python -m scripts.base_train "${PRETRAIN_FLAGS[@]}"
fi

# ---------------- SFT ----------------
echo ">>> Step 6/6: Finnish SFT..."
SFT_FLAGS=(
    --device-batch-size="$BS"
    --max-seq-len="$MAX_SEQ_LEN"
    --load-optimizer=0
    --run="${WANDB_RUN:-dummy}"
)
if [[ "$NPROC" -gt 1 ]]; then
    torchrun --standalone --nproc_per_node="$NPROC" -m scripts.chat_sft -- "${SFT_FLAGS[@]}"
else
    python -m scripts.chat_sft "${SFT_FLAGS[@]}"
fi

echo "============================================"
echo " Done! Next steps:"
echo "   chat:     python -m scripts.chat_cli"
echo "   eval:     python -m scripts.run_finbench --source sft"
echo "   download: rsync -avz \$SUOMICHAT_BASE_DIR/chatsft_checkpoints/ local:/path/"
echo "============================================"
