#!/bin/bash

# Speedrun: train a Finnish GPT-2-grade LLM on a single 8×H100 node.
# Pretraining + SFT, ~3 hours end to end.

# 1) Simplest launch:
#      bash runs/speedrun.sh
# 2) In a screen session (run is long):
#      screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
# 3) With wandb logging:
#      WANDB_RUN=speedrun bash runs/speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/suomichat
export OMP_NUM_THREADS=1
export SUOMICHAT_BASE_DIR="$HOME/.cache/suomichat"
mkdir -p $SUOMICHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup (optional). Run `wandb login` once, then set WANDB_RUN.
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy  # special-cased: no wandb logging
fi

# -----------------------------------------------------------------------------
# Reset the report directory
python -m suomichat.report reset

# -----------------------------------------------------------------------------
# Finnish data + tokenizer
# Download an initial 8 shards (~2B chars) so tokenizer training has data.
python -m scripts.prepare_data --num-shards 8
# Kick off the rest of the shards in the background while tokenizer trains.
# d24 GPT-2-grade pretraining wants ~150 shards; bump for headroom.
python -m scripts.prepare_data --num-shards 170 &
DATASET_DOWNLOAD_PID=$!
# Train Finnish BPE (vocab=32768) on the initial shards.
python -m scripts.tok_train
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Pretrain (8×H100, d24, FP8)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 \
    --run=$WANDB_RUN

# -----------------------------------------------------------------------------
# SFT
# Generate Finnish identity conversations (no S3 download needed).
python scripts/gen_identity.py

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --device-batch-size=16 --max-seq-len=2048 --load-optimizer=0 \
    --run=$WANDB_RUN

# -----------------------------------------------------------------------------
# Try the model:
#   python -m scripts.chat_cli                    # CLI chat
#   python -m scripts.chat_web                    # web UI on :8000
#   python scripts/run_finbench.py --source sft   # FIN-bench eval

# -----------------------------------------------------------------------------
# Generate the report
python -m suomichat.report generate
