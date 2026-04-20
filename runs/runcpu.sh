#!/bin/bash

# Showing an example run for exercising some of the code paths on the CPU (or MPS on Macbooks)
# This script was last updated/tuned on Jan 17, 2026.

# Run as:
# bash runs/runcpu.sh

# NOTE: Training LLMs requires GPU compute and $$$. You will not get far on your Macbook.
# Think of this run as educational/fun demo, not something you should expect to work well.
# You may also want to run this script manually and one by one, copy pasting commands into your terminal.

# all the setup stuff
export SUOMICHAT_BASE_DIR="$HOME/.cache/suomichat"
mkdir -p $SUOMICHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# train tokenizer on ~2B characters (~34 seconds on my MacBook Pro M3 Max)
python -m scripts.prepare_data --num-shards 8
python -m scripts.tok_train --max-chars=2000000000
python -m scripts.tok_eval

# train a small 4 layer model
# I tuned this run to complete in about 30 minutes on my MacBook Pro M3 Max.
# To get better results, try increasing num_iterations, or get other ideas from your favorite LLM.
python -m scripts.base_train \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --sample-every=100 \
    --num-iterations=5000 \
    --run=$WANDB_RUN

# SFT (~10 minutes on my MacBook Pro M3 Max)
python scripts/gen_identity.py
python -m scripts.chat_sft \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --num-iterations=1500 \
    --run=$WANDB_RUN

# Chat with the model over CLI
# Tiny CPU-trained models won't say much coherent — try simple Finnish prompts.
# python -m scripts.chat_cli -p "Mikä on Suomen pääkaupunki?"

# Or use the web UI:
# python -m scripts.chat_web
