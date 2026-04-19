# SuomiChat on Modal.com

Train a Finnish GPT on cloud H100 GPUs. ~$8 for d12, ~$44 for d20.

## Setup

```bash
pip install modal
modal setup  # one-time auth
```

## Step 1: Upload data

Pre-tokenized data must be on Modal's cloud storage first:

```bash
# Create volumes
modal volume create suomichat-data
modal volume create suomichat-checkpoints

# Upload tokenized shards + tokenizer (~14 GB, takes ~5 min)
modal volume put suomichat-data data-fi-v2/base_data_tokenized/ /base_data_tokenized/
modal volume put suomichat-data data-fi-v2/tokenizer/ /tokenizer/
modal volume put suomichat-data data-fi-v2/base_data_climbmix/ /base_data_climbmix/
```

## Step 2: Train

```bash
# d12 on 1x H100 (~2.5h, ~$8)
python modal/train.py train --depth 12

# d20 on 1x H100 (~10h, ~$33)
python modal/train.py train --depth 20

# d20 on 8x H100 (~1.7h, ~$44)
python modal/train.py train --depth 20 --gpus 8
```

## Step 3: Download results

```bash
modal volume get suomichat-checkpoints / ./my-checkpoints/
```

Then use locally:
```bash
export SUOMICHAT_BASE_DIR=./my-checkpoints
python -m scripts.chat_cli
```

## Cost estimates

| Config | GPU | Time | Cost |
|---|---|---|---|
| d12 ratio=8 | 1x H100 | ~2.5h | ~$8 |
| d20 ratio=8 | 1x H100 | ~10h | ~$33 |
| d20 ratio=8 | 8x H100 | ~1.7h | ~$44 |
| d20 ratio=20 | 8x H100 | ~4h | ~$104 |

Prices based on Modal's H100 SXM at ~$3.25/GPU/h.
