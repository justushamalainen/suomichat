# SuomiChat on Modal.com

Train a Finnish GPT on cloud H100 GPUs. Single dial: `--depth`. GPU count
is auto-picked (1× H100 for d<22, 8× H100 for d>=22), batch size is
auto-resolved by `suomichat/hardware.py` from the container's actual VRAM.

## Setup

```bash
pip install modal
modal setup            # one-time auth
```

## Step 1: Upload data

Pre-tokenized data must be on Modal's cloud storage first:

```bash
modal volume create suomichat-data
modal volume create suomichat-checkpoints

# Upload tokenized shards + tokenizer + raw shards (~14GB, ~5min)
modal volume put suomichat-data ~/.cache/suomichat/base_data_tokenized/ /base_data_tokenized/
modal volume put suomichat-data ~/.cache/suomichat/tokenizer/ /tokenizer/
modal volume put suomichat-data ~/.cache/suomichat/base_data_climbmix/ /base_data_climbmix/

# Upload curated SFT data if you have it
modal volume put suomichat-data ~/.cache/suomichat/sft_train.jsonl /sft_train.jsonl
```

## Step 2: Train

```bash
python modal/train.py train --depth 12          # 1× H100, ~$8
python modal/train.py train --depth 20          # 1× H100, ~$33 (default for d<22)
python modal/train.py train --depth 24          # 8× H100, ~$58 (default for d>=22)
python modal/train.py train --depth 20 --gpus 1 # force single GPU
python modal/train.py sft --depth 12            # SFT only on existing checkpoint
```

## Step 3: Download results

```bash
modal volume get suomichat-checkpoints / ./my-checkpoints/
```

Then use locally:

```bash
export SUOMICHAT_BASE_DIR=$(pwd)/my-checkpoints
python -m scripts.chat_cli
python -m scripts.run_finbench --source sft
```

## Cost estimates

| Config | GPU | Time | Cost |
|---|---|---|---|
| d12 ratio=8 | 1× H100 | ~1 h | ~$3 |
| d20 ratio=8 | 1× H100 | ~10 h | ~$33 |
| d24 ratio=8 (FP8) | 8× H100 | ~2.5 h | ~$58 |
| d24 ratio=20 (FP8) | 8× H100 | ~6 h | ~$140 |

Prices based on Modal H100 SXM at ~$3.25/GPU/h.

## Diagnostics

```bash
python modal/train.py diagnose       # quick container-import sanity check
```

## Notes

- Container image bakes in lm-evaluation-harness (LumiOpen fork) so
  end-of-training FIN-bench works out of the box
- Only `suomichat-data` and `suomichat-checkpoints` volumes persist;
  containers and GPUs are released the moment training finishes
- If the run fails mid-flight: `modal app list` to find orphaned apps
  and `modal app stop <id>` to clean up
