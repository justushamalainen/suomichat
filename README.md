# SuomiChat

Finnish LLM training harness. Train a Finnish GPT from scratch on whatever
GPU you have — single 48GB workstation card up to 8× H100 cluster.

Fork of [karpathy/nanochat](https://github.com/karpathy/nanochat) — same
architecture, optimizer, and training infrastructure, configured Finnish
from the ground up: Finnish pretraining data, Finnish tokenizer, Finnish
SFT, Finnish evaluation.

## Quick start

```bash
# Setup
git clone <this-repo> && cd suomichat
uv venv && uv sync --extra gpu --extra eval   # eval extra pulls FIN-bench harness
source .venv/bin/activate

# Train (single dial: --depth; everything else autodetected from your GPU)
SUOMICHAT_DEPTH=12 bash runs/suomichat.sh

# Chat with the trained model
python -m scripts.chat_cli
python -m scripts.chat_web              # web UI at :8000
python -m scripts.run_finbench --source sft
```

## What's different from nanochat?

| Component | nanochat | suomichat |
|---|---|---|
| Pretraining data | ClimbMix (English, 400B tokens) | FineWeb2-fi + Wikipedia-fi + HPLT-fi + mC4-fi + Reddit-fi |
| Tokenizer | English BPE (32K vocab) | Finnish BPE (32K vocab, ~4.8 chars/tok) |
| SFT data | SmolTalk + MMLU + GSM8K | FinnishAlpaca + Finnish identity (+ optional curated v2 mixture) |
| Evaluation | DCLM CORE (English) | FIN-bench-v2 (Finnish) |
| Identity | English assistant | SuomiChat — suomenkielinen assistentti |
| Web UI | English | Finnish |
| Knobs | many | one (`--depth`); rest auto-detected from GPU |

Everything else (GPT architecture, Muon+AdamW optimizer, Flash Attention,
FP8, distributed training, inference engine, KV cache, code execution) is
identical to nanochat.

## Hardware autodetect

`suomichat/hardware.py` inspects the GPU at startup and picks
`device_batch_size`, `--fp8`, `nproc` from `(depth, hw)`:

| Hardware | Detected | d6 | d12 | d20 | d24 |
|---|---|---|---|---|---|
| RTX 6000 Ada (48GB) | SM 8.9, no FA3, no FP8 | bs=32 | bs=16 | bs=4 | bs=1 (warn — likely OOM) |
| 1× H100 (80GB) | SM 9.0, FA3, FP8 | bs=64 | bs=32 | bs=16 | bs=16 +FP8 |
| 8× H100 (8×80GB) | SM 9.0, FA3, FP8 | bs=64 | bs=32 | bs=16 | bs=16 +FP8 |

Override with `SUOMICHAT_BS=N` or `SUOMICHAT_NPROC=N`.

## Pipeline

```
scripts/prepare_data.py     → Download Finnish data, create parquet shards
scripts/gen_identity.py     → Generate Finnish identity conversations
scripts/tok_train.py        → Train Finnish BPE tokenizer
scripts/tokenize_shards.py  → Pre-tokenize parquet shards (helps multi-GPU)
scripts/base_train.py       → Pretrain GPT on Finnish text
scripts/chat_sft.py         → Fine-tune on Finnish instructions
scripts/run_finbench.py     → FIN-bench-v2 evaluation (also runs end-of-train automatically)
scripts/chat_cli.py         → CLI chat
scripts/chat_web.py         → Web UI
suomichat/hardware.py       → Hardware autodetect + recommend_config(depth)
```

## Model scaling

| Depth | Params | RTX 6000 Ada | 1× H100 | 8× H100 |
|---|---|---|---|---|
| 6 | ~15M | ~30 min | ~5 min | ~2 min |
| 12 | ~286M | ~6 hours | ~1 hour | ~15 min |
| 20 | ~896M | multi-day | ~10 hours | ~2 hours |
| 24 | ~1.4B | won't fit | ~16 hours (FP8) | ~2.5 hours (FP8) |

Pick the deepest model your hardware tier can handle. The wrapper picks
batch size and parallelism for you.

## Finnish data sources

Pretraining mix (configured in `scripts/prepare_data.py`):
- **35%** [Finnish-NLP/Fineweb2_Finnish_fineweb_edu_predicted](https://huggingface.co/datasets/Finnish-NLP/Fineweb2_Finnish_fineweb_edu_predicted) (edu ≥ 2.0)
- **35%** [Finnish-NLP/wikipedia_20230501_fi_cleaned](https://huggingface.co/datasets/Finnish-NLP/wikipedia_20230501_fi_cleaned) (epoch-looped for small dataset)
- **15%** [Finnish-NLP/HPLT_1.2_fi_cleaned](https://huggingface.co/datasets/Finnish-NLP/HPLT_1.2_fi_cleaned)
- **10%** [Finnish-NLP/mc4_fi_cleaned](https://huggingface.co/datasets/Finnish-NLP/mc4_fi_cleaned)
- **5%** [Finnish-NLP/Reddit_fi_2006_2022](https://huggingface.co/datasets/Finnish-NLP/Reddit_fi_2006_2022)

SFT data:
- Default: [datacrunch/finnish_alpaca](https://huggingface.co/datasets/datacrunch/finnish_alpaca) (52K rows) + Finnish identity conversations from `scripts/gen_identity.py`
- Bring your own: any Finnish jsonl in suomichat conversation format works.
  Each line is a JSON list of `{role, content}` messages (optional `system`
  at position 0). Place at `$SUOMICHAT_BASE_DIR/sft_train.jsonl` or pass
  `--sft-file=PATH` to `chat_sft`.

## Running on Modal.com (cloud H100)

```bash
modal setup                                          # one-time auth
python modal/train.py train --depth 12               # 1× H100 (default for d<22)
python modal/train.py train --depth 24               # 8× H100 (default for d>=22)
python modal/train.py train --depth 20 --gpus 1      # force single GPU
modal volume get suomichat-checkpoints / ./out/      # download
```

See `modal/README.md` for cost estimates and data upload instructions.

## Evaluation

FIN-bench-v2 runs automatically at the end of `base_train` and `chat_sft`.
Per-task scores and the FIN-CORE composite land in wandb and the report.
Pass `--skip-finbench` to disable for fast iteration.

Standalone:

```bash
python -m scripts.run_finbench --source sft               # full FIN-CORE suite
python -m scripts.run_finbench --source sft --limit 100   # quick smoke
python -m scripts.run_finbench --tasks belebele_fin_cf_fbv2_p0  # one task
```

### Installing lm-evaluation-harness manually

If `--extra eval` didn't work for some reason:

```bash
git clone https://github.com/LumiOpen/lm-evaluation-harness ../lm-evaluation-harness
uv pip install -e ../lm-evaluation-harness
```

## Environment

```bash
export SUOMICHAT_BASE_DIR=/path/to/data   # default: ~/.cache/suomichat
export SUOMICHAT_DTYPE=float32            # override auto-detected dtype
export SUOMICHAT_BS=N                     # override autodetected batch size
export SUOMICHAT_NPROC=N                  # override GPU count
```

## License

MIT (inherited from nanochat)

## Attribution

Based on [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy.
