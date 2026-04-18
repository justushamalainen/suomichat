# SuomiChat

Finnish LLM training harness. Train a Finnish GPT from scratch on a single GPU.

Fork of [karpathy/nanochat](https://github.com/karpathy/nanochat) — same architecture, optimizer, and training infrastructure, but configured for Finnish from the ground up: Finnish pretraining data, Finnish tokenizer, Finnish SFT, Finnish evaluation.

## Quick start

```bash
# Setup
git clone <this-repo> && cd suomichat
uv venv && uv sync --extra gpu
source .venv/bin/activate

# Train (single GPU, ~8h for depth 12)
bash runs/suomichat.sh

# Chat
python -m scripts.chat_cli
python -m scripts.chat_web  # web UI at :8000
```

## What's different from nanochat?

| Component | nanochat | suomichat |
|---|---|---|
| Pretraining data | ClimbMix (English, 400B tokens) | FineWeb2-fi + Wikipedia-fi + Reddit-fi |
| Tokenizer | English BPE (32K vocab) | Finnish BPE (32K vocab, ~4.8 chars/tok) |
| SFT data | SmolTalk + MMLU + GSM8K | FinnishAlpaca + Finnish identity |
| Evaluation | DCLM CORE (English) | FIN-bench-v2 (Finnish) |
| Identity | English assistant | SuomiChat — suomenkielinen assistentti |
| Web UI | English | Finnish |

Everything else (GPT architecture, Muon+AdamW optimizer, Flash Attention, FP8, distributed training, inference engine, KV cache, code execution) is identical to nanochat.

## Pipeline

```
scripts/prepare_data.py   → Download Finnish data, create parquet shards
scripts/gen_identity.py   → Generate Finnish identity conversations
scripts/tok_train.py      → Train Finnish BPE tokenizer
scripts/base_train.py     → Pretrain GPT on Finnish text
scripts/chat_sft.py       → Fine-tune on Finnish instructions
scripts/chat_cli.py       → CLI chat
scripts/chat_web.py       → Web UI
scripts/run_finbench.py   → FIN-bench-v2 evaluation
```

## Model scaling

Single dial: `--depth`. Everything else auto-scales.

| Depth | Params | VRAM (est.) | Training time (single GPU) |
|---|---|---|---|
| 6 | ~15M | 3 GB | ~2 min |
| 12 | ~286M | 17 GB | ~5 hours |
| 16 | ~500M | 30 GB | ~18 hours |
| 20 | ~896M | 45 GB | ~30 hours |

## Finnish data sources

Pretraining mix (configured in `scripts/prepare_data.py`):
- **70%** [Finnish-NLP/Fineweb2_Finnish_fineweb_edu_predicted](https://huggingface.co/datasets/Finnish-NLP/Fineweb2_Finnish_fineweb_edu_predicted) (edu score >= 2.0)
- **20%** [Finnish-NLP/wikipedia_20230501_fi_cleaned](https://huggingface.co/datasets/Finnish-NLP/wikipedia_20230501_fi_cleaned) (3x upsampled)
- **10%** [Finnish-NLP/Reddit_fi_2006_2022](https://huggingface.co/datasets/Finnish-NLP/Reddit_fi_2006_2022)

SFT data:
- [datacrunch/finnish_alpaca](https://huggingface.co/datasets/datacrunch/finnish_alpaca) (52K Finnish instructions)
- Hand-curated Finnish identity conversations

## Evaluation

FIN-bench-v2 integration via lm-evaluation-harness wrapper:

```bash
# Install LumiOpen's harness (has Finnish tasks)
pip install -e /path/to/lm-evaluation-harness

# Run Finnish CORE evaluation
python scripts/run_finbench.py --source sft
```

## Environment

```bash
export SUOMICHAT_BASE_DIR=/path/to/data  # default: ~/.cache/suomichat
export SUOMICHAT_DTYPE=float32           # override auto-detected dtype
export SUOMICHAT_SFT_MIXTURE=en          # use English SFT (default: fi)
```

## License

MIT (inherited from nanochat)

## Attribution

Based on [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy.
