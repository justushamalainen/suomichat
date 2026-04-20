# SuomiChat

Finnish LLM training harness. Train a Finnish GPT from scratch on whatever
GPU you have — single 48 GB workstation card up to 8× H100 cluster.

> **Attribution**: SuomiChat is built on top of
> [karpathy/nanochat](https://github.com/karpathy/nanochat). The GPT
> architecture, Muon+AdamW optimizer, Flash Attention integration, FP8
> training path, inference engine, KV cache, and tool-execution loop are
> inherited from nanochat.

The code is adapted to work with Finnish datasets and evals.

## Quick start

```bash
# Setup
git clone https://github.com/justushamalainen/suomichat && cd suomichat
uv venv && uv sync --extra gpu --extra eval   # eval extra pulls FIN-bench harness
source .venv/bin/activate

# Train (single dial: --depth; everything else autodetected from your GPU)
SUOMICHAT_DEPTH=12 bash runs/suomichat.sh

# Chat with the trained model
python -m scripts.chat_cli
python -m scripts.chat_web              # web UI at :8000
python -m scripts.run_finbench --source sft
```

## Hardware autodetect

`suomichat/hardware.py` inspects the GPU at startup and picks
`device_batch_size`, `--fp8`, `nproc` from `(depth, hw)`:

| Hardware | Detected | d6 | d12 | d20 | d24 |
|---|---|---|---|---|---|
| RTX 6000 Ada (48GB) | SM 8.9, no FA3, no FP8 | bs=32 | bs=16 | bs=4 | bs=1 (warn — likely OOM) |
| 1× H100 (80GB) | SM 9.0, FA3, FP8 | bs=64 | bs=32 | bs=16 FP8 | bs=32 FP8 |
| 8× H100 (8×80GB) | SM 9.0, FA3, FP8 | bs=64 | bs=32 | bs=16 FP8 | bs=32 FP8 |

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
| 26 | ~1.9B | won't fit | ~24 hours (FP8) | ~3 hours (FP8) |

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
- Bring your own: any Finnish jsonl in conversation format works. Each
  line is a JSON list of `{role, content}` messages (optional `system`
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

## Running on Nebius (bare-metal H100)

```bash
# SkyPilot (recommended — declarative, autostop, multi-cloud)
pip install "skypilot-nightly[nebius]"
sky launch -c suomichat nebius/suomichat.yaml --env DEPTH=24
sky exec suomichat nebius/suomichat.yaml --env DEPTH=26    # reuse cluster
sky autostop suomichat -i 30                                # idle-shutdown

# Or bare-metal SSH
bash nebius/train.sh --depth 24
```

See `nebius/README.md` for provisioning, cost reference, Object Storage
mounts, and Blackwell/B300 notes.

## Evaluation

FIN-bench-v2 runs automatically at the end of `base_train` and `chat_sft`
(rank-sharded across all DDP ranks so the whole cluster evaluates in
parallel). Per-task scores and the FIN-CORE composite land in wandb and
the report.

Default suite is 6 MCF tasks (multiple-choice format — model picks
A/B/C/D). MCF is more robust for small models than CF (continuation
loglikelihood) because every choice is a single letter token, so
fluency/frequency bias doesn't dominate.

Standalone:

```bash
python -m scripts.run_finbench --source sft               # full FIN-CORE suite
python -m scripts.run_finbench --source sft --limit 100   # quick smoke
python -m scripts.run_finbench --tasks belebele_fin_mcf_fbv2_p0  # one task
```

Pass `--skip-finbench` to `base_train` / `chat_sft` to disable the
automatic post-training eval.

### Installing lm-evaluation-harness manually

If `--extra eval` didn't work for some reason:

```bash
git clone https://github.com/LumiOpen/lm-evaluation-harness ../lm-evaluation-harness
uv pip install -e ../lm-evaluation-harness
```

Pinned commit: `6e0a60abb74cf098bc5511b25caf66566586f9f0`.

## Environment

```bash
export SUOMICHAT_BASE_DIR=/path/to/data   # default: ~/.cache/suomichat
export SUOMICHAT_DTYPE=float32            # override auto-detected dtype
export SUOMICHAT_BS=N                     # override autodetected batch size
export SUOMICHAT_NPROC=N                  # override GPU count
```

## Inference

```bash
python -m scripts.chat_cli -g d26 -p "Mikä on Suomen pääkaupunki?"
python -m scripts.chat_web -g d26 --repetition-penalty 1.3   # web UI on :8000
```

`--repetition-penalty` (default 1.3) reduces the model's tendency to
ramble or repeat itself. Can also be set per-request in the
`/chat/completions` API.

## Data attribution

The Finnish datasets and evaluation tasks that make this project work
are maintained by others. Please cite them if you use the resulting
model or any of its components downstream.

**Pretraining data** — [Finnish-NLP](https://huggingface.co/Finnish-NLP)
(TurkuNLP / University of Turku and collaborators):
- [Fineweb2_Finnish_fineweb_edu_predicted](https://huggingface.co/datasets/Finnish-NLP/Fineweb2_Finnish_fineweb_edu_predicted)
- [wikipedia_20230501_fi_cleaned](https://huggingface.co/datasets/Finnish-NLP/wikipedia_20230501_fi_cleaned)
- [HPLT_1.2_fi_cleaned](https://huggingface.co/datasets/Finnish-NLP/HPLT_1.2_fi_cleaned)
- [mc4_fi_cleaned](https://huggingface.co/datasets/Finnish-NLP/mc4_fi_cleaned)
- [Reddit_fi_2006_2022](https://huggingface.co/datasets/Finnish-NLP/Reddit_fi_2006_2022)

**SFT data**:
- [datacrunch/finnish_alpaca](https://huggingface.co/datasets/datacrunch/finnish_alpaca) — 52K Finnish Alpaca-style instruction pairs (DataCrunch.io)
- [LumiOpen/poro2-instruction-collection](https://huggingface.co/datasets/LumiOpen/poro2-instruction-collection) — multilingual instruction collection, filtered to Finnish (LumiOpen / Silo AI)
- [Chaanim/finnish_math_reasoning](https://huggingface.co/datasets/Chaanim/finnish_math_reasoning) — ~100K Finnish math problems with chain-of-thought reasoning

**Evaluation** — [FIN-bench-v2](https://github.com/TurkuNLP/FIN-bench) and
its LumiOpen [lm-evaluation-harness](https://github.com/LumiOpen/lm-evaluation-harness)
fork. Tasks used in FIN-CORE: GoldenSwag-HT-FI, ScandiSent-FI,
SIB200-FI, Belebele-FIN, FIN-bench analogies, FIN-bench general
knowledge (authored/translated by TurkuNLP, LumiOpen, and the original
benchmark contributors — see each task's citation).

**Model architecture + training infrastructure** —
[karpathy/nanochat](https://github.com/karpathy/nanochat) (Andrej Karpathy).

## License

MIT.
