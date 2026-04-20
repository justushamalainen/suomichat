# SuomiChat — Notes for Next Iteration

Lessons learned from the first training runs (d12, d20, d24). These should guide the next iteration.

## Training Configuration

### Sequence Length Mismatch → NaN
- **Critical**: If pretrain uses `--max-seq-len=1024` but SFT inherits this, SFT goes NaN.
- **Fix**: Always pass `--max-seq-len=2048` explicitly to `chat_sft.py`.
- **Better fix for next iteration**: Default seq_len should be 2048 everywhere. Only reduce for pretrain speed if needed, and always override back for SFT.

### Optimizer State Sharding
- Training on 8 GPUs saves per-rank optimizer files (`optim_*_rank0.pt` through `rank7.pt`).
- Loading on 1 GPU for SFT fails with shape mismatch.
- **Fix**: Use `--load-optimizer=0` when switching GPU count between pretrain and SFT.
- **Better fix**: SFT should always start with fresh optimizer (LR is reset anyway).

### Batch Size for 8×H100
- `bs=64` with d24 FP8 OOMs on 80GB H100 (uses 78/80 GB).
- `bs=32` works safely. Same micro-batch token count as nanochat's `bs=16, seq=2048`.
- For d24+, match nanochat exactly: `bs=16, seq=2048`.
- Try `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to potentially fit bs=64.

### Data Ratio
- ratio=8 (speed-optimized) produces reasonable models but undertrained for knowledge tasks.
- ratio=6 was even worse for knowledge.
- **Recommendation**: Use ratio=12 minimum for production models, ratio=8 for development iteration.
- Chinchilla-optimal (ratio=20) would require 2-3× more training time.

## Data Pipeline

### Pre-tokenization
- Pre-tokenized data gives NO speed improvement on single GPU (CPU tokenizer keeps up).
- **Does help** on 8×H100 where GPU is 4-8× faster — CPU tokenizer becomes the bottleneck.
- Worth doing for multi-GPU cloud training; skip for single-GPU local iteration.

### Finnish Data Sources (Ranked by Quality)
1. **Finnish Wikipedia** — highest quality, knowledge-dense. Upsample 5-10×.
2. **FineWeb2-fi (edu≥2.0)** — good web text, edu-score filtered. ~50B chars available.
3. **HPLT 1.2-fi** — long-form web content. 33B chars.
4. **mC4-fi** — Google web crawl. 55B chars but lower quality. HF cache dir structure differs from other Finnish-NLP datasets.
5. **Reddit-fi** — informal register. Short docs (194 chars avg). Good for diversity, not volume.

### Data Download
- HF streaming is slow (~1.4 MB/s per source). For large datasets, download to local cache first with `load_dataset()` (non-streaming), then repackage from local parquets.
- Parallel download of all sources: ~65 min for 300GB.
- Repackage from local parquets: ~12 min for 241 shards.
- **Avoid** `load_dataset()` for FineWeb2 without enough disk — the Arrow cache generation is very slow for 33M rows.

### Epoch-Looping for Small Datasets
- Wikipedia (4.9B chars) exhausts quickly at 35% weight. Epoch-looping with reshuffled order works well.
- Much better than inline repeat (`upsample=3` yields `doc,doc,doc,doc,...` — wasted training signal).
- At 35% weight with 139 shards, Wikipedia cycles ~7 epochs. Acceptable for sub-1B models.

## SFT Data

### Current State (v1)
- 346K rows total (300K conversations + 40K spelling + 6K identity).
- 6.6× more than the initial 52K FinnishAlpaca.
- **Missing**: Math (GSM8K equivalent), Knowledge (MMLU equivalent), Code.
- nanochat uses ~760K SFT rows — we're at 45% of that.

### SFT v2 (current — ready to train)
- 215K rows: 142K conversations (filtered), 33K identity, 40K spelling
- **System prompt** on every row: "Olet SuomiChat, suomenkielinen tekoälyavustaja."
- **Repetition filter**: 149K rows dropped (50% of original conversations were repetitive!)
- **Response truncation**: 16K responses capped at 1500 chars
- **Diverse identity**: 200 unique conversations from Sonnet + 2K seeds × 15 epochs
- File: `/home/janitor/llm-training/suomichat-sft/data/final/suomichat_sft_v2.train.jsonl`
- **Not yet uploaded to Modal** — needs `modal volume put suomichat-data`

### SFT v2 key finding
- FIN-CORE scores were IDENTICAL between 52K and 346K SFT (0.387) — the benchmark tests base model knowledge via loglikelihood, not SFT chat quality
- Need conversational eval metrics (not just FIN-bench) to measure SFT improvements
- Identity was lost with 300K conversations drowning 6K identity rows — v2 fixes with 33K identity + system prompts

### Priorities for v3
1. **More conversation data** — Poro2 has 1.4M rows but only ~10% are Finnish. Better Finnish filtering (fastText + KenLM) would yield more.
2. **Finnish MMLU** — Translate the 100K MMLU auxiliary_train split to Finnish. DeepL cost: ~€25.
3. **Finnish GSM8K** — Translate 7.5K math problems with numeric preservation. DeepL cost: ~€3.
4. **Code + Math reasoning** — These are language-agnostic. Consider including English code/math data (like nanochat does) even in the Finnish model.
5. **Finnish SpellingBee** — Already synthetic (40K from Kotus). Could expand with compound words and morphological tasks unique to Finnish.

### Identity
- 90 seed conversations expanded to 2K with template variation.
- **Problem**: Model still loops/repeats after 2-3 sentences.
- **Fix**: Need repetition penalty at inference, or more diverse training data to prevent loops.

## Evaluation

### English CORE/ChatCORE is Useless
- All English eval tasks (CORE, ChatCORE) score near random for a Finnish model.
- **Remove** from suomichat entirely. Replace with FIN-bench-v2.
- This saves ~5 min per training run.

### FIN-bench-v2 Integration
- Custom `lm_eval` wrapper (`scripts/lm_eval_wrapper.py`) works with nanochat's checkpoint format.
- **Current FIN-CORE** uses 7 tasks. Should add more knowledge tasks (Finnish MMLU, TyDiQA-fi, BeleBele-fi) for better GPT-2 comparability.
- SIB200-fi scores are unstable across runs — consider replacing or downweighting.

### Benchmark Results Summary

| Model | Params | Data | SFT | FIN-CORE | val_bpb |
|---|---|---|---|---|---|
| d12 v1 | 286M | 3 sources, ratio=12 | 52K | 0.405 | 0.892 |
| d20 v1 | 897M | 3 sources, ratio=6 | 52K | 0.405 | 0.807 |
| d24 Modal | 1.38B | 5 sources, ratio=8 | 52K | 0.387 | ~0.65 |
| d24 SFT v2 | 1.38B | 5 sources, ratio=8 | 346K | TBD | TBD |

## Modal.com / Cloud Training

### What Works
- `modal volume put/get` for data transfer.
- Local code via `Image.add_local_dir()` — no git repo needed.
- `torchrun` needs `--` separator before script args (otherwise `--run=dummy` conflicts with `--run-path`).

### What to Fix
- Stale containers spawn when local process reconnects. Kill orphan local processes.
- `subprocess.call()` streams output to Modal logs but not to local terminal.
- `app.run()` + `.remote()` buffers all output until function returns — bad for long-running jobs.
- Consider using `modal deploy` + webhooks for production runs.
- H100 SXM vs H200: Modal may assign either. H200 has 141GB VRAM (fits bs=64 for d24).

### Cost Reference
| Config | Time | GPU-hours | Cost |
|---|---|---|---|
| d24 pretrain 8×H100 | ~2h | 16 | ~$52 |
| d24 SFT 1×H100 (346K) | ~32 min | 0.5 | ~$1.60 |
| d26 pretrain 1×H100 | ~20h | 20 | ~$65 |

## Architecture / Code

### suomichat vs nanochat Diff
- Package renamed `nanochat → suomichat`.
- `dataset.py` — Finnish data sources (no ClimbMix download).
- `chat_sft.py` — Loads `suomichat_sft_v1.train.jsonl` if present, falls back to FinnishAlpaca.
- `dataloader.py` — Auto-detects pre-tokenized shards (`base_data_tokenized/`).
- `modal/train.py` — Cloud training with Modal.com.
- `scripts/lm_eval_wrapper.py` — FIN-bench-v2 evaluation.
- `scripts/prepare_data.py` — Finnish data download + repackage.
- `scripts/tokenize_shards.py` — Pre-tokenization.

### TODO for Next Iteration
- [x] Remove English CORE/ChatCORE from `base_train.py` and `chat_sft.py` (Phase 1)
- [x] Wire FIN-bench directly into training loop (replace CORE eval) (Phase 2)
- [x] Add `--skip-finbench` flag for faster training iteration (Phase 2)
- [ ] Fix Modal output streaming (use `modal deploy` or webhooks)
- [ ] Add repetition penalty to inference (`chat_cli.py`, `chat_web.py`)
- [ ] Support checkpoint resume after Modal preemption
- [ ] Finnish-RL on Finnish math/reasoning rewards (chat_rl.py was deleted in
      Phase 1; would need a fresh Finnish-flavored implementation)
