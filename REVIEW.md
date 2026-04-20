# Pre-merge Review: `finnish-only-refactor` → `master`

Independent review by code-review subagent on 2026-04-19, after the
36-commit refactor finished. Read this before deciding to push or merge.

## Branch summary

36 commits, +1167 / -2254 lines across 31 files. Strips all English-task
code (ARC/MMLU/GSM8K/HumanEval/SmolTalk/SpellingBee, ChatCORE/CORE evals,
`chat_rl`), wires the LumiOpen FIN-bench harness directly into both
`base_train` and `chat_sft`, introduces `suomichat/hardware.py` so the
user's only knob is `--depth` (everything else — batch size, FP8, nproc
— is derived from `(depth, detected_hw)`), rewrites README +
modal/README around that flow, and adds 478 lines of `program.md`
documenting the entire run. Net effect: a smaller, mono-purpose Finnish
trainer with a one-knob UX validated end-to-end on local hardware.

## What works (verified)

- **d6 smoke pretrain** — 200 iters, 2.23 min, val_bpb 3.27→1.64,
  ~360K tok/sec, peak 10.4 GB. Logs in `/tmp/suomichat-runs/d6_smoke_pretrain.log`,
  checkpoint at `data-fi-v2/base_checkpoints/d6/model_000200.pt`.
- **d6 SFT smoke** — FinnishAlpaca fallback path runs, 17 steps,
  val_bpb 1.60→1.01. Confirms graceful fallback when `sft_train.jsonl`
  symlink absent (`d6_smoke_sft.log`).
- **d12 throughput sweep** — bs=8/16/32 all yield ~90K tok/sec, ~22%
  MFU. Compute-bound; autoconfig's bs=16 is at the optimum.
- **d12 SFT v2 (215K rows) + FIN-bench** — 90 steps in 9.61 min,
  FIN-CORE 0.334, full per-task table in program.md. CustomJSON
  system-message support (commit 8bdfa0f) needed for v2 mixture.
- **d12 SFT v2+math (220K rows)** — math format learned ("2+2 = Vastaus
  on 4." vs prior "2+2=2"), FIN-CORE 0.329 (within noise).
- **lm_eval wrapper in-process** — `run_finbench()` callable from both
  training scripts with rank-0 guard + barrier + ImportError fallback.
- **Hardware autodetect** — `suomichat/hardware.py:_BS_TABLE` is the
  authoritative bs source; eval'd into bash via `runs/suomichat.sh` and
  resolved inside the Modal container at `modal/train.py`.

## Known limitations (by design)

- **No FA3/FP8 on RTX 6000 Ada (SM 8.9)**. Trainer prints a loud warning
  in both base_train and chat_sft.
- **d24 won't fit on Ada at any bs** — table marks 0 and `recommend_config`
  falls back to bs=1 with `WARNING:` (`hardware.py:122-125`).
- **Math ceiling at d12**: model learned format but cannot compute
  larger arithmetic. Either go d20+ or add tool-calling.
- **FIN-CORE doesn't differentiate SFT mixes** — confirmed in v2 vs
  v2+math = 0.334 vs 0.329. The eval is a base-model loglikelihood probe.
- **`--max-seq-len=2048` is locked** in `hardware.py:142` because lower
  triggers SFT-inheritance NaN (NOTES.md:8-10).

## Risk assessment for fresh-user merge

- **`~/.cache/suomichat/sft_train.jsonl` is a symlink to your local
  `suomichat-sft/...` only.** A fresh clone has nothing there, hits the
  FinnishAlpaca fallback automatically — that path works. README.md
  updated to remove the broken `[suomichat-sft](https://github.com/...)`
  placeholder.
- **`modal/README.md` references three upload paths** including
  `base_data_climbmix` (historical name kept for nanochat compatibility,
  documented in `dataset.py:20-22`). Functionally fine but visually
  confusing — consider a one-line note if it ever bothers a user.
- **`runs/speedrun.sh` and `runs/runcpu.sh`** are reference scripts but
  neither passes `--skip-finbench`; on a fresh box without lm_eval they
  print "Skipping FIN-bench: ..." and continue. Graceful path exists,
  not a bug.
- **`tasks/poro2_fi.py:FinnishAlpaca`** still hardcodes the
  `datacrunch/finnish_alpaca` HF path. Fresh user with no HF cache
  triggers a download on first SFT run (~150 MB).
- **NOTES.md TODO list** updated: 3 items marked done (CORE removal,
  FIN-bench wiring, --skip-eval/--skip-finbench), 4 items still open
  (Modal output streaming, repetition penalty, checkpoint resume,
  Finnish-RL).
- **`dev/` directory** still contains 4 English nanochat artifacts
  (estimate_gpt3_core.ipynb, LEADERBOARD.md, repackage_data_reference.py,
  gen_synthetic_data.py, scaling_analysis.ipynb) — kept as historical
  record per Phase 5 decision; safe to delete if undesired.
- **No CI** — `tests/` exists but no evidence tests were run. Skim them
  before push.

## Pre-merge checklist

1. ~~Fix the dead `[suomichat-sft](https://github.com/...)` URL in
   README.md~~ — DONE on 2026-04-19 in this same review pass.
2. ~~Update NOTES.md TODO list to mark completed items~~ — DONE.
3. **Decide the `sft_train.jsonl` symlink question.** Symlink is in
   `~/.cache/`, not the repo. Documented decision in README:
   bring-your-own jsonl or use the FinnishAlpaca fallback.
4. **Skim `program.md`** end-to-end for accuracy. Most critical for a
   future reader: the "Phase 4 deferred" decision and the math
   experiment results.
5. **Optional**: delete the `dev/` artifacts (English nanochat reference
   material). Never referenced by suomichat code.
6. ~~**Run `python -m pytest tests/`** before push~~ — DONE on
   2026-04-20. 13 passed, 10 FA3-specific tests skipped (correct on
   non-Hopper). Install with `uv sync --group dev` first; pytest is in
   the dev group, not in the default install.
7. **Push**: `git push origin finnish-only-refactor` and open the PR.
   Don't merge directly to master without the PR review pass — 2254
   lines deleted is enough that a second look is cheap insurance.

## Recommended follow-up ranking

The four open math/knowledge directions, ranked by cost vs value:

**1. Ship as-is.** The branch is in a clean, documented state with
end-to-end evidence. Merging now locks in the win and stops further
state drift. Math is a known gap but doesn't block the harness work.

**2. d20 on Modal (~$33, ~10h).** Cheap relative to the 36-commit
refactor and tests two things: (a) the autoconfig table for 1× H100,
(b) whether bigger params actually fix the embarrassing "47+23=7" case.
If d20 still gets it wrong, that proves model size alone won't fix it
and option 4 becomes the right next bet.

**3. GSM8K-fi with reasoning (~€3 + ½ day DeepL).** Translate 7K
problems with chain-of-thought. Lower-risk than tool-calling and
improves both d12 and d20+.

**4. Tool-calling SFT data (1-2 days dev).** Synthesize Finnish
examples that delegate math to `<|python_*|>` blocks. Highest
potential ceiling but also the most novel work — defer until 2 and 3
hit a clear plateau.

---

*Review generated by independent subagent. Files referenced are at
`/home/janitor/llm-training/suomichat`. Logs at `/tmp/suomichat-runs/`.*
