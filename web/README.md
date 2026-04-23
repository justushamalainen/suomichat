# SuomiChat in the browser (WebGPU, from scratch)

A minimal-dependency WebGPU runtime for the suomichat model. **The goal
is understanding, not speed or download size.** Every line is meant to
be readable; we use the dumbest correct algorithm at every step. Once
you understand the whole stack, optimization is the next phase.

Target: **d12 SFT** (286M params, ~1.14 GB of fp32 weights).

## Why from scratch?

Existing browser LLM runtimes (Web-LLM, transformers.js) work great
with stock Llama-style architectures, but suomichat has a bunch of
small custom tweaks (smear gate, backout residual, per-layer scalars,
QK norm, ResFormer value embeddings, sliding-window attention with the
SSSL pattern). Porting all of those into someone else's runtime is
half the work of writing your own — and you learn ~10× more by
writing it.

## Layout

```
web/
├── README.md         (this file)
├── convert.py        Python: PyTorch checkpoint → weights.bin + tensors.json
├── server.py         tiny HTTP server (WebGPU needs http://, not file://)
├── index.html        UI + WebGPU init
├── runtime.js        main runtime: weight loading, scheduling, forward pass
├── tokenizer.js      placeholder tokenizer (real one comes in a later phase)
└── shaders/
    ├── embedding.wgsl
    ├── rmsnorm.wgsl  (later phase)
    ├── matmul.wgsl   (later phase)
    └── ...
```

## Build phases (each independently runnable)

| Phase | Goal | What's in it |
|---|---|---|
| **1** | Weights load + first shader runs | convert.py, runtime.js, embedding.wgsl |
| 2 | Single-layer forward matches PyTorch | rmsnorm, matmul, attention, mlp |
| 3 | Full d12 forward on 1 token | per-layer lambdas, smear, backout, value embeds, lm_head |
| 4 | Autoregressive generation | KV cache, sampling, stopping criteria |
| 5 | Real tokenizer | swap placeholder for huggingface/tokenizers WASM |
| 6 | UI polish | streaming token display, conversation state |

Each phase is a runnable browser page. After phase 1 you can already
inspect the model's first-layer embedding for a single token and
compare it against PyTorch.

## Running it

```bash
cd web
python convert.py --model-tag d12  # writes weights.bin + tensors.json
python server.py                   # serves localhost:8080
```

Then open `http://localhost:8080/` in Chrome (Firefox WebGPU still
flag-gated as of 2026-04). Open the JS console — phase 1 prints the
loaded tensors and the embedding for token id 100 (just so you can
see something happening).

## Verifying correctness

For each phase, run `python verify.py --layer N` (added in phase 2) —
it loads the model in PyTorch, runs the same input, and prints both
tensors side-by-side. If the WebGPU output matches to ~1e-3 tolerance,
phase passes. (fp32 → fp32 should match closely; if/when we go to fp16
we expect ~1e-2.)

## What's not here

- No quantization. Weights are fp32. Browser download = 1.14 GB. Fine
  for `localhost`; not OK for a public site.
- No tokenizer until phase 5. Phase 1-4 use a hand-curated token list.
- No optimization. The matmul shader is the dumbest possible nested
  loop — easy to read, ~30× slower than a tiled one. Improving it is
  Phase 7+ (out of scope for the learning project).
