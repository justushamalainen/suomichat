# WebGPU runtime — total plan

Single source of truth. One phase = one branch = one PR. Don't branch from
any other phase (keep them linear on main). Don't add phases outside this
list without a strong reason — the goal is to finish, not to explore.

## Scope

Run the existing **suomichat d12 SFT** model in the browser via WebGPU,
with autoregressive generation and a basic chat UI. fp32 weights (no
quantization, no size optimization). Correctness over speed at every step.

## Non-goals

- Quantization (INT8/INT4) — deferred
- Performance optimization (tiled matmul, fused kernels) — deferred
- Mobile / Safari support — deferred; Chromium only
- Training in browser — never; inference only
- Multi-GPU in browser — never (browser runtimes are single-adapter)
- Other suomichat model sizes (d6, d26) — only d12 targeted; trivial to swap

## Phase table

Each phase lands on `web/phase-<N>-<name>` branch, gets its own PR.
Acceptance criteria: `cd web && python test_runtime.py` shows all tests green.

| # | Branch | Deliverable | Status |
|---|---|---|---|
| 1 | `web/phase-1-skeleton` (in commit 78dcdb3) | WebGPU init, weight loader, embedding shader, headless test harness | ✅ merged |
| 2 | `web/phase-2-rmsnorm` (this branch) | RMSNorm shader, `rmsnorm()` export, `test_rmsnorm` | 🔨 in PR |
| 3 | `web/phase-3-matmul` | Naive matmul shader (one thread per output element), `matmul()` export, `test_matmul` on random A (M×K) × B (K×N) | next |
| 4 | `web/phase-4-attn-single-layer` | Attention block forward for **T=1** (no KV cache yet): c_q/c_k/c_v projections, RoPE apply, QK norm, scaled dot product with causal mask, c_proj. Test against PyTorch's single-layer attention output on a random hidden-state input. | |
| 5 | `web/phase-5-mlp` | MLP block: c_fc matmul → `relu²` shader → c_proj matmul. Test against PyTorch MLP on random input. | |
| 6 | `web/phase-6-block-forward` | Wire one full Block: `resid_lambdas[i] * x + x0_lambdas[i] * x0` → attn + residual → mlp + residual. Test against `model.transformer.h[0].forward(...)` on random input. | |
| 7 | `web/phase-7-value-embeds` | ResFormer value embeddings on alternating layers: lookup `value_embeds[str(i)](idx)`, apply `ve_gate` (tiny linear + sigmoid × 3), add to V path. `has_ve(i, n_layer)` check. Test against a real block with value embeddings active. | |
| 8 | `web/phase-8-full-forward` | All 12 blocks in a loop + `backout_lambda` subtract at the halfway mark + final norm + `lm_head` matmul + `softcap: 15 * tanh(logits / 15)`. Test: `model.forward(torch.tensor([[100]]))` vs WebGPU end-to-end on token 100. | |
| 9 | `web/phase-9-kv-cache` | Pre-allocate K/V buffers of shape `(n_layer, max_seq_len, n_kv_head * head_dim, 2)` per layer. Each step appends new K/V. Attention now reads the full cache. Test: multi-token forward vs PyTorch `.generate()`. | |
| 10 | `web/phase-10-smear` | `prev_embedding` buffer. From step-2 onward, apply `smear_lambda * sigmoid(smear_gate(x[..., :24])) * prev_embedding` to current embedding. Test: two-token generation matches PyTorch. | |
| 11 | `web/phase-11-sampling-loop` | `argmax`, `top_k`, and temperature sampling shaders. JS autoregressive loop: token → forward → sample → append → repeat, until `<|assistant_end|>` or `<|bos|>`. Test: greedy sampling of 16 tokens matches PyTorch greedy. | |
| 12 | `web/phase-12-tokenizer-ui` | Load `tokenizer.json` in browser (huggingface/tokenizers WASM OR a pure-JS BPE decoder). Add minimal chat UI: input box, conversation state with `<\|user_start\|>` / `<\|assistant_start\|>` special tokens, streaming token-by-token display. Test: end-to-end "Moi! Kuka olet?" produces Finnish output. | |

After Phase 12 the runtime is feature-complete for single-turn chat.
Multi-turn conversations fall out trivially (just keep appending to the KV cache).

## Branch / PR conventions

- Branch off latest `main` (not the previous phase branch) — avoids compounded conflicts
- One commit per branch is fine; squash-merge into main
- Test must pass before opening PR (`python test_runtime.py` green)
- Each PR description should paste the test output and any interesting numerics

## Tolerance guide

- RMSNorm, matmul, element-wise ops: `atol=1e-5` expected, often bit-exact in fp32
- Attention (softmax + multi-matmul chain): `atol=1e-4` (fp32 accumulation drifts)
- Full forward pass: `atol=1e-3` (compounded)
- Sampling: compare distributions, not exact picks (unless temperature=0)

## What NOT to do

- Don't rewrite phase 1 or 2 shaders as "warm-up optimization" — they're
  fine. Optimization phase lives after phase 12, if ever.
- Don't add fp16 or INT8 yet. Whole model is fp32 end-to-end in this plan.
- Don't generalize shaders speculatively. If a shader is called with a
  single fixed shape today, hardcode that shape; generalize when the
  *second* call-site appears.
- Don't touch `suomichat/gpt.py` to make the WebGPU port easier. The
  browser runtime must mirror the PyTorch model exactly, not vice versa.

## Roughly how long this takes

Each phase is ~1-4 hours of focused work (shader + runtime.js plumbing +
test + debugging numerics drift). All 10 remaining phases: ~25-40 hours.
Parallel with learning; a phase a day is a reasonable cadence.
