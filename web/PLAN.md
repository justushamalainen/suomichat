# WebGPU runtime — implementation plan

Everything lands on **one branch** (`web/phase-2-rmsnorm`, extended). One commit
per phase inside that branch. Squash-merge to `main` only at the end, when the
whole runtime works.

## Scope

Run the existing **suomichat d12 SFT** model in the browser via WebGPU,
with autoregressive greedy generation of a user-supplied Finnish prompt.
fp32 weights end-to-end. Correctness over speed. No optimization, no
quantization, no mobile, no multi-model generalisation.

## Testing approach

All verification goes through `web/test_runtime.py`:

```python
async def test_<op>(...):
    ref = ref_<op>(...)                           # PyTorch on CPU
    got = await run_browser_op("<op>", args)      # headless Chromium / WebGPU
    compare(ref, got, atol=...)
```

- Reference is plain PyTorch, always fp32, on CPU, using the actual loaded
  suomichat model weights where relevant (so we test *our* model, not a
  generic transformer).
- WebGPU runs via Playwright + `test.html` in headless Chromium with
  `--enable-unsafe-webgpu --use-vulkan` on the local RTX 6000 Ada.
- Comparison is `max|ref - got|` under a dtype-appropriate `atol`.

Add one `async def test_<op>` per op. `test_runtime.py main()` dispatches all of
them in order. A `--filter` CLI arg lets you run one test at a time while
debugging a specific op.

**Tolerance guide (all fp32 → fp32):**

| Op family | atol | Why |
|---|---|---|
| Embedding lookup, element-wise ops, matmul on small shapes | 0.0 | bit-exact; same order of ops in both runtimes |
| RMSNorm, RoPE, single-matmul composites | 1e-5 | one reduction or small pipeline, no accumulation drift |
| Single attention block | 1e-4 | softmax + multi-matmul chain; fp32 accumulation order differs between torch and our shader |
| MLP block | 1e-5 | two matmuls + pointwise, minimal drift |
| Full transformer block | 1e-4 | attention + MLP compounded |
| Full 12-layer forward pass | 1e-3 | 12× composition; still well under any token-picking boundary |
| Greedy sampling | **exact match** | argmax picks the same token even with 1e-3 drift 99.99%+ of the time; on clear-winner logits it's always identical |

## Order of phases (each = one commit on this branch)

Each phase lists: **what** (the op), **shader file**, **runtime.js export**,
**test function**, **PyTorch reference** (the exact thing we compare
against), and **acceptance**.

### ✅ Phase 1 — Embedding lookup (done)

- Shader: `shaders/embedding.wgsl`
- Runtime: `embeddingLookup(device, model, tokenId)`
- Test: `test_embedding(model, token_id)`
- Ref: `model.transformer.wte.weight[token_id]`
- Acceptance: `atol=1e-5`. Observed: 0.0 (bit-exact).

### ✅ Phase 2 — RMSNorm (done)

- Shader: `shaders/rmsnorm.wgsl` (workgroup-shared-memory tree reduction)
- Runtime: `rmsnorm(device, model, input, eps=1e-5)`
- Test: `test_rmsnorm(seed=0, n=768)` and `seed=1`
- Ref: `F.rms_norm(x, (n,), eps=1e-5)`
- Acceptance: `atol=1e-5`. Observed: 0.0.

### Phase 3 — Naive matmul

The one op everything else uses.

- **Shader**: `shaders/matmul.wgsl` — computes `C (M×N) = A (M×K) @ B (K×N)`.
  Dispatch: `(M, N)` workgroups of 1 thread each. Each thread loops over K.
  (Dumb, slow, correct.)
- **Runtime**: `matmul(device, aBuf, aShape, bBuf, bShape)` — takes two
  GPUBuffers (not CPU arrays; keeps data on-GPU for chained ops). Returns
  output GPUBuffer. A CPU-input convenience overload for tests only.
- **Test**: `test_matmul(seed, M, K, N)` — random A, B, compare to `A @ B`.
  Run at M=1 K=768 N=768 (single-token projection shape) and M=768 K=768 N=2048 (MLP shape).
- **Ref**: `ref = a @ b`
- **Acceptance**: `atol=5e-4` (fp32 accumulation order differs from torch; relative error stays ~1e-6).

### Phase 4 — Elementwise ops

Three tiny shaders we need repeatedly.

- **Shaders**: `add.wgsl`, `mul.wgsl`, `scalar_mul.wgsl`, `relu2.wgsl`, `sigmoid.wgsl`
- **Runtime**: `add`, `mul`, `scalarMul`, `relu2`, `sigmoid` — all on
  GPUBuffers, return GPUBuffers.
- **Tests**: one per op, random inputs of shape (768,) and (1, 768)
- **Refs**: `a + b`, `a * b`, `alpha * x`, `F.relu(x).square()`, `torch.sigmoid(x)`
- **Acceptance**: bit-exact.

### Phase 5 — RoPE apply

- **Shader**: `shaders/rope.wgsl` — rotates the last dim in pairs:
  ```
  d = head_dim // 2
  y[..., :d]  = x[..., :d] * cos + x[..., d:] * sin
  y[..., d:]  = x[..., :d] * (-sin) + x[..., d:] * cos
  ```
- **Runtime**: `ropeApply(device, xBuf, shape, cosBuf, sinBuf, T0)` where
  `T0` is the cache offset (0 for first token).
- **Test**: `test_rope` with random Q shape `(1, 1, 6, 128)` and the
  pre-computed `cos`/`sin` from the loaded model's `model.cos` buffer.
- **Ref**: `suomichat.gpt.apply_rotary_emb(q, cos[:, T0:T0+1], sin[:, T0:T0+1])`
- **Acceptance**: `atol=1e-5`.

### Phase 6 — Single attention block (T=1, no KV cache)

First real layer composite. Tests our ability to chain shaders on-GPU.

- **Runtime**: `attentionBlock(device, model, layerIdx, xBuf, T0)` — reads
  all weights for `model.transformer.h[layerIdx].attn` from the loaded
  model, runs full forward, returns output GPUBuffer.
  - c_q / c_k / c_v projections (matmul)
  - reshape into heads
  - RoPE on Q, K
  - RMSNorm on Q, K (QK norm)
  - scale: `q *= 1.2; k *= 1.2`
  - softmax-based scaled dot product (T=1 = trivial; just computes
    `softmax(qk^T/sqrt(hd)) v` over a single key/value — identity attention).
    We'll use a real softmax + matmul via shaders; skipping FA3.
  - c_proj matmul
- **Shader**: `shaders/softmax.wgsl` (new, one workgroup reduction like RMSNorm)
- **Test**: `test_attention_block_layer0` — random `x` shape `(1, 1, 768)`,
  run our `attentionBlock` and PyTorch's `model.transformer.h[0].attn(x, None, cos_sin, (-1,0), None)`,
  compare outputs. T0=0.
- **Ref**: `model.transformer.h[0].attn(x_pytorch, ve=None, cos_sin=(cos[:, :1], sin[:, :1]), window_size=(-1, 0), kv_cache=None)`
- **Acceptance**: `atol=1e-4`.

*Note:* suomichat's attn block also handles `ve` (value embeddings) on
alternating layers. Layer 0 does NOT have value embeddings
(`has_ve(0, 12)` is False because 12 is even; see `gpt.py`), so we can
skip them for the first attention test.

### Phase 7 — Value embeddings

- **Runtime**: extend `attentionBlock` to handle the `ve` path: if
  `str(layerIdx)` is in `model.value_embeds`, lookup `value_embeds[str(layerIdx)](token_id)`,
  compute gate via `ve_gate` (tiny linear + sigmoid × 3 scale factor),
  `v = v + gate * ve`. Then a fresh test on a layer that has it.
- **Test**: `test_attention_block_layer1` (first layer where `has_ve(1,12)` is True).
- **Ref**: `model.transformer.h[1].attn(x, ve, cos_sin, (-1,0), None)` with
  `ve = model.value_embeds['1'](token_tensor)`
- **Acceptance**: `atol=1e-4`.

### Phase 8 — MLP block

- **Runtime**: `mlpBlock(device, model, layerIdx, xBuf)` — c_fc matmul →
  relu² → c_proj matmul.
- **Test**: `test_mlp_layer0` — random `x`, compare our output to
  `model.transformer.h[0].mlp(x)`.
- **Ref**: `model.transformer.h[0].mlp(x)`
- **Acceptance**: `atol=1e-5`.

### Phase 9 — Full transformer block

- **Runtime**: `transformerBlock(device, model, layerIdx, xBuf, x0Buf, T0)` —
  does the full block:
  ```
  x = resid_lambdas[i] * x + x0_lambdas[i] * x0
  x = x + attentionBlock(norm(x), ve, cos_sin)
  x = x + mlpBlock(norm(x))
  ```
- **Test**: `test_block_layer0` — same pattern as above, random x, x0.
- **Ref**: `model.transformer.h[0](x, ve=None, cos_sin=..., window_size=..., kv_cache=None)`,
  where the caller of the real block handles the `resid_lambdas * x + x0_lambdas * x0`
  wrap — we replicate that wrap in our test ref.
- **Acceptance**: `atol=1e-4`.

### Phase 10 — Full forward pass (first token)

- **Runtime**: `forward(device, model, tokenId)` — loops all 12 blocks:
  1. embedding(tokenId) → x, also save `x0 = norm(x)`
  2. (no smear on first token)
  3. for i in range(n_layer):
     - load `ve = value_embeds[str(i)](token)` if `has_ve(i, n_layer)`
     - `x = transformerBlock(i, x, x0, T0=0)`
     - if `i == n_layer // 2`: save `x_backout = x`
  4. `x = x - backout_lambda * x_backout`
  5. `x = norm(x)`
  6. `logits = lm_head(x)`
  7. `logits = 15.0 * tanh(logits / 15.0)` (softcap)
  8. slice to `:vocab_size` (unpad)
- **Shader**: `shaders/tanh_softcap.wgsl` (softcap op)
- **Test**: `test_full_forward_first_token` — pick token_id = `<|bos|>` id,
  compute logits via `model(torch.tensor([[bos_id]]))` and via WebGPU.
  Compare.
- **Ref**: `model(torch.tensor([[token_id]]))` → logits
- **Acceptance**: `atol=1e-3`.

### Phase 11 — KV cache

Everything up to phase 10 was T=1 / no cache. This phase adds persistent
K/V buffers so subsequent tokens can attend over history.

- **Runtime**:
  - `initKVCache(model, maxSeqLen)` — preallocates per-layer K/V GPUBuffers
    of shape `(max_seq_len, n_kv_head * head_dim)`, plus a `cache_seqlens`
    uniform.
  - `attentionBlock(...)` updated: append new K/V to cache, attend over
    `cache[:cache_seqlens+1]`.
  - `forward(device, model, tokenId, cache)` — takes the cache object,
    advances it.
- **Test**: `test_kv_cache_two_tokens` — run greedy generation for 2 tokens
  through both PyTorch (`model.generate([bos, x])`) and WebGPU.
  Compare picked token IDs.
- **Ref**: `list(model.generate([bos, token1], max_tokens=1, temperature=0))`
- **Acceptance**: picked token IDs **match exactly** (greedy is deterministic).

### Phase 12 — Smear gate

Smear applies from step 2 onward (when there's a previous embedding).
Requires a new `prev_embedding` cache buffer.

- **Runtime**:
  - Add `prev_embedding` to the cache object.
  - Before the transformer blocks, if `prev_embedding is not None`:
    ```
    gate = smear_lambda * sigmoid(smear_gate(x[:, :, :24]))
    x = x + gate * prev_embedding
    ```
  - After the pass: update `prev_embedding = x[:, -1:, :]` (post-norm).
- **Test**: `test_smear_two_tokens` — same as test_kv_cache_two_tokens but
  verifying that smear applies on the second token.
- **Ref**: PyTorch's `model.forward` with `kv_cache` populated.
- **Acceptance**: picked token IDs match exactly.

### Phase 13 — Sampling + generation loop

- **Runtime**:
  - `argmax(device, logitsBuf, vocabSize)` shader — returns a single u32.
  - `generate(device, model, promptTokens, maxTokens, temperature=0.0)` —
    JavaScript autoregressive loop. Calls `forward(...)` per token,
    extracts last-position logits, picks, appends, repeats. Stops at
    `<|assistant_end|>` or `<|bos|>` or maxTokens.
- **Test**: `test_greedy_generate_16` — prompt the model with a short
  token sequence, generate 16 tokens via WebGPU and via PyTorch greedy.
  Token sequences must match.
- **Ref**: `list(model.generate(prompt_tokens, max_tokens=16, temperature=0))`
- **Acceptance**: exact match on all 16 tokens.

### Phase 14 — Tokenizer + chat UI

- **Runtime**:
  - Load `tokenizer.json` via the `@huggingface/tokenizers` WASM build
    (pinned version).
  - `encode(text)` / `decode(ids)`.
  - Helper that builds a chat conversation:
    ```
    tokens = [bos, user_start, ...encode(userMsg), user_end, assistant_start]
    ```
- **UI** (extends `index.html`):
  - Input textarea + Submit button
  - Streaming token display (per-token `decode` for new token only)
  - Stop button (aborts the generation loop cleanly)
- **Test**: `test_end_to_end` — tokenize "Moi! Kuka olet?", generate 50 tokens
  via WebGPU greedy, compare final decoded string character-for-character to
  PyTorch greedy.
- **Ref**: PyTorch `chat_cli.py` equivalent path with `temperature=0`, `top_k=1`.
- **Acceptance**: identical output strings.

## Single branch workflow

```
git checkout web/phase-2-rmsnorm

# For each phase:
#   write shader + runtime + test
#   python test_runtime.py (narrow to one test with --filter=<name>)
#   iterate until green
#   git add + commit with message "Phase N: <op>"
#   git push

# After phase 14 is green:
gh pr create --base main --head web/phase-2-rmsnorm \
    --title "WebGPU runtime for d12 SFT (Phases 1-14)" \
    --body "see web/PLAN.md"

# User squash-merges to main.
```

## What we're deliberately leaving out

(Already stated; listing for clarity.)

- fp16/INT8/INT4 — after phase 14 if at all
- Tiled matmul / fused kernels — never, in this project
- Sliding window attention — model was trained with SSSL but for browser
  we use full causal attention (will produce numerically different but
  functionally correct output; retrain with `--window-pattern=L` if you
  want exact parity)
- Multi-turn conversation state in the UI — trivial extension after
  phase 14; out of scope for the plan
