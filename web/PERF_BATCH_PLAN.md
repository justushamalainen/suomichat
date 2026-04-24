# Multi-token batching plan

Branch: `web/perf-batched-tokens` (off `web/perf-batching`).

## Why this should win

At d6 (n_embd=384), forwardT spends ~135 ms/token, dominated by per-dispatch
overhead (~200 dispatches × ~0.65 ms). PyTorch CPU does the same forward in
~5 ms because each op call is ~50 µs vs WebGPU's ~400 µs.

We can't shrink WebGPU's per-dispatch cost — but we CAN process multiple
tokens per dispatch. Process T=8 tokens per `forwardT(tokens)` call:

- Each linear projection becomes (T × n_embd) @ (n_embd × N) — same number
  of dispatches, ~T× more work each. Dispatch overhead amortises.
- RoPE applies to T positions in one dispatch.
- SDPA computes (T queries) × (Tk keys) attention with causal mask in one
  shader, instead of T separate decode steps.
- KV cache appends T entries per layer in one go.
- LM head projection processes T rows.

Estimated: prefill 8 tokens in ~150 ms vs 8 × 135 = 1080 ms. ~7× speedup
for prefill. Decode (one token at a time) stays at ~135 ms — but most chat
prompts are 50+ tokens of prefill, so total throughput improves dramatically.

## Phases

### MT1 — Per-position RoPE
- Update `ropeApplyT` to accept either a single T0 or an array of positions
  (one per row). Pass positions as a small uniform or a tiny storage buffer.
- New shader `rope_multi.wgsl` (or extend `rope.wgsl`) reads cos/sin at
  the row-specific position.
- Test: shape (T=4, head_dim) with positions [0..3], compare to PyTorch
  per-position RoPE.

### MT2 — T>1 SDPA shaders with causal mask
- `sdpa_scores_t.wgsl`: scores[t, h, tk] = q[t, h] · K_cache[tk, h_kv] / sqrt(hd)
  — but only for `tk <= T0 + t` (causal). Out-of-range scores get -inf.
- `sdpa_output_t.wgsl`: out[t, h, d] = sum_tk attn[t, h, tk] * V_cache[tk, h_kv, d]
- Multi-row softmax already works via existing `softmax.wgsl`.
- Multi-position cache append: `_copyBuffer(kS, 0, cache.k, T0*kvDim*4, T*kvDim*4)`.

### MT3 — Multi-token forwardT
- `forwardBatchT(device, model, tokenIds, cache)`:
  - Embed each token via existing `embedding.wgsl` (one row per token);
    one dispatch via small modification (loop over T).
  - rmsnorm (T rows already supported)
  - smear: PyTorch applies smear to positions 1+, gating each position by
    its own gate(x[t, :24]). Either implement multi-position smear or
    note that prefill from a fresh cache has no prev_embedding so smear
    only fires for the second-and-later positions WITHIN the batch.
    (See gpt.py:438 for prefill semantics.)
- 6 transformer blocks: each takes (T, n_embd), outputs (T, n_embd).
  Most ops already work; attention is the only one that needs the
  T>1 SDPA path.
- Backout, final norm, lm_head, softcap → returns (T, vocab).
- Test: forwardBatchT([t0, t1, t2, t3]) vs 4 separate forwardT calls
  (with KV cache reset between). PyTorch model([[t0..t3]]) is the
  ground truth. Argmax of last position must match.

### MT4 — greedyGenerate with prefill batch
- Prefill: one `forwardBatchT(promptTokens)` call → logits for last
  position → first generated token.
- Decode: existing per-token `forwardT` loop.
- Test: same `chat_e2e` test, should still match PyTorch exactly.

### MT5 — Bench prefill speedup
- Add `bench_prefill_batch` op: time `forwardBatchT(N tokens)` vs
  N separate `forwardT` calls.
- Acceptance: ≥ 3× speedup for N=8.

## Acceptance per phase

- 30/30 (or higher) tests pass.
- New tests for batched paths match single-token path numerically
  (atol 5e-4 for intermediate, exact argmax for end-to-end).
- Phase MT5 bench gate: ≥ 3× prefill speedup.

## Risks

- Causal masking in WGSL — getting -inf right matters; need exp(-inf)=0
  to behave correctly through softmax.
- KV cache append for T>1: must append at correct offset.
- Smear gate position semantics in prefill mode — PyTorch's prefill
  smear is subtle (positions 1+ only).
- The fp32 accumulation order may drift between batched and single-step;
  test tolerances must accommodate.
