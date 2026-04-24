# Command-buffer replay plan (decode acceleration)

Branch: `web/perf-cmdbuf-replay` (off `web/perf-batched-tokens`).

## Goal

Drop decode from ~135 ms/token to <45 ms/token (≥3× speedup).

## Why this should win

Each WebGPU `device.queue.submit` triggers driver validation, command
serialisation, and queue insertion: ~0.65 ms × 180 dispatches per
forward = ~117 ms of pure CPU overhead. The actual GPU compute is
~18 ms. PyTorch CUDA hits ~5 ms because CUDA Graphs let it record
once and replay with near-zero per-launch cost.

WebGPU has no built-in graph API but `device.queue.submit([cmdBuf])`
can be called repeatedly with the same `cmdBuf`. Build it once, replay
per token. Per-token CPU cost collapses to a few `writeBuffer`s + one
`submit`.

## Hard constraints (why this is a refactor)

A `GPUCommandBuffer` bakes in:
- Bind groups (so the buffers they reference are fixed)
- Dispatch dimensions (so workgroup counts are fixed)
- `copyBufferToBuffer` source/dest offsets (so cache offsets are fixed)
- Pipeline + bind-group references

Anything that varies per token must therefore route through:
- Uniform buffers (writable between submits with `writeBuffer`)
- Storage buffers we ping-pong on JS side via parity (which command
  buffer we submit, not which bind group is in it)

Variable-per-token state in the current decode:
1. Input token id → embedding lookup
2. `cache.seqlens` → SDPA dispatch size + cache-append offset + RoPE T0
3. `cache.prevEmbeddingT` swap (smear reads prev, then prev becomes cur)

## Phases

### R1 — static dispatch dimensions for SDPA
- `sdpa_scores.wgsl` / `sdpa_output.wgsl`: dispatch always covers
  MAX_SEQ_LEN keys; shader early-returns when `tk >= valid_Tk`.
- New params: `Tk_dispatched` (used for indexing, large) and
  `Tk_valid` (used for masking, comes from the per-token state).
- Test: existing SDPA tests still pass with the unified dispatch shape.

### R2 — variable-state uniform buffer
- One small uniform buffer (`decode_state`) of ~16 bytes:
  - `token_id: u32` — current decode token
  - `T0: u32`       — current `cache.seqlens` (= position of this token)
  - reserved × 2
- Refactor `embedding_batch.wgsl` (or a dedicated `embedding_decode.wgsl`)
  to read `token_id` from the state buffer instead of via the per-call
  uniform.
- Refactor `sdpa_scores.wgsl` and `rope.wgsl` to read `T0` from the
  state buffer.

### R3 — position-uniform cache append
- New shader `kv_append.wgsl`: `cache[T0 * kvDim + i] = kS[i]`
  for i in `0..kvDim`. Reads T0 from `decode_state`.
- Replace the two `copyBufferToBuffer` calls in `attentionBlockT`'s
  cache path with two `kv_append` dispatches.

### R4 — ping-pong prev_embedding
- Cache holds TWO `prevEmbeddingT` buffers (`prev[0]`, `prev[1]`).
- Even-numbered decode steps: smear reads `prev[0]`, writes `prev[1]`.
- Odd-numbered steps: reads `prev[1]`, writes `prev[0]`.
- Pre-record TWO command buffers (`cmd_even`, `cmd_odd`); submit
  alternately. (Within a single command buffer, bind group is fixed.)

### R5 — record + replay
- After warmup, build `cmd_even` and `cmd_odd` once via `beginSession`
  → encode the entire decode forward → `encoder.finish()` (do NOT submit).
- Replay loop:
  ```
  for step in range(max_new):
      writeBuffer(state, [token_id, T0])
      submit([cmd_even if step%2==0 else cmd_odd])
      argmax → downloadU32 → next token
  ```
- The session abstraction needs a `record()` mode that does NOT submit
  and instead returns the finished command buffer.

### R6 — bench + acceptance
- New bench op `bench_decode_replay` measures: decode 50 tokens via
  the replay path vs decode 50 tokens via standard forwardT.
- Acceptance: ≥3× speedup, decode under 45 ms/token, all 34+ correctness
  tests still green (especially `test_chat_e2e`, `test_kv_cache_two_tokens`,
  `test_greedy_generate`).

## Risks

- **WebGPU command buffer reuse**: spec says it's allowed. Chromium's
  implementation has been stable since 2024. Verify with a tiny
  smoke test in R5 before committing the full refactor.
- **Static SDPA dispatch wastes GPU work**: most threads early-return.
  Negligible at d6 (T_max ~32) — check at higher T.
- **State-uniform writeBuffer cost**: each writeBuffer is ~50 µs even
  for 16 bytes. We do 1-2 per token, so ~0.1 ms total. Fine.
- **Smear ping-pong correctness**: even/odd parity must match the
  reads/writes. Easy to off-by-one. Test by running 16 tokens and
  comparing to the non-replay path.

## Acceptance per phase

R1–R4: 30/30+ correctness tests still green (including chat_e2e and
kv_cache_two_tokens — these stress the variable-state paths).
R5: replay smoke test produces same logits as standard forwardT.
R6: bench gate ≥3× decode speedup.
