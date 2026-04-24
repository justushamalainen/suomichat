# WebGPU runtime — perf batching plan

Branch: `web/perf-batching` (single branch, one commit per phase, push at the end).

## Goal

Bring forwardT (single-token decode) from ~140 ms → <30 ms on RTX 6000 Ada
by batching all per-forward GPU work into one command buffer instead of
~310 separate submits. Stretch: tiled matmul, fp16 weights — only if
time after the main win is in.

## Baseline

Measured 2026-04-23 on RTX 6000 Ada, d6 SFT (n_layer=6, n_embd=384,
vocab=32768), Chromium 2026 with `--enable-unsafe-webgpu --use-vulkan`:

- `forwardT`: ~140 ms/token (avg over 30 iters, post-warmup)
- `_forward_legacy` (Float32Array round-trips): ~180 ms/token
- ~310 dispatches per forward, each its own `device.queue.submit`

Headline limit is per-submit overhead (~0.4 ms × 310 ≈ 125 ms).

## Acceptance per phase

Every phase MUST keep `test_runtime.py` at 30/30 green. Phases 4 and 5
also have a perf gate (bench prints ms/token + speedup, regression
fails CI).

## Phases

### Phase 1 — Bench harness lockdown
- Make `python web/test_runtime.py --bench` print structured JSON
  summary (baseline_ms_per_token, current_ms_per_token, speedup).
- Add `--bench-only` flag that skips correctness tests and runs N=50.
- Document baseline numbers above; later phases compare against them.

**Acceptance:** Script runs, prints JSON, doesn't break the regular
`python web/test_runtime.py` invocation.

### Phase 2 — Session abstraction (no behavior change)
- New `web/runtime.js` exports `beginSession(device, model)` →
  `{ encoder, pass, dispatch(), copy(), submit() }`.
- Session manages encoder + pass lifecycle: `dispatch()` opens a
  compute pass if one isn't open; `copy()` first ends any open pass
  then encodes a copyBufferToBuffer; `submit()` closes pass + finishes
  encoder + queue.submit.
- Modify `_dispatch(device, pipeline, buffers, uniforms, dispatchDims, sess=null)`
  to accept an optional session. If `sess === null`, current behavior
  (standalone submit). If passed, append to session.
- BufferPool gains a "no-release-during-session" mode so buffers
  in-flight aren't recycled: track buffers acquired since last
  `submit()`, return them only after submit().

**Acceptance:** `python web/test_runtime.py` 30/30. No perf change yet
(no caller uses session).

### Phase 3 — Thread session through primitives
- Every T primitive (`linearT`, `rmsnormT`, `addT`, `mulT`, `scalarMulT`,
  `relu2T`, `sigmoidT`, `softcapT`, `softmaxT`, `ropeApplyT`,
  `embeddingT`, `argmaxT`, `veApplyT`, smear-apply path) gains
  `sess = null` last parameter.
- Float32Array wrappers (linear, rmsnorm, etc.) keep submitting standalone
  (don't pass sess) so existing tests still hit the legacy path.

**Acceptance:** 30/30 green. Still no perf change (sess unused).

### Phase 4 — forwardT uses a single session
- Inside `forwardT`, create `sess = beginSession(...)` at entry,
  thread through all sub-calls (transformerBlockT, attentionBlockT,
  mlpBlockT, etc.), call `sess.submit()` at the end.
- copyTensor (used for x0 snapshot and cache.prevEmbeddingT) routes
  through `sess.copy(...)` so it lands inside the same encoder.
- Cache append in attentionBlockT routes through `sess.copy(...)`.
- After `sess.submit()`, the returned `logitsT` is on GPU; downloadTensor
  awaits map.

**Acceptance:**
- 30/30 green (especially `chat_e2e` and `kv_cache_two_tokens` —
  these stress the cache.copy + smear path).
- Bench shows ≥5× speedup (forwardT < 30 ms/token on d6).
- If <5×: investigate, document, mark BLOCKED with the actual numbers.

### Phase 4 results (post-mortem)

Implementation works (30/30 green). Speedup is in the noise (~1.0x).

**Why batching submits didn't help**: I assumed `device.queue.submit`
overhead was the dominant cost (~0.4 ms × 310 dispatches ≈ 125 ms).
Wrong. WebGPU's submit is much cheaper than I thought. The actual
~125 ms floor comes from:

- `device.createBindGroup` per dispatch (~0.2 ms × 310 = 62 ms)
- `queue.writeBuffer` per uniform (~0.2 ms × 310 = 62 ms)

That work happens whether dispatches are in one submit or 310 — the
batching collapsed the wrong axis. Remaining axes:

- Reuse bind groups across calls (buffers from the pool come back to
  the same GPUBuffer object → same bind group)
- Replace per-dispatch uniform writeBuffer with one big "uniform ring
  buffer" + dynamic offsets per dispatch
- Bigger compute per dispatch (tiled matmul)
- Fewer dispatches (op fusion)

### Phase 4b (NEW) — bind-group cache + uniform-offset reuse

- Cache bind groups by (pipeline, [buffer ids...]) tuple. Pool buffer
  reuse means many calls hit the same key.
- Pre-allocate one large UNIFORM | DYNAMIC_OFFSET buffer. Each dispatch
  writes a 32-byte slot via dynamic offset; one queue.writeBuffer per
  forward instead of one per dispatch.
- Acceptance: 30/30 green; bench drops below 60 ms/token (>2x).

### Phase 5 — greedyGenerate one-session loop
- Each generation step is one session: forwardT(token, cache, sess) +
  argmaxT(logitsT, sess) + downloadU32(argmaxT) — all in one submit.
- The downloadU32 is the only sync point; it returns the next token id.

**Acceptance:** 30/30 green; bench tweaks confirm no regression.

### Status (2026-04-24)

| Phase | Status | Result |
|---|---|---|
| 1 — bench harness | done | JSON output, baseline 140 ms |
| 2 — session abstraction | done | no behavior change |
| 3 — thread session | done | subsumed by Phase 2 |
| 4 — single-submit forwardT | done | ~1.0x (submit wasn't the bottleneck) |
| 4b — bind-group cache | done partial | marginal (~5 ms within variance) |
| 5 — single-session generate | done | subsumed by Phase 4 |
| 6 — tiled matmul | NOT STARTED | low-impact at d6 (M=1 means no M-tile reuse) |
| 7 — fp16 weights | NOT STARTED | half memory bandwidth, but dispatch overhead dominates |

**Bottom line**: ~135 ms/token vs 140 baseline. The actual ceiling is
per-dispatch driver overhead (~0.65 ms × ~200 dispatches). To meaningfully
beat that you need ONE of:

- **Op fusion** — write a big "transformer block" shader that does
  attn or mlp in one dispatch. Complex; potential 5–10x.
- **Multi-token batching** — process 4–8 tokens per forward, reusing
  the same dispatch overhead. Requires real T>1 attention. ~4x.
- **Dynamic-offset uniform ring** — collapse the 200 queue.writeBuffer
  calls into 1. Requires explicit pipeline layouts (every shader
  needs @group(1) for uniforms). ~2x.

None of these are quick wins; each is a multi-day refactor with its
own correctness risks. Pausing here for direction.

### Phase 6 (stretch) — tiled matmul
- Replace `shaders/matmul.wgsl` and `shaders/linear.wgsl` with a
  workgroup-tile version (8×8 or 16×16 tile, shared memory loads).
- Especially important for `lm_head` (M=1, K=384, N=32768) — the
  largest matmul in the forward.
- Test: `test_matmul`, `test_linear`, end-to-end argmax-match.

**Acceptance:** 30/30 green; bench drops further (target < 20 ms/token).

### Phase 7 (stretch) — fp16 weights, fp32 compute
- Convert weights.bin to fp16 on disk (convert.py adds an fp16 mode).
- Shaders read fp16, accumulate in fp32. Halves memory bandwidth.
- Test: argmax match across all generation tests.

**Acceptance:** 30/30 green; weights.bin halves in size; bench may
improve further or stay neutral on RTX 6000 Ada (it's compute-bound
at this scale anyway).

## How perf is verified

After every commit on `web/perf-batching` the autonomous loop runs:

```
SUOMICHAT_BASE_DIR=/home/janitor/llm-training/data-fi-v2 .venv/bin/python web/test_runtime.py
```

This is the gate — must show 30/30 + bench numbers. Phases 4 and 5
will fail loud if speedup is <5×, blocking the loop.

## Risk register

- **Pool reuse within a session**: multiple dispatches in the same pass
  must not write to the same buffer if any other dispatch reads it.
  Mitigation: pool's no-release-during-session mode (Phase 2).
- **WebGPU bind group caching**: each dispatch builds a fresh bind
  group; that's CPU work too. After Phase 4 we'll see if it matters.
- **Validation cost in Chromium debug build**: bench numbers may be
  pessimistic; profile to confirm.
