# Dynamic-offset uniform ring plan (Option C)

Branch: `web/perf-uniform-ring` (off `web/perf-cmdbuf-replay`).

## Goal

Decode <100 ms/token (~1.4× faster than 135 ms baseline). Realistic
ceiling without command-buffer reuse.

## Why this should win

Per-decode-forward we currently make ~200 separate `device.queue.writeBuffer`
calls — one per dispatch's small uniform. Each costs ~0.2 ms = ~40 ms
total. Dynamic-offset uniforms collapse those into ONE writeBuffer
that fills a ring of pre-aligned slots.

Bind group caching (already in place) reuses bind groups across
forwards. With the uniform binding pinned to a fixed ring buffer,
the bind groups stay valid forever — no rebuild per forward.

## Architecture

- One `UniformRing`: a single uniform buffer of MAX_SLOTS × SLOT_SIZE
  bytes (e.g. 256 × 256 = 64 KB), allocated once per model.
- `_writeUniforms` becomes "acquire slot + stage values into a JS-side
  ArrayBuffer". Returns `{buffer: ring.buffer, offset}`.
- At session.submit(): one `device.queue.writeBuffer` to push the
  staged ArrayBuffer to GPU, then submit.
- Shaders move uniforms to `@group(1) @binding(0)`. Pipeline layout
  for each shader is now explicit (storage entries in @group(0),
  uniform with hasDynamicOffset in @group(1)).
- `_dispatch` calls `pass.setBindGroup(1, sharedRingBg, [offset])`.

## Phases

### C1 — UniformRing infra + shared @group(1) bind-group layout
- `web/runtime.js`: add `UniformRing` class. Shared
  `BindGroupLayout` for the uniform-with-dynamic-offset binding (used
  by every shader's pipeline layout).

### C2 — Convert ONE shader as proof (scalar_mul)
- Move uniform to `@group(1) @binding(0)`.
- Build explicit `PipelineLayout` for scalar_mul (auto-derived helper).
- Switch the scalar_mul JS call sites to use the ring slot.
- Verify `test_elementwise scalar_mul` still passes.

### C3 — Convert remaining primitives
- All elementwise (add, mul, relu2, sigmoid, softcap), rmsnorm,
  matmul, linear, rope, embedding, embedding_batch, ve_apply,
  ve_apply_t, smear_apply, smear_apply_t, softmax, sdpa_scores,
  sdpa_output, kv_append, embedding_state, smear_apply_state.
- ~20 shaders. Each: WGSL @group(1) move + JS pipeline layout +
  call-site switch to ring slot.

### C4 — One writeBuffer per forward
- Session collects all _writeUniforms into the staged ArrayBuffer.
- session.submit() does ONE `writeBuffer` then submits the encoder.
- Verify all 41 tests still green.

### C5 — Bench decode
- Re-run forwardT bench. Expected drop from ~135 ms to ~95-100 ms
  (~30-40 ms saved on writeBuffer overhead).
- Acceptance: ≥ 25% decode speedup, all tests still green.

## Risks

- **minUniformBufferOffsetAlignment**: typically 256, must be a
  multiple. Slot size = 256 bytes. Largest uniform we have is
  sdpa_scores at 32 bytes — 224 bytes wasted per slot. 256 slots ×
  256 bytes = 64 KB, easy to fit in browser limits.
- **Auto-layout escape hatch**: if explicit layouts turn out to be
  too painful for a particular shader, revert that shader to "auto"
  + per-call uniform buffer (keeps the rest of the win).
- **Test rope_multi**: the rope_state path will need similar
  treatment if/when we use it (currently rope still writes its T0
  via per-call uniform — fine, just one of the 18 dynamic uniforms
  per forward).

## Status update (post-C5): no speedup

C1-C4 all landed correctly. 41/41 tests still green. The ring buffer
collapses ~180 per-dispatch writeBuffer calls into one writeBuffer per
session at submit time.

Bench (RTX 6000 Ada, d6 SFT, N=100, three runs):
  forwardT_ms_per_token: 122, 135, 145 ms   (~10% variance)
  baseline_ms_per_token: 140 ms

Net effect on decode: indistinguishable from baseline. The estimated
~36 ms savings from collapsed writeBuffers did not materialise.

Hypothesis: in Chromium 2026 / Vulkan, `device.queue.writeBuffer` for
small (16-byte) uniforms is much cheaper than I assumed (probably
batched/coalesced internally by the driver). The ~0.2 ms per
writeBuffer figure I reasoned from is wrong.

Other observations:
- The infrastructure (ring, shared bind-group layout, explicit
  pipeline layouts, getRingPipeline, _stageUniforms, _dispatchRing)
  is sound and reusable. Future opt that wants dynamic-offset
  uniforms can build on it.
- 41/41 correctness tests still pass, including the perf-sensitive
  e2e tests. The migration is correct, just doesn't move the needle
  at this model scale on this driver.

What's left as the actual decode bottleneck:
- Per-dispatch GPU compute time (lm_head matmul ~5-10 ms is the worst)
- Per-dispatch pipeline+bind-group setup at the driver level
  (~0.5-0.6 ms × 180 dispatches)
- The fundamental WebGPU per-dispatch overhead that PyTorch CUDA
  doesn't pay (~0.4 ms vs ~5 µs)

Closing the remaining gap likely requires either:
- A different runtime entirely (e.g., emscripten + manual Vulkan)
- Multi-token batching for decode (speculative decoding) — speculative
  draft model proposes K tokens, verify with one forwardBatchT call
- Op fusion via tiled matmul (NOT the single-workgroup fusion that
  failed at d6 — would need multi-workgroup tiling that preserves
  parallelism)

C5 marked BLOCKED on the perf gate but COMPLETE on correctness.

## Acceptance per phase

C1: 41/41 green (no behavior change).
C2: 41/41 green (one shader migrated).
C3: 41/41 green after each batch of shader migrations.
C4: 41/41 green; session writes its uniforms once.
C5: bench decode <100 ms; all tests green.
