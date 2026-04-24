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

## Acceptance per phase

C1: 41/41 green (no behavior change).
C2: 41/41 green (one shader migrated).
C3: 41/41 green after each batch of shader migrations.
C4: 41/41 green; session writes its uniforms once.
C5: bench decode <100 ms; all tests green.
