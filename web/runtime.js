// SuomiChat WebGPU runtime — Phase 1.
//
// Goal of this file: be the easiest possible thing that loads a
// suomichat checkpoint into GPU memory and runs ONE op (embedding
// lookup). Everything fancy comes later.

// ---------------------------------------------------------------------
// 1. WebGPU device init.
// ---------------------------------------------------------------------
export async function initWebGPU() {
  if (!navigator.gpu) {
    throw new Error("WebGPU not supported — try Chrome 113+ or enable the flag in Firefox.");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter available.");

  // Request maxBufferSize = 2 GB so we can fit d12 fp32 weights.
  // Default in Chrome is 256 MB which is way too small for ML.
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: Math.min(2 ** 31 - 1, adapter.limits.maxStorageBufferBindingSize),
      maxBufferSize: Math.min(2 ** 31 - 1, adapter.limits.maxBufferSize),
    },
  });

  // Fail-fast on uncaptured GPU errors so we see them immediately.
  device.addEventListener?.("uncapturederror", e => console.error("WebGPU error:", e.error));

  return { adapter, device };
}

// ---------------------------------------------------------------------
// 2. Model loader.
// ---------------------------------------------------------------------
//
// `manifest.tensors` gives name -> {offset, shape, dtype}. We fetch
// weights.bin as one ArrayBuffer, then create one GPUBuffer per
// tensor by copying the slice into GPU memory.
//
// Returns a `model` object with:
//   - config:   the GPTConfig (n_layer, n_embd, ...)
//   - tensors:  Map<name, { buffer: GPUBuffer, shape: number[], dtype: string }>
//   - shaders:  cached compute pipelines (filled in lazily)
// ---------------------------------------------------------------------
export async function loadModel(device, config, manifest, weightsUrl) {
  const resp = await fetch(weightsUrl);
  if (!resp.ok) throw new Error(`Failed to fetch ${weightsUrl}: ${resp.status}`);
  const totalBytes = +resp.headers.get("content-length") || manifest.total_size;
  console.log(`Fetching ${(totalBytes / 1024 ** 2).toFixed(1)} MB of weights…`);
  const buf = await resp.arrayBuffer();
  if (buf.byteLength !== manifest.total_size) {
    console.warn(`weights.bin size ${buf.byteLength} != manifest ${manifest.total_size}`);
  }

  // Per-tensor GPU upload. We use a STORAGE buffer because the shaders
  // will read these weights via storage bindings.
  const tensors = new Map();
  for (const [name, meta] of Object.entries(manifest.tensors)) {
    const numel = meta.shape.reduce((a, b) => a * b, 1);
    const bytesPer = manifest.bytes_per_element;
    const sizeBytes = numel * bytesPer;
    const slice = buf.slice(meta.offset, meta.offset + sizeBytes);

    const gpuBuf = device.createBuffer({
      label: name,
      size: sizeBytes,
      // COPY_SRC lets us read small scalar tensors back to CPU (resid_lambdas etc.)
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(gpuBuf, 0, slice);
    tensors.set(name, { buffer: gpuBuf, shape: meta.shape, dtype: meta.dtype });
  }
  await device.queue.onSubmittedWorkDone();

  return { config, tensors, shaders: new Map() };
}

// ---------------------------------------------------------------------
// 3. Shader loader + caching.
// ---------------------------------------------------------------------
async function getPipeline(device, model, name) {
  if (model.shaders.has(name)) return model.shaders.get(name);

  const wgsl = await (await fetch(`./shaders/${name}.wgsl`)).text();
  const module = device.createShaderModule({ label: name, code: wgsl });
  const pipeline = device.createComputePipeline({
    label: name,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  model.shaders.set(name, pipeline);
  return pipeline;
}

// ---------------------------------------------------------------------
// 4. Helper: turn a Float32Array into a STORAGE GPUBuffer.
// ---------------------------------------------------------------------
function uploadF32(device, label, arr) {
  const buf = device.createBuffer({
    label, size: arr.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(buf, 0, arr);
  return buf;
}

// ---------------------------------------------------------------------
// 5. Helper: read a STORAGE buffer back to CPU as Float32Array.
// ---------------------------------------------------------------------
async function downloadF32(device, srcBuffer, byteLength) {
  const staging = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(srcBuffer, 0, staging, 0, byteLength);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}

// ---------------------------------------------------------------------
// 6. Generalised embedding lookup: one row of a (vocab, dim) weight.
//    Used by both wte (token embedding) and value_embeds[i].
// ---------------------------------------------------------------------
export async function embeddingLookupNamed(device, model, weightName, tokenId) {
  const w = model.tensors.get(weightName);
  if (!w) throw new Error(`${weightName} missing from manifest`);
  const [vocab, dim] = w.shape;
  if (tokenId < 0 || tokenId >= vocab) throw new Error(`token id ${tokenId} out of range`);

  const pipeline = await getPipeline(device, model, "embedding");
  const outBytes = dim * 4;
  const out = device.createBuffer({
    label: `${weightName}_out`, size: outBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const uniformData = new Uint32Array([tokenId, dim, vocab, 0]);
  const uniforms = device.createBuffer({
    size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniforms, 0, uniformData);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: w.buffer } },
      { binding: 1, resource: { buffer: out } },
      { binding: 2, resource: { buffer: uniforms } },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(dim / 64));
  pass.end();
  const staging = device.createBuffer({
    size: outBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  enc.copyBufferToBuffer(out, 0, staging, 0, outBytes);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap(); staging.destroy(); out.destroy(); uniforms.destroy();
  return result;
}

// ---------------------------------------------------------------------
// 6b. The first real operation: embedding lookup (wte).
// ---------------------------------------------------------------------
//
// Equivalent in PyTorch:
//     x = model.transformer.wte(token_id)   # shape (n_embd,)
//
// We dispatch one workgroup with `n_embd` threads. Each thread copies
// one element from row `tokenId` of the embedding matrix into the
// output buffer. (Yes, this is the slowest possible way to do an
// embedding lookup. It's also the most readable.)
// ---------------------------------------------------------------------
export async function embeddingLookup(device, model, tokenId) {
  const wte = model.tensors.get("transformer.wte.weight");
  if (!wte) throw new Error("transformer.wte.weight missing from manifest");
  const [vocab, nEmbd] = wte.shape;
  if (tokenId < 0 || tokenId >= vocab) throw new Error(`token id ${tokenId} out of range`);

  const pipeline = await getPipeline(device, model, "embedding");

  // Output buffer: one row of the embedding matrix.
  const outBytes = nEmbd * 4;
  const out = device.createBuffer({
    label: "embedding_out",
    size: outBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // Uniforms: token id + dimensions. Uniforms must be 16-byte aligned.
  const uniformData = new Uint32Array([tokenId, nEmbd, vocab, 0]);
  const uniforms = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniforms, 0, uniformData);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: wte.buffer } },
      { binding: 1, resource: { buffer: out } },
      { binding: 2, resource: { buffer: uniforms } },
    ],
  });

  // Dispatch: one workgroup, ceil(nEmbd / 64) workgroup-x threads handled
  // by the shader's @workgroup_size(64).
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(nEmbd / 64));
  pass.end();

  // Stage out → CPU so we can return it to JS as a Float32Array.
  const staging = device.createBuffer({
    size: outBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  enc.copyBufferToBuffer(out, 0, staging, 0, outBytes);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  return result;
}

// ---------------------------------------------------------------------
// 7. Matmul: C (M×N) = A (M×K) @ B (K×N), row-major f32.
// ---------------------------------------------------------------------
//
// Takes Float32Array inputs for a test-friendly API. Internally uploads
// to GPU, dispatches the matmul shader, reads back the result. Future
// phases will want a pure-GPU variant (aBuf, bBuf → cBuf) so we don't
// round-trip through CPU; we add that when the first non-test caller
// appears.
// ---------------------------------------------------------------------
export async function matmul(device, model, a, aShape, b, bShape) {
  const [M, K] = aShape;
  const [K2, N] = bShape;
  if (K !== K2) throw new Error(`matmul shape mismatch: (${M},${K}) @ (${K2},${N})`);

  const pipeline = await getPipeline(device, model, "matmul");
  const aBuf = uploadF32(device, "matmul_a", a);
  const bBuf = uploadF32(device, "matmul_b", b);
  const cBuf = device.createBuffer({
    label: "matmul_c", size: M * N * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const ub = new ArrayBuffer(16);
  const u32 = new Uint32Array(ub);
  u32[0] = M; u32[1] = K; u32[2] = N;
  const uniforms = device.createBuffer({
    size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniforms, 0, ub);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: aBuf } },
      { binding: 1, resource: { buffer: bBuf } },
      { binding: 2, resource: { buffer: cBuf } },
      { binding: 3, resource: { buffer: uniforms } },
    ],
  });

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(M / 8), Math.ceil(N / 8));
  pass.end();
  device.queue.submit([enc.finish()]);

  const result = await downloadF32(device, cBuf, M * N * 4);
  aBuf.destroy(); bBuf.destroy(); cBuf.destroy(); uniforms.destroy();
  return result;
}

// ---------------------------------------------------------------------
// Element-wise ops (add, mul, scalar_mul, relu2, sigmoid).
// All take Float32Array input(s) + return Float32Array for test-friendliness.
// ---------------------------------------------------------------------
function buildBinaryUniforms(device, n) {
  const ub = new ArrayBuffer(16);
  new Uint32Array(ub, 0, 1)[0] = n;
  const buf = device.createBuffer({
    size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, ub);
  return buf;
}

async function _elemBinary(device, model, name, a, b) {
  if (a.length !== b.length) throw new Error(`${name}: shape mismatch ${a.length} vs ${b.length}`);
  const n = a.length;
  const pipeline = await getPipeline(device, model, name);
  const aBuf = uploadF32(device, `${name}_a`, a);
  const bBuf = uploadF32(device, `${name}_b`, b);
  const cBuf = device.createBuffer({
    label: `${name}_c`, size: n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const uniforms = buildBinaryUniforms(device, n);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: aBuf } },
      { binding: 1, resource: { buffer: bBuf } },
      { binding: 2, resource: { buffer: cBuf } },
      { binding: 3, resource: { buffer: uniforms } },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(n / 64));
  pass.end();
  device.queue.submit([enc.finish()]);
  const result = await downloadF32(device, cBuf, n * 4);
  aBuf.destroy(); bBuf.destroy(); cBuf.destroy(); uniforms.destroy();
  return result;
}

async function _elemUnary(device, model, name, a, extraFloat = null) {
  const n = a.length;
  const pipeline = await getPipeline(device, model, name);
  const aBuf = uploadF32(device, `${name}_a`, a);
  const cBuf = device.createBuffer({
    label: `${name}_c`, size: n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const ub = new ArrayBuffer(16);
  new Uint32Array(ub, 0, 1)[0] = n;
  if (extraFloat !== null) new Float32Array(ub, 4, 1)[0] = extraFloat;
  const uniforms = device.createBuffer({
    size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniforms, 0, ub);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: aBuf } },
      { binding: 1, resource: { buffer: cBuf } },
      { binding: 2, resource: { buffer: uniforms } },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(n / 64));
  pass.end();
  device.queue.submit([enc.finish()]);
  const result = await downloadF32(device, cBuf, n * 4);
  aBuf.destroy(); cBuf.destroy(); uniforms.destroy();
  return result;
}

// ---------------------------------------------------------------------
// Linear layer: y = x @ weight.T (no bias). Weight tensor lives on GPU
// from loadModel; pass its name (e.g. "transformer.h.0.attn.c_q.weight").
// ---------------------------------------------------------------------
export async function linear(device, model, x, M, K, weightName) {
  const w = model.tensors.get(weightName);
  if (!w) throw new Error(`linear: weight ${weightName} not in manifest`);
  const [N, Kw] = w.shape;
  if (Kw !== K) throw new Error(`linear shape mismatch: x has K=${K}, weight has K=${Kw}`);

  const pipeline = await getPipeline(device, model, "linear");
  const xBuf = uploadF32(device, "linear_x", x);
  const yBuf = device.createBuffer({
    label: "linear_y", size: M * N * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const ub = new ArrayBuffer(16);
  const u32 = new Uint32Array(ub);
  u32[0] = M; u32[1] = K; u32[2] = N;
  const uniforms = device.createBuffer({
    size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniforms, 0, ub);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: xBuf } },
      { binding: 1, resource: { buffer: w.buffer } },
      { binding: 2, resource: { buffer: yBuf } },
      { binding: 3, resource: { buffer: uniforms } },
    ],
  });

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(M / 8), Math.ceil(N / 8));
  pass.end();
  device.queue.submit([enc.finish()]);

  const result = await downloadF32(device, yBuf, M * N * 4);
  xBuf.destroy(); yBuf.destroy(); uniforms.destroy();
  return result;
}

// ---------------------------------------------------------------------
// RoPE: rotate the last dim of x using model.cos / model.sin at time T0.
// x has shape (N, D) where D = head_dim; cos/sin have shape
// (rotary_seq_len, d) with d = D/2 (these buffers already live on the GPU
// from loadModel).
// ---------------------------------------------------------------------
export async function ropeApply(device, model, x, N, D, T0 = 0) {
  if (D % 2 !== 0) throw new Error("RoPE: head_dim must be even");
  const d = D / 2;
  const pipeline = await getPipeline(device, model, "rope");
  const cos = model.tensors.get("cos");
  const sin = model.tensors.get("sin");
  if (!cos || !sin) throw new Error("RoPE: cos/sin buffers missing from manifest");

  const xBuf = uploadF32(device, "rope_x", x);
  const yBuf = device.createBuffer({
    label: "rope_y", size: N * D * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const ub = new ArrayBuffer(16);
  const u32 = new Uint32Array(ub);
  u32[0] = N; u32[1] = D; u32[2] = d; u32[3] = T0;
  const uniforms = device.createBuffer({
    size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniforms, 0, ub);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: xBuf } },
      { binding: 1, resource: { buffer: cos.buffer } },
      { binding: 2, resource: { buffer: sin.buffer } },
      { binding: 3, resource: { buffer: yBuf } },
      { binding: 4, resource: { buffer: uniforms } },
    ],
  });

  const total = N * D;
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(total / 64));
  pass.end();
  device.queue.submit([enc.finish()]);

  const result = await downloadF32(device, yBuf, N * D * 4);
  xBuf.destroy(); yBuf.destroy(); uniforms.destroy();
  return result;
}

// ---------------------------------------------------------------------
// KV cache: pre-allocated per-layer K, V buffers + seqlens counter.
// Used by attentionBlock when passed; enables multi-token generation.
// ---------------------------------------------------------------------
export function initKVCache(device, model, maxSeqLen) {
  const cfg = model.config;
  const head_dim = cfg.n_embd / cfg.n_head;
  const kvDim = cfg.n_kv_head * head_dim;
  const bytes = maxSeqLen * kvDim * 4;

  const cache = {
    maxSeqLen,
    seqlens: 0,
    kBuffers: [],
    vBuffers: [],
    prevEmbedding: null,   // for smear
  };
  for (let i = 0; i < cfg.n_layer; i++) {
    cache.kBuffers.push(device.createBuffer({
      label: `k_cache_${i}`, size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    }));
    cache.vBuffers.push(device.createBuffer({
      label: `v_cache_${i}`, size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    }));
  }
  return cache;
}

export function destroyKVCache(cache) {
  for (const b of cache.kBuffers) b.destroy();
  for (const b of cache.vBuffers) b.destroy();
}

// ---------------------------------------------------------------------
// Internal: pure-GPU SDPA pieces, used by attention-with-cache path.
// These keep data on-GPU between stages, unlike the test-friendly
// softmax() helper above.
// ---------------------------------------------------------------------
async function sdpaScoresGPU(device, model, qBuf, kCacheBuf, scoresBuf, nH, nKV, hd, Tk) {
  const pipeline = await getPipeline(device, model, "sdpa_scores");
  const ub = new ArrayBuffer(32);
  const u32 = new Uint32Array(ub);
  const f32 = new Float32Array(ub);
  u32[0] = nH; u32[1] = nKV; u32[2] = hd; u32[3] = Tk;
  f32[4] = 1.0 / Math.sqrt(hd);
  const uniforms = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(uniforms, 0, ub);
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: qBuf } },
      { binding: 1, resource: { buffer: kCacheBuf } },
      { binding: 2, resource: { buffer: scoresBuf } },
      { binding: 3, resource: { buffer: uniforms } },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.ceil(nH / 8), Math.ceil(Tk / 8));
  pass.end();
  device.queue.submit([enc.finish()]);
  uniforms.destroy();
}

async function softmaxRowsGPU(device, model, scoresBuf, rows, n) {
  const pipeline = await getPipeline(device, model, "softmax");
  const ub = new ArrayBuffer(16);
  const u32 = new Uint32Array(ub);
  u32[0] = rows; u32[1] = n;
  const uniforms = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(uniforms, 0, ub);
  // softmax.wgsl reads x and writes y to different buffers. Use a tmp.
  const tmp = device.createBuffer({
    size: rows * n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const enc0 = device.createCommandEncoder();
  enc0.copyBufferToBuffer(scoresBuf, 0, tmp, 0, rows * n * 4);
  device.queue.submit([enc0.finish()]);

  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: tmp } },
      { binding: 1, resource: { buffer: scoresBuf } },
      { binding: 2, resource: { buffer: uniforms } },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(rows);
  pass.end();
  device.queue.submit([enc.finish()]);
  tmp.destroy(); uniforms.destroy();
}

async function sdpaOutputGPU(device, model, attnBuf, vCacheBuf, outBuf, nH, nKV, hd, Tk) {
  const pipeline = await getPipeline(device, model, "sdpa_output");
  const ub = new ArrayBuffer(16);
  const u32 = new Uint32Array(ub);
  u32[0] = nH; u32[1] = nKV; u32[2] = hd; u32[3] = Tk;
  const uniforms = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(uniforms, 0, ub);
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: attnBuf } },
      { binding: 1, resource: { buffer: vCacheBuf } },
      { binding: 2, resource: { buffer: outBuf } },
      { binding: 3, resource: { buffer: uniforms } },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.ceil(nH / 8), Math.ceil(hd / 8));
  pass.end();
  device.queue.submit([enc.finish()]);
  uniforms.destroy();
}

// ---------------------------------------------------------------------
// Row-wise softmax. x shape (rows, n); y same shape.
// ---------------------------------------------------------------------
export async function softmax(device, model, x, rows, n) {
  const pipeline = await getPipeline(device, model, "softmax");
  const xBuf = uploadF32(device, "softmax_x", x);
  const yBuf = device.createBuffer({
    label: "softmax_y", size: rows * n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const ub = new ArrayBuffer(16);
  const u32 = new Uint32Array(ub);
  u32[0] = rows; u32[1] = n;
  const uniforms = device.createBuffer({
    size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniforms, 0, ub);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: xBuf } },
      { binding: 1, resource: { buffer: yBuf } },
      { binding: 2, resource: { buffer: uniforms } },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(rows);
  pass.end();
  device.queue.submit([enc.finish()]);
  const result = await downloadF32(device, yBuf, rows * n * 4);
  xBuf.destroy(); yBuf.destroy(); uniforms.destroy();
  return result;
}

// ---------------------------------------------------------------------
// Softcap: y = cap * tanh(x / cap). Element-wise unary.
// ---------------------------------------------------------------------
export async function softcap(device, model, a, cap = 15.0) {
  return _elemUnary(device, model, "softcap", a, cap);
}

// ---------------------------------------------------------------------
// Full forward pass for a single token. No KV cache (fresh state).
// Returns logits (vocab_size,).
// ---------------------------------------------------------------------
export async function forward(device, model, tokenId, cache = null) {
  const cfg = model.config;
  const n_layer = cfg.n_layer;
  const n_embd = cfg.n_embd;
  const T0 = cache ? cache.seqlens : 0;

  // 1. Embedding + norm
  let x = await embeddingLookup(device, model, tokenId);
  x = await rmsnorm(device, model, x);

  // 1b. Smear (decode T=1). PyTorch reference (gpt.py ~l.435):
  //       x_pre_smear = cache.prev_embedding
  //       cache.prev_embedding = x                   # store POST-norm PRE-smear
  //       if x_pre_smear is not None:
  //           gate = smear_lambda * sigmoid(smear_gate(x[..., :24]))
  //           x = x + gate * x_pre_smear
  if (cache) {
    const xPreSmear = cache.prevEmbedding;
    cache.prevEmbedding = new Float32Array(x);

    if (xPreSmear !== null) {
      if (model.smearLambda === undefined) {
        model.smearLambda = (await downloadF32(device, model.tensors.get("smear_lambda").buffer, 4))[0];
      }
      const xHead = x.slice(0, 24);
      const gateRaw = await linear(device, model, xHead, 1, 24, "smear_gate.weight");
      const gateSig = await sigmoid(device, model, gateRaw);
      const gateScaled = await scalarMul(device, model, gateSig, model.smearLambda);
      const gateBroadcast = new Float32Array(n_embd).fill(gateScaled[0]);
      const smeared = await mul(device, model, gateBroadcast, xPreSmear);
      x = await add(device, model, x, smeared);
    }
  }

  const x0 = new Float32Array(x);   // x0 baseline = post-smear normalized embedding

  // 2. Transformer blocks
  const backoutIdx = Math.floor(n_layer / 2);
  let xBackout = null;
  for (let i = 0; i < n_layer; i++) {
    let ve = null;
    const veName = `value_embeds.${i}.weight`;
    if (model.tensors.has(veName)) {
      ve = await embeddingLookupNamed(device, model, veName, tokenId);
    }
    x = await transformerBlock(device, model, i, x, x0, T0, ve, cache);
    if (i === backoutIdx) xBackout = new Float32Array(x);
  }

  if (cache) cache.seqlens += 1;

  // 3. Backout subtract: x = x - backout_lambda * x_backout
  if (xBackout !== null) {
    if (model.backoutLambda === undefined) {
      model.backoutLambda = (await downloadF32(device, model.tensors.get("backout_lambda").buffer, 4))[0];
    }
    const negScaled = await scalarMul(device, model, xBackout, -model.backoutLambda);
    x = await add(device, model, x, negScaled);
  }

  // 4. Final norm + lm_head
  x = await rmsnorm(device, model, x);
  const paddedVocab = model.tensors.get("lm_head.weight").shape[0];
  const logitsPadded = await linear(device, model, x, 1, n_embd, "lm_head.weight");

  // 5. Slice to true vocab and softcap
  const logits = logitsPadded.slice(0, cfg.vocab_size);
  return await softcap(device, model, logits, 15.0);
}

// ---------------------------------------------------------------------
// Full transformer block (one layer). Mirrors the outer loop in
// GPT.forward(), which does:
//   x = resid_lambdas[i] * x + x0_lambdas[i] * x0
//   x = x + block.attn(norm(x), ve, cos_sin, window, kv_cache)
//   x = x + block.mlp(norm(x))
// We cache the tiny per-layer scalar arrays on the model the first
// time we need them.
// ---------------------------------------------------------------------
async function ensureScalars(device, model) {
  if (model.residLambdas) return;
  const n = model.config.n_layer;
  model.residLambdas = await downloadF32(device, model.tensors.get("resid_lambdas").buffer, n * 4);
  model.x0Lambdas    = await downloadF32(device, model.tensors.get("x0_lambdas").buffer,    n * 4);
}

export async function transformerBlock(device, model, layerIdx, x, x0, T0 = 0, ve = null, cache = null) {
  await ensureScalars(device, model);
  const residL = model.residLambdas[layerIdx];
  const x0L    = model.x0Lambdas[layerIdx];

  // 1. Per-layer residual mix
  const resid = await scalarMul(device, model, x, residL);
  const x0m   = await scalarMul(device, model, x0, x0L);
  let   cur   = await add(device, model, resid, x0m);

  // 2. Attention + residual
  const xn    = await rmsnorm(device, model, cur);
  const ao    = await attentionBlock(device, model, layerIdx, xn, T0, ve, cache);
  cur         = await add(device, model, cur, ao);

  // 3. MLP + residual
  const xn2   = await rmsnorm(device, model, cur);
  const mo    = await mlpBlock(device, model, layerIdx, xn2);
  cur         = await add(device, model, cur, mo);

  return cur;
}

// ---------------------------------------------------------------------
// Attention block for one transformer layer, T=1, no KV cache.
//
// For T=1 and no cache, the attention mechanism trivialises: there is
// exactly one key/value and one query. softmax of a single score = 1.0,
// so attn_output equals V unchanged. We still compute Q/K projections
// and apply RoPE + QK norm + scaling so that later phases (with proper
// SDPA over a KV cache) can reuse this scaffolding directly.
//
// This phase doesn't test softmax numerics — that's Phase 11's job.
// What it DOES validate: Q/K/V projections, head-wise RoPE, head-wise
// QK norm, per-layer value embedding hookup (Phase 7), c_proj chain.
// ---------------------------------------------------------------------
export async function attentionBlock(device, model, layerIdx, x, T0 = 0, ve = null, cache = null) {
  const prefix = `transformer.h.${layerIdx}.attn`;
  const cfg = model.config;
  const n_embd = cfg.n_embd;
  const n_head = cfg.n_head;
  const n_kv = cfg.n_kv_head;
  const head_dim = n_embd / n_head;
  const M = x.length / n_embd;   // batch*time rows; for now = 1

  // 1. Q, K, V projections
  const q = await linear(device, model, x, M, n_embd, `${prefix}.c_q.weight`);
  const k = await linear(device, model, x, M, n_embd, `${prefix}.c_k.weight`);
  let   v = await linear(device, model, x, M, n_embd, `${prefix}.c_v.weight`);

  // 2. Value residual (ResFormer). Added in Phase 7.
  //    if ve !== null: v = v + gate * ve, where gate = 3 * sigmoid(ve_gate(x[..., :12]))
  if (ve !== null) {
    // ve_gate weight shape: (n_kv, 12). Input is first 12 elements of x per row.
    const xHead = x.slice(0, M * 12);   // contiguous for M=1
    const gateRaw = await linear(device, model, xHead, M, 12, `${prefix}.ve_gate.weight`);
    const gateSig = await sigmoid(device, model, gateRaw);
    const gateTriple = await scalarMul(device, model, gateSig, 3.0);
    // gateTriple has shape (M, n_kv). Broadcast across head_dim against ve and v.
    // v shape (M, n_kv, head_dim) flat. For each (m, h), scale ve[m,h,:] by gateTriple[m,h]
    // Use a tiny helper: expand gateTriple to match v's shape, then element-wise mul.
    const gateExpanded = new Float32Array(v.length);
    for (let m = 0; m < M; m++) {
      for (let h = 0; h < n_kv; h++) {
        const g = gateTriple[m * n_kv + h];
        for (let d = 0; d < head_dim; d++) {
          gateExpanded[m * n_kv * head_dim + h * head_dim + d] = g;
        }
      }
    }
    const veScaled = await mul(device, model, gateExpanded, ve);
    v = await add(device, model, v, veScaled);
  }

  // 3. RoPE on Q, K (in-place logically; ropeApply returns new array)
  const qR = await ropeApply(device, model, q, n_head, head_dim, T0);
  const kR = await ropeApply(device, model, k, n_kv, head_dim, T0);

  // 4. QK norm: rmsnorm each head's (head_dim,) vector separately
  const qN = new Float32Array(qR.length);
  const kN = new Float32Array(kR.length);
  for (let h = 0; h < n_head; h++) {
    const row = qR.slice(h * head_dim, (h + 1) * head_dim);
    qN.set(await rmsnorm(device, model, row), h * head_dim);
  }
  for (let h = 0; h < n_kv; h++) {
    const row = kR.slice(h * head_dim, (h + 1) * head_dim);
    kN.set(await rmsnorm(device, model, row), h * head_dim);
  }

  // 5. Scale by 1.2
  const qS = await scalarMul(device, model, qN, 1.2);
  const kS = await scalarMul(device, model, kN, 1.2);

  // 6. SDPA.
  //    Without cache (Phase 6 original path): T=1 → output = v.
  //    With cache: append new K/V at cache.seqlens, then run SDPA over [0..seqlens+1].
  let attnOut;
  if (cache === null) {
    attnOut = v;
  } else {
    // Append current k, v to cache at position cache.seqlens
    const kvDim = n_kv * head_dim;
    const writeOffset = cache.seqlens * kvDim * 4;
    device.queue.writeBuffer(cache.kBuffers[layerIdx], writeOffset, kS.buffer, kS.byteOffset, kS.byteLength);
    device.queue.writeBuffer(cache.vBuffers[layerIdx], writeOffset, v.buffer,  v.byteOffset,  v.byteLength);
    const Tk = cache.seqlens + 1;

    // Allocate scratch buffers for this step
    const qBuf    = uploadF32(device, "sdpa_q", qS);
    const scBuf   = device.createBuffer({
      size: n_head * Tk * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const outBuf  = device.createBuffer({
      size: n_head * head_dim * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    await sdpaScoresGPU (device, model, qBuf, cache.kBuffers[layerIdx], scBuf, n_head, n_kv, head_dim, Tk);
    await softmaxRowsGPU(device, model, scBuf, n_head, Tk);
    await sdpaOutputGPU (device, model, scBuf, cache.vBuffers[layerIdx], outBuf, n_head, n_kv, head_dim, Tk);

    attnOut = await downloadF32(device, outBuf, n_head * head_dim * 4);
    qBuf.destroy(); scBuf.destroy(); outBuf.destroy();
  }

  // 7. c_proj
  return await linear(device, model, attnOut, M, n_embd, `${prefix}.c_proj.weight`);
}

// ---------------------------------------------------------------------
// MLP block for one transformer layer.
//   y = c_proj(relu²(c_fc(x)))
// ---------------------------------------------------------------------
export async function mlpBlock(device, model, layerIdx, x) {
  const prefix = `transformer.h.${layerIdx}.mlp`;
  const n_embd = model.config.n_embd;
  const hidden = 4 * n_embd;           // c_fc expands by 4×
  const M = x.length / n_embd;         // batch*time rows
  const h = await linear(device, model, x, M, n_embd, `${prefix}.c_fc.weight`);
  const a = await relu2(device, model, h);
  const y = await linear(device, model, a, M, hidden, `${prefix}.c_proj.weight`);
  return y;
}

export const add         = (device, model, a, b)        => _elemBinary(device, model, "add", a, b);
export const mul         = (device, model, a, b)        => _elemBinary(device, model, "mul", a, b);
export const scalarMul   = (device, model, a, alpha)    => _elemUnary (device, model, "scalar_mul", a, alpha);
export const relu2       = (device, model, a)           => _elemUnary (device, model, "relu2", a);
export const sigmoid     = (device, model, a)           => _elemUnary (device, model, "sigmoid", a);

// ---------------------------------------------------------------------
// 8. RMSNorm.
// ---------------------------------------------------------------------
//
// Equivalent in PyTorch:
//     y = F.rms_norm(x, (x.size(-1),), eps=1e-5)   # no weight
//
// `input` is a Float32Array of length n (one row). Returns a new
// Float32Array of length n with the row normalized. Uses 1 workgroup
// of 64 threads internally (the shader does the reduction).
// ---------------------------------------------------------------------
export async function rmsnorm(device, model, input, eps = 1e-5) {
  const pipeline = await getPipeline(device, model, "rmsnorm");
  const n = input.length;

  const inBuf = uploadF32(device, "rmsnorm_in", input);
  const outBuf = device.createBuffer({
    label: "rmsnorm_out",
    size: n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // Uniform layout: u32 n; f32 eps; padding; (16-byte alignment)
  const uniforms = device.createBuffer({
    size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  // Use a typed-array view: 0=u32 n, 1=f32 eps, 2..3=padding
  const ub = new ArrayBuffer(16);
  new Uint32Array(ub, 0, 1)[0] = n;
  new Float32Array(ub, 4, 1)[0] = eps;
  device.queue.writeBuffer(uniforms, 0, ub);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inBuf } },
      { binding: 1, resource: { buffer: outBuf } },
      { binding: 2, resource: { buffer: uniforms } },
    ],
  });

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1);   // single row, single workgroup
  pass.end();
  device.queue.submit([enc.finish()]);

  const result = await downloadF32(device, outBuf, n * 4);
  inBuf.destroy(); outBuf.destroy(); uniforms.destroy();
  return result;
}
