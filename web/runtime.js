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
// 4. Tensor abstraction.
// ---------------------------------------------------------------------
//
// A `Tensor` is just `{buffer, shape, byteLength, label}` — a GPUBuffer
// plus its declared shape. All on-GPU ops in the new fast path take
// Tensors in and return Tensors out, never touching CPU until the very
// end of the forward pass.
//
// We keep the `Float32Array → primitive → Float32Array` test-friendly
// API as thin wrappers (uploadTensor + op_t + downloadTensor) so the
// existing test suite still works while the refactor lands piece by
// piece.
//
// Buffer pool (`BufferPool`) recycles GPUBuffers by (size, usage) to
// avoid the per-call createBuffer/destroy that dominates today's
// runtime. Caller responsibilities:
//   - allocTensor       → pool.acquire under the hood
//   - releaseTensor     → returns its buffer to the pool
// Pool buffers are zero-initialised by writeBuffer/dispatch on the
// first use; we never read from them before writing, so leftover bytes
// from a previous use are harmless.
// ---------------------------------------------------------------------
const TENSOR_USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

export class BufferPool {
  constructor() {
    this.free = new Map();   // key -> [GPUBuffer, ...]
    this.outstanding = 0;
  }
  _key(size, usage) { return `${size}/${usage}`; }
  acquire(device, size, usage = TENSOR_USAGE, label = "pool") {
    const k = this._key(size, usage);
    const list = this.free.get(k);
    if (list && list.length) {
      this.outstanding++;
      return list.pop();
    }
    this.outstanding++;
    return device.createBuffer({ label, size, usage });
  }
  release(buf) {
    if (!buf) return;
    const k = this._key(buf.size, buf.usage);
    let list = this.free.get(k);
    if (!list) { list = []; this.free.set(k, list); }
    list.push(buf);
    this.outstanding--;
  }
  destroy() {
    for (const list of this.free.values()) for (const b of list) b.destroy();
    this.free.clear();
  }
}

function _ensurePool(model) {
  if (!model.pool) model.pool = new BufferPool();
  return model.pool;
}

// Allocate a Tensor of `shape` (number[]). Uses the model's pool.
export function allocTensor(device, model, shape, label = "tensor", usage = TENSOR_USAGE) {
  const numel = shape.reduce((a, b) => a * b, 1);
  const byteLength = numel * 4;
  const buffer = _ensurePool(model).acquire(device, byteLength, usage, label);
  return { buffer, shape, byteLength, label, _pooled: true };
}

// Wrap an existing GPUBuffer (e.g. weight tensor from loadModel) as a Tensor
// without going through the pool. Caller owns lifecycle.
export function wrapTensor(buffer, shape, label = "wrapped") {
  return { buffer, shape, byteLength: shape.reduce((a, b) => a * b, 1) * 4, label, _pooled: false };
}

// Upload a Float32Array as a new Tensor. Goes through the pool.
export function uploadTensor(device, model, arr, shape = null, label = "uploaded") {
  if (shape === null) shape = [arr.length];
  const t = allocTensor(device, model, shape, label);
  device.queue.writeBuffer(t.buffer, 0, arr);
  return t;
}

// Read a Tensor back to CPU as a Float32Array. Synchronous-style (await).
// Does NOT release the source tensor — caller decides.
export async function downloadTensor(device, t) {
  const staging = device.createBuffer({
    size: t.byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(t.buffer, 0, staging, 0, t.byteLength);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap(); staging.destroy();
  return out;
}

// Return a tensor's buffer to the pool. After this the tensor must not
// be used again. Safe on non-pooled tensors (no-op).
export function releaseTensor(model, t) {
  if (t && t._pooled) _ensurePool(model).release(t.buffer);
}

// ---------------------------------------------------------------------
// 4b. Legacy: turn a Float32Array into a one-shot STORAGE GPUBuffer.
// Kept for the existing primitives until they migrate to Tensor in/out.
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
// 4c. Dispatch helper: pool a 16-byte (or 32-byte) uniform buffer,
// build a bind group, submit one dispatch, return. Caller releases the
// uniform buffer via the pool after the next idle cycle (we just queue
// and forget — WebGPU queue is in-order so the next acquire/use is safe).
// ---------------------------------------------------------------------
const UNIFORM_USAGE = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

function _writeUniforms(device, model, u32s, f32Pairs = []) {
  // u32s: array of u32 values starting at offset 0.
  // f32Pairs: array of [byteOffset, f32Value] for f32 fields mixed in.
  const minSize = Math.max(16, u32s.length * 4);
  const size = (minSize + 15) & ~15;
  const buf = _ensurePool(model).acquire(device, size, UNIFORM_USAGE, "uniforms");
  const ab = new ArrayBuffer(size);
  const u32 = new Uint32Array(ab);
  for (let i = 0; i < u32s.length; i++) u32[i] = u32s[i];
  if (f32Pairs.length) {
    const dv = new DataView(ab);
    for (const [off, v] of f32Pairs) dv.setFloat32(off, v, true);
  }
  device.queue.writeBuffer(buf, 0, ab);
  return buf;
}

// Submit a single compute dispatch. `bufferBindings` is an array of
// GPUBuffers in binding-index order (bindings 0..N-1); `uniformsBuf` is
// bound at the next index. `dispatch` is [x] or [x, y] or [x, y, z].
function _dispatch(device, pipeline, bufferBindings, uniformsBuf, dispatch) {
  const entries = bufferBindings.map((b, i) => ({ binding: i, resource: { buffer: b } }));
  if (uniformsBuf) entries.push({ binding: bufferBindings.length, resource: { buffer: uniformsBuf } });
  const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(dispatch[0], dispatch[1] ?? 1, dispatch[2] ?? 1);
  pass.end();
  device.queue.submit([enc.finish()]);
}

// ---------------------------------------------------------------------
// 4d. Tensor-typed primitives (the "fast path" — Tensor in, Tensor out,
// no CPU round-trip). The Float32Array-style primitives below are now
// thin wrappers over these.
// ---------------------------------------------------------------------

export async function rmsnormT(device, model, x, eps = 1e-5) {
  const n = x.shape[x.shape.length - 1];
  const pipeline = await getPipeline(device, model, "rmsnorm");
  const y = allocTensor(device, model, x.shape, "rmsnorm_y");
  const u = _writeUniforms(device, model, [n], [[4, eps]]);
  _dispatch(device, pipeline, [x.buffer, y.buffer], u, [1]);   // single row, single workgroup
  _ensurePool(model).release(u);
  return y;
}

async function _elemBinaryT(device, model, name, a, b) {
  if (a.byteLength !== b.byteLength) throw new Error(`${name}T: shape mismatch ${a.byteLength} vs ${b.byteLength}`);
  const n = a.byteLength / 4;
  const pipeline = await getPipeline(device, model, name);
  const c = allocTensor(device, model, a.shape, `${name}_out`);
  const u = _writeUniforms(device, model, [n]);
  _dispatch(device, pipeline, [a.buffer, b.buffer, c.buffer], u, [Math.ceil(n / 64)]);
  _ensurePool(model).release(u);
  return c;
}

async function _elemUnaryT(device, model, name, a, extraF32 = null) {
  const n = a.byteLength / 4;
  const pipeline = await getPipeline(device, model, name);
  const c = allocTensor(device, model, a.shape, `${name}_out`);
  const u = _writeUniforms(device, model, [n], extraF32 !== null ? [[4, extraF32]] : []);
  _dispatch(device, pipeline, [a.buffer, c.buffer], u, [Math.ceil(n / 64)]);
  _ensurePool(model).release(u);
  return c;
}

export const addT       = (d, m, a, b)         => _elemBinaryT(d, m, "add", a, b);
export const mulT       = (d, m, a, b)         => _elemBinaryT(d, m, "mul", a, b);
export const scalarMulT = (d, m, a, alpha)     => _elemUnaryT (d, m, "scalar_mul", a, alpha);
export const relu2T     = (d, m, a)            => _elemUnaryT (d, m, "relu2", a);
export const sigmoidT   = (d, m, a)            => _elemUnaryT (d, m, "sigmoid", a);
export const softcapT   = (d, m, a, cap = 15)  => _elemUnaryT (d, m, "softcap", a, cap);

// RoPE on Tensor x of shape (N, D). T0 = position offset into cos/sin.
export async function ropeApplyT(device, model, x, T0 = 0) {
  const D = x.shape[x.shape.length - 1];
  if (D % 2 !== 0) throw new Error("ropeT: D must be even");
  const N = x.shape.length === 1 ? 1 : x.shape[0];
  const d = D / 2;
  const pipeline = await getPipeline(device, model, "rope");
  const cos = model.tensors.get("cos"), sin = model.tensors.get("sin");
  if (!cos || !sin) throw new Error("ropeT: cos/sin missing");
  const y = allocTensor(device, model, x.shape, "rope_y");
  const u = _writeUniforms(device, model, [N, D, d, T0]);
  _dispatch(device, pipeline, [x.buffer, cos.buffer, sin.buffer, y.buffer], u,
            [Math.ceil(N * D / 64)]);
  _ensurePool(model).release(u);
  return y;
}

// Row-wise softmax. x shape (rows, n).
export async function softmaxT(device, model, x, rows, n) {
  const pipeline = await getPipeline(device, model, "softmax");
  const y = allocTensor(device, model, x.shape, "softmax_y");
  const u = _writeUniforms(device, model, [rows, n]);
  _dispatch(device, pipeline, [x.buffer, y.buffer], u, [rows]);
  _ensurePool(model).release(u);
  return y;
}

// Embedding lookup: a row of `weightName`. Returns Tensor of shape (dim,).
export async function embeddingT(device, model, weightName, tokenId) {
  const w = model.tensors.get(weightName);
  if (!w) throw new Error(`embeddingT: ${weightName} missing`);
  const [vocab, dim] = w.shape;
  const pipeline = await getPipeline(device, model, "embedding");
  const y = allocTensor(device, model, [dim], `${weightName}_row`);
  const u = _writeUniforms(device, model, [tokenId, dim, vocab, 0]);
  _dispatch(device, pipeline, [w.buffer, y.buffer], u, [Math.ceil(dim / 64)]);
  _ensurePool(model).release(u);
  return y;
}

export async function linearT(device, model, x, weightName) {
  const w = model.tensors.get(weightName);
  if (!w) throw new Error(`linearT: ${weightName} missing`);
  const [N, K] = w.shape;
  const M = x.shape.length === 1 ? 1 : x.shape[0];
  if (x.shape[x.shape.length - 1] !== K) {
    throw new Error(`linearT shape mismatch: x last-dim ${x.shape[x.shape.length - 1]} vs weight K=${K}`);
  }
  const pipeline = await getPipeline(device, model, "linear");
  const y = allocTensor(device, model, [M, N], "linear_y");
  const u = _writeUniforms(device, model, [M, K, N, 0]);
  _dispatch(device, pipeline, [x.buffer, w.buffer, y.buffer], u,
            [Math.ceil(M / 8), Math.ceil(N / 8)]);
  _ensurePool(model).release(u);
  return y;
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
// Embedding lookup wrappers. Both call embeddingT under the hood.
// ---------------------------------------------------------------------
export async function embeddingLookupNamed(device, model, weightName, tokenId) {
  const t = await embeddingT(device, model, weightName, tokenId);
  const out = await downloadTensor(device, t);
  releaseTensor(model, t);
  return out;
}

export async function embeddingLookup(device, model, tokenId) {
  return embeddingLookupNamed(device, model, "transformer.wte.weight", tokenId);
}

// ---------------------------------------------------------------------
// 7. Matmul: C (M×N) = A (M×K) @ B (K×N), row-major f32. Test-only.
// ---------------------------------------------------------------------
export async function matmulT(device, model, a, b) {
  const [M, K] = a.shape, [K2, N] = b.shape;
  if (K !== K2) throw new Error(`matmulT: (${M},${K}) @ (${K2},${N})`);
  const pipeline = await getPipeline(device, model, "matmul");
  const c = allocTensor(device, model, [M, N], "matmul_c");
  const u = _writeUniforms(device, model, [M, K, N, 0]);
  _dispatch(device, pipeline, [a.buffer, b.buffer, c.buffer], u, [Math.ceil(M / 8), Math.ceil(N / 8)]);
  _ensurePool(model).release(u);
  return c;
}

export async function matmul(device, model, a, aShape, b, bShape) {
  const aT = uploadTensor(device, model, a, aShape, "matmul_a");
  const bT = uploadTensor(device, model, b, bShape, "matmul_b");
  const cT = await matmulT(device, model, aT, bT);
  const out = await downloadTensor(device, cT);
  releaseTensor(model, aT); releaseTensor(model, bT); releaseTensor(model, cT);
  return out;
}

// ---------------------------------------------------------------------
// Float32Array element-wise wrappers. Each calls the corresponding T
// version. Bigger composites use the T versions directly.
// ---------------------------------------------------------------------
async function _binWrap(device, model, name, aArr, bArr) {
  const aT = uploadTensor(device, model, aArr);
  const bT = uploadTensor(device, model, bArr);
  const cT = await _elemBinaryT(device, model, name, aT, bT);
  const out = await downloadTensor(device, cT);
  releaseTensor(model, aT); releaseTensor(model, bT); releaseTensor(model, cT);
  return out;
}
async function _unWrap(device, model, name, aArr, extra = null) {
  const aT = uploadTensor(device, model, aArr);
  const cT = await _elemUnaryT(device, model, name, aT, extra);
  const out = await downloadTensor(device, cT);
  releaseTensor(model, aT); releaseTensor(model, cT);
  return out;
}

// ---------------------------------------------------------------------
// Float32Array → Float32Array wrapper for tests. Internally uploads,
// dispatches via linearT, downloads, releases. Composites should call
// linearT directly so they never touch CPU.
// ---------------------------------------------------------------------
export async function linear(device, model, x, M, K, weightName) {
  const xT = uploadTensor(device, model, x, [M, K], "linear_x");
  const yT = await linearT(device, model, xT, weightName);
  const out = await downloadTensor(device, yT);
  releaseTensor(model, xT); releaseTensor(model, yT);
  return out;
}

// Float32Array wrapper around ropeApplyT.
export async function ropeApply(device, model, x, N, D, T0 = 0) {
  const xT = uploadTensor(device, model, x, [N, D], "rope_x");
  const yT = await ropeApplyT(device, model, xT, T0);
  const out = await downloadTensor(device, yT);
  releaseTensor(model, xT); releaseTensor(model, yT);
  return out;
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

// Float32Array wrapper around softmaxT.
export async function softmax(device, model, x, rows, n) {
  const xT = uploadTensor(device, model, x, [rows, n], "softmax_x");
  const yT = await softmaxT(device, model, xT, rows, n);
  const out = await downloadTensor(device, yT);
  releaseTensor(model, xT); releaseTensor(model, yT);
  return out;
}

// Float32Array wrapper around softcapT.
export async function softcap(device, model, a, cap = 15.0) {
  return _unWrap(device, model, "softcap", a, cap);
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
// Greedy generation: prefill prompt, then sample argmax `maxNew` times.
// Returns full token sequence (prompt + generated). Temperature 0.
//
// Each step is an independent forward(token, cache) call — the cache
// carries K/V across positions, smear fires from step 2+. We argmax
// in JS (logits already on the CPU); a one-element argmax shader would
// add complexity for no gain at d12 vocab=64K.
// ---------------------------------------------------------------------
function _argmax(arr) {
  let best = 0, bv = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > bv) { bv = arr[i]; best = i; }
  }
  return best;
}

export async function greedyGenerate(device, model, promptTokens, maxNew, maxSeqLen = null) {
  if (!Array.isArray(promptTokens) || promptTokens.length === 0) {
    throw new Error("greedyGenerate: promptTokens must be a non-empty array");
  }
  const total = promptTokens.length + maxNew;
  const cache = initKVCache(device, model, maxSeqLen ?? total);

  let lastLogits = null;
  for (const tok of promptTokens) {
    lastLogits = await forward(device, model, tok, cache);
  }

  const out = promptTokens.slice();
  for (let i = 0; i < maxNew; i++) {
    const next = _argmax(lastLogits);
    out.push(next);
    if (i < maxNew - 1) lastLogits = await forward(device, model, next, cache);
  }

  destroyKVCache(cache);
  return out;
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

export const add       = (d, m, a, b)     => _binWrap(d, m, "add", a, b);
export const mul       = (d, m, a, b)     => _binWrap(d, m, "mul", a, b);
export const scalarMul = (d, m, a, alpha) => _unWrap (d, m, "scalar_mul", a, alpha);
export const relu2     = (d, m, a)        => _unWrap (d, m, "relu2", a);
export const sigmoid   = (d, m, a)        => _unWrap (d, m, "sigmoid", a);

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
// Float32Array wrapper around rmsnormT.
export async function rmsnorm(device, model, input, eps = 1e-5) {
  const xT = uploadTensor(device, model, input, [input.length], "rmsnorm_in");
  const yT = await rmsnormT(device, model, xT, eps);
  const out = await downloadTensor(device, yT);
  releaseTensor(model, xT); releaseTensor(model, yT);
  return out;
}
