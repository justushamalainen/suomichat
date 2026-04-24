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
//
// If `model.activeSession` is set, releases are deferred until that
// session's submit() so the buffer isn't recycled while the GPU may
// still be reading it from a not-yet-submitted dispatch.
export function releaseTensor(model, t) {
  if (POOL_DEBUG_NO_RELEASE) return;
  if (!t || !t._pooled) return;
  if (model.activeSession) {
    model.activeSession.deferRelease(t.buffer);
  } else {
    _ensurePool(model).release(t.buffer);
  }
}

// DEBUG: set true to make releaseTensor a no-op (each tensor gets a
// fresh buffer; pool only grows). Use to isolate pool-reuse races.
export const POOL_DEBUG_NO_RELEASE = false;

// ---------------------------------------------------------------------
// 4d-bis. Bind-group cache.
//
// createBindGroup is ~0.2 ms in Chromium (validation overhead). With
// 310 dispatches per forward, that's ~62 ms wasted on bind-group
// objects we already built last forward. Buffers come from the pool
// → identical GPUBuffer objects across forwards → identical bind
// groups → cacheable.
//
// Key: concatenated buffer ids (each buffer gets a stable id via
// WeakMap on first sight) + pipeline label.
// ---------------------------------------------------------------------
const _bufferIds = new WeakMap();
let _nextBufferId = 0;
function _bufferId(buf) {
  let id = _bufferIds.get(buf);
  if (id === undefined) { id = _nextBufferId++; _bufferIds.set(buf, id); }
  return id;
}

function _bindGroupKey(pipeline, buffers, uniformsBuf) {
  let k = pipeline.label || "?";
  for (const b of buffers) k += "/" + _bufferId(b);
  if (uniformsBuf) k += "u" + _bufferId(uniformsBuf);
  return k;
}

function _getBindGroup(device, model, pipeline, bufferBindings, uniformsBuf) {
  if (!model.bindGroupCache) model.bindGroupCache = new Map();
  const key = _bindGroupKey(pipeline, bufferBindings, uniformsBuf);
  let bg = model.bindGroupCache.get(key);
  if (bg) return bg;
  const entries = bufferBindings.map((b, i) => ({ binding: i, resource: { buffer: b } }));
  if (uniformsBuf) entries.push({ binding: bufferBindings.length, resource: { buffer: uniformsBuf } });
  bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
  model.bindGroupCache.set(key, bg);
  return bg;
}

// ---------------------------------------------------------------------
// 4e. Session: a single command encoder + lazy compute pass that
// accumulates many dispatches and submits ONCE.
//
// Why: each device.queue.submit has ~0.4 ms driver overhead. A single
// forwardT issues ~310 dispatches; submitting each one separately
// burns ~125 ms of pure overhead. Batching them collapses that to one
// submit per forward.
//
// Usage:
//   const sess = beginSession(device, model);
//   try {
//     ... primitives that take sess as last arg ...
//   } finally {
//     await sess.submit();   // ends pass, finishes encoder, queue.submit
//   }
//
// While `model.activeSession` is set, releaseTensor + _writeUniforms
// defer their pool releases so the same buffer isn't recycled
// mid-session (would cause RAW/WAW hazards in the same submit).
// ---------------------------------------------------------------------
export function beginSession(device, model) {
  if (model.activeSession) throw new Error("beginSession: already in a session");
  const sess = {
    device,
    model,
    encoder: device.createCommandEncoder({ label: "session" }),
    pass: null,
    pendingReleases: [],

    _ensurePass() {
      if (!this.pass) this.pass = this.encoder.beginComputePass();
      return this.pass;
    },
    _endPass() {
      if (this.pass) { this.pass.end(); this.pass = null; }
    },

    dispatch(pipeline, bufferBindings, uniformsBuf, dim) {
      const bg = _getBindGroup(device, model, pipeline, bufferBindings, uniformsBuf);
      const pass = this._ensurePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(dim[0], dim[1] ?? 1, dim[2] ?? 1);
    },

    copy(src, srcOff, dst, dstOff, size) {
      this._endPass();   // copy must be outside any compute pass
      this.encoder.copyBufferToBuffer(src, srcOff, dst, dstOff, size);
    },

    deferRelease(buf) { this.pendingReleases.push(buf); },

    async submit() {
      this._endPass();
      device.queue.submit([this.encoder.finish()]);
      const pool = _ensurePool(model);
      for (const b of this.pendingReleases) pool.release(b);
      this.pendingReleases.length = 0;
      model.activeSession = null;
    },
  };
  model.activeSession = sess;
  return sess;
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
  let minSize = Math.max(16, u32s.length * 4);
  for (const [off] of f32Pairs) minSize = Math.max(minSize, off + 4);
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

// Release a transient uniform buffer. Defers if a session is active so
// the buffer isn't reused before the submit that consumes it.
function _releaseUniform(model, buf) {
  if (model.activeSession) {
    model.activeSession.deferRelease(buf);
  } else {
    _ensurePool(model).release(buf);
  }
}

// Submit a single compute dispatch. `bufferBindings` is an array of
// GPUBuffers in binding-index order (bindings 0..N-1); `uniformsBuf` is
// bound at the next index. `dispatch` is [x] or [x, y] or [x, y, z].
//
// If `model.activeSession` is set, the dispatch is appended to the
// session's encoder instead of submitting standalone. Caller passes
// `model` (we read activeSession from it).
function _dispatch(device, model, pipeline, bufferBindings, uniformsBuf, dispatch) {
  if (model.activeSession) {
    model.activeSession.dispatch(pipeline, bufferBindings, uniformsBuf, dispatch);
    return;
  }
  const bg = _getBindGroup(device, model, pipeline, bufferBindings, uniformsBuf);
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

// RMSNorm. x shape (..., n). Each row of length `n` is normalised
// independently. Defaults: `n` = x.shape[-1], `rows` = numel / n.
// Both can be overridden — useful for per-head QK norm where the
// tensor is laid out as (n_head * head_dim) flat and you want each
// head's `head_dim` slice normalized separately.
export async function rmsnormT(device, model, x, eps = 1e-5, rowsOverride = null, nOverride = null) {
  const n = nOverride ?? x.shape[x.shape.length - 1];
  const numel = x.byteLength / 4;
  const rows = rowsOverride ?? (numel / n);
  const pipeline = await getPipeline(device, model, "rmsnorm");
  const y = allocTensor(device, model, x.shape, "rmsnorm_y");
  const u = _writeUniforms(device, model, [n], [[4, eps]]);
  _dispatch(device, model, pipeline, [x.buffer, y.buffer], u, [rows]);
  _releaseUniform(model, u);
  return y;
}

async function _elemBinaryT(device, model, name, a, b) {
  if (a.byteLength !== b.byteLength) throw new Error(`${name}T: shape mismatch ${a.byteLength} vs ${b.byteLength}`);
  const n = a.byteLength / 4;
  const pipeline = await getPipeline(device, model, name);
  const c = allocTensor(device, model, a.shape, `${name}_out`);
  const u = _writeUniforms(device, model, [n]);
  _dispatch(device, model, pipeline, [a.buffer, b.buffer, c.buffer], u, [Math.ceil(n / 64)]);
  _releaseUniform(model, u);
  return c;
}

async function _elemUnaryT(device, model, name, a, extraF32 = null) {
  const n = a.byteLength / 4;
  const pipeline = await getPipeline(device, model, name);
  const c = allocTensor(device, model, a.shape, `${name}_out`);
  const u = _writeUniforms(device, model, [n], extraF32 !== null ? [[4, extraF32]] : []);
  _dispatch(device, model, pipeline, [a.buffer, c.buffer], u, [Math.ceil(n / 64)]);
  _releaseUniform(model, u);
  return c;
}

export const addT       = (d, m, a, b)         => _elemBinaryT(d, m, "add", a, b);
export const mulT       = (d, m, a, b)         => _elemBinaryT(d, m, "mul", a, b);
export const scalarMulT = (d, m, a, alpha)     => _elemUnaryT (d, m, "scalar_mul", a, alpha);
export const relu2T     = (d, m, a)            => _elemUnaryT (d, m, "relu2", a);
export const sigmoidT   = (d, m, a)            => _elemUnaryT (d, m, "sigmoid", a);
export const softcapT   = (d, m, a, cap = 15)  => _elemUnaryT (d, m, "softcap", a, cap);

// RoPE on Tensor x. The shader treats x as (N, D) where D = head_dim
// (NOT n_embd). When x is laid out flat as (n_head*head_dim,) you MUST
// pass headDim explicitly so the rotation operates per-head.
export async function ropeApplyT(device, model, x, T0 = 0, headDim = null, headsPerToken = null) {
  const D = headDim ?? x.shape[x.shape.length - 1];
  if (D % 2 !== 0) throw new Error("ropeT: D must be even");
  const numel = x.byteLength / 4;
  const N = numel / D;
  const d = D / 2;
  const H = headsPerToken ?? N;     // T=1 → all rows pos T0; T>1 → pass n_head/n_kv
  const pipeline = await getPipeline(device, model, "rope");
  const cos = model.tensors.get("cos"), sin = model.tensors.get("sin");
  if (!cos || !sin) throw new Error("ropeT: cos/sin missing");
  const y = allocTensor(device, model, x.shape, "rope_y");
  const u = _writeUniforms(device, model, [N, D, d, T0, H]);
  _dispatch(device, model, pipeline, [x.buffer, cos.buffer, sin.buffer, y.buffer], u,
            [Math.ceil(N * D / 64)]);
  _releaseUniform(model, u);
  return y;
}

// Row-wise softmax. x shape (rows, n).
export async function softmaxT(device, model, x, rows, n) {
  const pipeline = await getPipeline(device, model, "softmax");
  const y = allocTensor(device, model, x.shape, "softmax_y");
  const u = _writeUniforms(device, model, [rows, n]);
  _dispatch(device, model, pipeline, [x.buffer, y.buffer], u, [rows]);
  _releaseUniform(model, u);
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
  _dispatch(device, model, pipeline, [w.buffer, y.buffer], u, [Math.ceil(dim / 64)]);
  _releaseUniform(model, u);
  return y;
}

export async function linearT(device, model, x, weightName) {
  const w = model.tensors.get(weightName);
  if (!w) throw new Error(`linearT: ${weightName} missing`);
  const [N, K] = w.shape;
  const M = x.shape.length === 1 ? 1 : x.shape[0];
  // We allow x's last-dim to be ≥ K — the linear shader indexes x[m*K + k]
  // and reads only the first K elements per row. ve_gate / smear_gate use
  // this to consume a prefix of n_embd without an explicit slice.
  if (x.shape[x.shape.length - 1] < K) {
    throw new Error(`linearT: x last-dim ${x.shape[x.shape.length - 1]} < weight K=${K}`);
  }
  const pipeline = await getPipeline(device, model, "linear");
  const y = allocTensor(device, model, [M, N], "linear_y");
  const u = _writeUniforms(device, model, [M, K, N, 0]);
  _dispatch(device, model, pipeline, [x.buffer, w.buffer, y.buffer], u,
            [Math.ceil(M / 8), Math.ceil(N / 8)]);
  _releaseUniform(model, u);
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
  _dispatch(device, model, pipeline, [a.buffer, b.buffer, c.buffer], u, [Math.ceil(M / 8), Math.ceil(N / 8)]);
  _releaseUniform(model, u);
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
export async function ropeApply(device, model, x, N, D, T0 = 0, H = null) {
  const xT = uploadTensor(device, model, x, [N, D], "rope_x");
  const yT = await ropeApplyT(device, model, xT, T0, D, H);
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
    prevEmbedding: null,    // legacy Float32Array slot (used by old forward())
    prevEmbeddingT: null,   // GPU Tensor slot (used by forwardT)
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
  if (cache.prevEmbeddingT) cache.prevEmbeddingT.buffer.destroy?.();
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
// LEGACY forward — kept for tests that drove the Float32Array smear cache.
// New callers should use forwardT directly.
export async function _forward_legacy(device, model, tokenId, cache = null) {
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

// Float32Array-returning forward (test-friendly). Calls forwardT then
// downloads + slices to vocab_size.
export async function forward(device, model, tokenId, cache = null) {
  const t = await forwardT(device, model, tokenId, cache);
  const arr = await downloadTensor(device, t);
  releaseTensor(model, t);
  return arr.slice(0, model.config.vocab_size);
}

// ---------------------------------------------------------------------
// Argmax (GPU-side). Returns a Tensor whose buffer holds one u32 — the
// argmax index of `x`. The buffer is stored as 4 bytes; downloadU32
// reads it back to JS as a single integer.
// ---------------------------------------------------------------------
export async function argmaxT(device, model, x) {
  const pipeline = await getPipeline(device, model, "argmax");
  const out = allocTensor(device, model, [1], "argmax_out");
  const u = _writeUniforms(device, model, [x.byteLength / 4]);
  _dispatch(device, model, pipeline, [x.buffer, out.buffer], u, [1]);
  _releaseUniform(model, u);
  return out;
}

export async function downloadU32(device, t) {
  const staging = device.createBuffer({
    size: t.byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(t.buffer, 0, staging, 0, t.byteLength);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap(); staging.destroy();
  return out;
}

// In-GPU buffer copy helper. Used for snapshotting cache.prevEmbedding.
// Routes through the active session if there is one (so the copy lands
// in the same encoder as the surrounding compute).
function copyTensor(device, model, src) {
  const dst = allocTensor(device, model, src.shape, (src.label || "tensor") + "_copy");
  _copyBuffer(device, model, src.buffer, 0, dst.buffer, 0, src.byteLength);
  return dst;
}

function _copyBuffer(device, model, src, srcOff, dst, dstOff, size) {
  if (model.activeSession) {
    model.activeSession.copy(src, srcOff, dst, dstOff, size);
    return;
  }
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, srcOff, dst, dstOff, size);
  device.queue.submit([enc.finish()]);
}

// Pre-load scalar constants (resid_lambdas, x0_lambdas, smear_lambda,
// backout_lambda) before any session begins. Inside a session the
// awaiting download would deadlock — the buffer to be read is part of
// the model weights, but the queue's outstanding work hasn't been
// submitted yet, so mapAsync would never resolve.
async function _preloadScalars(device, model) {
  if (model.scalarsLoaded) return;
  await ensureScalars(device, model);
  if (model.tensors.has("smear_lambda") && model.smearLambda === undefined) {
    model.smearLambda = (await downloadF32(device, model.tensors.get("smear_lambda").buffer, 4))[0];
  }
  if (model.tensors.has("backout_lambda") && model.backoutLambda === undefined) {
    model.backoutLambda = (await downloadF32(device, model.tensors.get("backout_lambda").buffer, 4))[0];
  }
  model.scalarsLoaded = true;
}

// ---------------------------------------------------------------------
// forwardT — fully on-GPU forward pass. Input: token id (JS int).
// Output: logits Tensor of shape [vocab_size]. The cache (if provided)
// is mutated in place. Only the token id crosses CPU/GPU at the entry;
// only the caller's downloadTensor (or argmaxT + downloadU32) crosses
// at the exit.
// ---------------------------------------------------------------------
export async function forwardT(device, model, tokenId, cache = null) {
  const cfg = model.config;
  const n_layer = cfg.n_layer;
  const n_embd  = cfg.n_embd;
  const T0 = cache ? cache.seqlens : 0;

  // Scalar constants must be downloaded BEFORE the session — once a
  // session is active, downloads would deadlock (queue work isn't
  // submitted, mapAsync never resolves).
  await _preloadScalars(device, model);

  // Pre-fetch any pipelines we'll use mid-session. getPipeline awaits
  // a fetch on first miss; resolving that inside the session is fine
  // (no GPU dependency) but pre-fetching keeps the hot path purely
  // synchronous-style.
  await getPipeline(device, model, "embedding");
  await getPipeline(device, model, "rmsnorm");
  await getPipeline(device, model, "linear");
  await getPipeline(device, model, "rope");
  await getPipeline(device, model, "scalar_mul");
  await getPipeline(device, model, "add");
  await getPipeline(device, model, "mul");
  await getPipeline(device, model, "relu2");
  await getPipeline(device, model, "sigmoid");
  await getPipeline(device, model, "softcap");
  await getPipeline(device, model, "softmax");
  await getPipeline(device, model, "ve_apply");
  await getPipeline(device, model, "smear_apply");
  await getPipeline(device, model, "sdpa_scores");
  await getPipeline(device, model, "sdpa_output");

  const sess = beginSession(device, model);

  // 1. Embedding + norm (post-norm, pre-smear)
  let x = await embeddingT(device, model, "transformer.wte.weight", tokenId);
  x.shape = [1, n_embd];
  const xNormed = await rmsnormT(device, model, x); releaseTensor(model, x);
  x = xNormed;

  // 1b. Smear (decode T=1).
  if (cache) {
    const xPreSmear = cache.prevEmbeddingT;
    cache.prevEmbeddingT = copyTensor(device, model, x);

    if (xPreSmear !== null) {
      const gateRaw    = await linearT  (device, model, x, "smear_gate.weight");
      const gateSig    = await sigmoidT (device, model, gateRaw); releaseTensor(model, gateRaw);
      const gateScaled = await scalarMulT(device, model, gateSig, model.smearLambda); releaseTensor(model, gateSig);
      const pipeline = await getPipeline(device, model, "smear_apply");
      const u = _writeUniforms(device, model, [n_embd]);
      _dispatch(device, model, pipeline, [gateScaled.buffer, xPreSmear.buffer, x.buffer], u, [Math.ceil(n_embd / 64)]);
      _releaseUniform(model, u);
      releaseTensor(model, gateScaled);
      releaseTensor(model, xPreSmear);
    }
  }

  const x0 = copyTensor(device, model, x);

  // 2. Transformer blocks
  const backoutIdx = Math.floor(n_layer / 2);
  let xBackout = null;
  for (let i = 0; i < n_layer; i++) {
    let ve = null;
    const veName = `value_embeds.${i}.weight`;
    if (model.tensors.has(veName)) {
      ve = await embeddingT(device, model, veName, tokenId);
      ve.shape = [1, ve.byteLength / 4];
    }
    const next = await transformerBlockT(device, model, i, x, x0, T0, ve, cache);
    if (ve) releaseTensor(model, ve);
    releaseTensor(model, x);
    x = next;
    if (i === backoutIdx) xBackout = copyTensor(device, model, x);
  }
  releaseTensor(model, x0);

  if (cache) cache.seqlens += 1;

  // 3. Backout subtract
  if (xBackout !== null) {
    const negScaled = await scalarMulT(device, model, xBackout, -model.backoutLambda);
    releaseTensor(model, xBackout);
    const newX = await addT(device, model, x, negScaled); releaseTensor(model, negScaled);
    releaseTensor(model, x); x = newX;
  }

  // 4. Final norm + lm_head + softcap
  const xN = await rmsnormT(device, model, x); releaseTensor(model, x);
  const logitsPadded = await linearT(device, model, xN, "lm_head.weight"); releaseTensor(model, xN);
  const logitsCapped = await softcapT(device, model, logitsPadded, 15.0); releaseTensor(model, logitsPadded);

  // Single submit for the whole forward.
  await sess.submit();
  return logitsCapped;
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

  // Prefill: run forwardT for each prompt token. We only care about the
  // last one's logits (used to sample token 0 of the generation).
  let lastLogitsT = null;
  for (const tok of promptTokens) {
    if (lastLogitsT) releaseTensor(model, lastLogitsT);
    lastLogitsT = await forwardT(device, model, tok, cache);
  }

  // Generate: argmaxT each step on GPU; only the chosen token id (1 int)
  // ever crosses CPU/GPU. forwardT is the only thing that handles the
  // 32K-vocab logits array — it never leaves the GPU.
  const out = promptTokens.slice();
  for (let i = 0; i < maxNew; i++) {
    const argT = await argmaxT(device, model, lastLogitsT);
    const next = (await downloadU32(device, argT))[0];
    releaseTensor(model, argT);
    releaseTensor(model, lastLogitsT);
    out.push(next);
    if (i < maxNew - 1) {
      lastLogitsT = await forwardT(device, model, next, cache);
    } else {
      lastLogitsT = null;
    }
  }
  if (lastLogitsT) releaseTensor(model, lastLogitsT);

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

// transformerBlockT — Tensor in/out. Mirrors PyTorch GPT.forward inner loop:
//   x = resid_lambdas[i]*x + x0_lambdas[i]*x0
//   x = x + attn(norm(x))
//   x = x + mlp(norm(x))
export async function transformerBlockT(device, model, layerIdx, x, x0, T0 = 0, ve = null, cache = null) {
  await ensureScalars(device, model);
  const residL = model.residLambdas[layerIdx];
  const x0L    = model.x0Lambdas[layerIdx];

  // 1. Per-layer residual mix
  const resid = await scalarMulT(device, model, x,  residL);
  const x0m   = await scalarMulT(device, model, x0, x0L);
  let   cur   = await addT       (device, model, resid, x0m);
  releaseTensor(model, resid); releaseTensor(model, x0m);

  // 2. Attention + residual
  const xn = await rmsnormT(device, model, cur);
  const ao = await attentionBlockT(device, model, layerIdx, xn, T0, ve, cache);
  releaseTensor(model, xn);
  const cur2 = await addT(device, model, cur, ao);
  releaseTensor(model, cur); releaseTensor(model, ao);
  cur = cur2;

  // 3. MLP + residual
  const xn2 = await rmsnormT(device, model, cur);
  const mo  = await mlpBlockT(device, model, layerIdx, xn2);
  releaseTensor(model, xn2);
  const cur3 = await addT(device, model, cur, mo);
  releaseTensor(model, cur); releaseTensor(model, mo);
  return cur3;
}

export async function transformerBlock(device, model, layerIdx, x, x0, T0 = 0, ve = null, cache = null) {
  const n_embd = model.config.n_embd;
  const M = x.length / n_embd;
  const xT  = uploadTensor(device, model, x,  [M, n_embd], "tb_x");
  const x0T = uploadTensor(device, model, x0, [M, n_embd], "tb_x0");
  const veT = ve ? uploadTensor(device, model, ve, [M, model.config.n_kv_head * (n_embd / model.config.n_head)], "tb_ve") : null;
  const yT  = await transformerBlockT(device, model, layerIdx, xT, x0T, T0, veT, cache);
  const out = await downloadTensor(device, yT);
  releaseTensor(model, xT); releaseTensor(model, x0T);
  if (veT) releaseTensor(model, veT);
  releaseTensor(model, yT);
  return out;
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
// ---------------------------------------------------------------------
// Attention block (Tensor in/out). Single-token (M=1) inference.
// Mirrors PyTorch GPT.attention: c_q/c_k/c_v projections → ve gate
// (optional) → RoPE → per-head QK rmsnorm → scale → SDPA → c_proj.
//
// For M=1 with no cache: SDPA trivialises (one query, one key) so the
// output equals V. With cache: append (K, V) to per-layer cache buffers
// then run sdpa_scores → softmax → sdpa_output over the full T_k.
// ---------------------------------------------------------------------
export async function attentionBlockT(device, model, layerIdx, x, T0 = 0, ve = null, cache = null) {
  const prefix = `transformer.h.${layerIdx}.attn`;
  const cfg = model.config;
  const n_head   = cfg.n_head;
  const n_kv     = cfg.n_kv_head;
  const head_dim = cfg.n_embd / n_head;

  // 1. Q, K, V projections
  const q = await linearT(device, model, x, `${prefix}.c_q.weight`);
  const k = await linearT(device, model, x, `${prefix}.c_k.weight`);
  let   v = await linearT(device, model, x, `${prefix}.c_v.weight`);

  // 2. Value residual: v += 3 * sigmoid(ve_gate(x[..,:12])) * ve   (only if ve provided)
  //    ve_gate weight shape (n_kv, 12) — linearT reads only first 12 elements of x.
  if (ve !== null) {
    const gateRaw    = await linearT  (device, model, x, `${prefix}.ve_gate.weight`);
    const gateSig    = await sigmoidT (device, model, gateRaw);    releaseTensor(model, gateRaw);
    const gateTriple = await scalarMulT(device, model, gateSig, 3.0); releaseTensor(model, gateSig);
    await veApplyT(device, model, gateTriple, ve, v, head_dim);   // v += gateTriple[h] * ve[i]
    releaseTensor(model, gateTriple);
  }

  // 3. RoPE on Q, K — pass head_dim explicitly so rotation is per-head.
  const qR = await ropeApplyT(device, model, q, T0, head_dim); releaseTensor(model, q);
  const kR = await ropeApplyT(device, model, k, T0, head_dim); releaseTensor(model, k);

  // 4. QK norm: per-head rmsnorm. qR/kR are flat (1, n_head*head_dim)
  //    but we want each `head_dim`-long slice normalised independently,
  //    so pass nOverride=head_dim and rowsOverride=n_head.
  const qN = await rmsnormT(device, model, qR, 1e-5, n_head, head_dim); releaseTensor(model, qR);
  const kN = await rmsnormT(device, model, kR, 1e-5, n_kv,   head_dim); releaseTensor(model, kR);

  // 5. Scale by 1.2 (suomichat-specific QK scale, atop 1/sqrt(hd) inside SDPA)
  const qS = await scalarMulT(device, model, qN, 1.2); releaseTensor(model, qN);
  const kS = await scalarMulT(device, model, kN, 1.2); releaseTensor(model, kN);

  // 6. SDPA
  let attnT;
  if (cache === null) {
    attnT = v; v = null;   // T=1, no cache: trivial — output = v
  } else {
    const kvDim = n_kv * head_dim;
    const writeOffset = cache.seqlens * kvDim * 4;
    _copyBuffer(device, model, kS.buffer, 0, cache.kBuffers[layerIdx], writeOffset, kvDim * 4);
    _copyBuffer(device, model, v.buffer,  0, cache.vBuffers[layerIdx], writeOffset, kvDim * 4);
    const Tk = cache.seqlens + 1;

    // sdpa_scores: (n_head, Tk)
    const scoresT = allocTensor(device, model, [n_head, Tk], "sdpa_scores");
    {
      const pipeline = await getPipeline(device, model, "sdpa_scores");
      const u = _writeUniforms(device, model,
        [n_head, n_kv, head_dim, Tk],
        [[16, 1.0 / Math.sqrt(head_dim)]]);
      _dispatch(device, model, pipeline,
        [qS.buffer, cache.kBuffers[layerIdx], scoresT.buffer],
        u, [Math.ceil(n_head / 8), Math.ceil(Tk / 8)]);
      _releaseUniform(model, u);
    }
    // softmax rows
    const attnW = await softmaxT(device, model, scoresT, n_head, Tk);
    releaseTensor(model, scoresT);
    // sdpa_output: (n_head, head_dim)
    attnT = allocTensor(device, model, [1, n_head * head_dim], "sdpa_out");
    {
      const pipeline = await getPipeline(device, model, "sdpa_output");
      const u = _writeUniforms(device, model, [n_head, n_kv, head_dim, Tk]);
      _dispatch(device, model, pipeline,
        [attnW.buffer, cache.vBuffers[layerIdx], attnT.buffer],
        u, [Math.ceil(n_head / 8), Math.ceil(head_dim / 8)]);
      _releaseUniform(model, u);
    }
    releaseTensor(model, attnW);
    releaseTensor(model, v);
  }
  releaseTensor(model, kS);

  // 7. c_proj
  const out = await linearT(device, model, attnT, `${prefix}.c_proj.weight`);
  releaseTensor(model, attnT);
  return out;
}

export async function attentionBlock(device, model, layerIdx, x, T0 = 0, ve = null, cache = null) {
  const n_embd = model.config.n_embd;
  const M = x.length / n_embd;
  const xT = uploadTensor(device, model, x, [M, n_embd], "attn_x");
  const veT = ve ? uploadTensor(device, model, ve, [M, model.config.n_kv_head * (n_embd / model.config.n_head)], "attn_ve") : null;
  const yT = await attentionBlockT(device, model, layerIdx, xT, T0, veT, cache);
  const out = await downloadTensor(device, yT);
  releaseTensor(model, xT);
  if (veT) releaseTensor(model, veT);
  releaseTensor(model, yT);
  return out;
}

// ---------------------------------------------------------------------
// MLP block (Tensor in/out): y = c_proj(relu²(c_fc(x)))
// ---------------------------------------------------------------------
export async function mlpBlockT(device, model, layerIdx, x) {
  const prefix = `transformer.h.${layerIdx}.mlp`;
  const h = await linearT(device, model, x, `${prefix}.c_fc.weight`);
  const a = await relu2T(device, model, h);
  releaseTensor(model, h);
  const y = await linearT(device, model, a, `${prefix}.c_proj.weight`);
  releaseTensor(model, a);
  return y;
}

export async function mlpBlock(device, model, layerIdx, x) {
  const n_embd = model.config.n_embd;
  const M = x.length / n_embd;
  const xT = uploadTensor(device, model, x, [M, n_embd], "mlp_x");
  const yT = await mlpBlockT(device, model, layerIdx, xT);
  const out = await downloadTensor(device, yT);
  releaseTensor(model, xT); releaseTensor(model, yT);
  return out;
}

// ---------------------------------------------------------------------
// VE-apply: v[i] += gate[h] * ve[i] in place. Runs the ve_apply shader.
// ---------------------------------------------------------------------
async function veApplyT(device, model, gate, ve, v, head_dim) {
  const numel = v.byteLength / 4;
  const pipeline = await getPipeline(device, model, "ve_apply");
  const u = _writeUniforms(device, model, [numel, head_dim]);
  _dispatch(device, model, pipeline, [gate.buffer, ve.buffer, v.buffer], u, [Math.ceil(numel / 64)]);
  _releaseUniform(model, u);
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
