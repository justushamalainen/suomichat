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
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
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
// 6. The first real operation: embedding lookup.
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
// 7. RMSNorm.
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
