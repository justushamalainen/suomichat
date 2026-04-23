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
// 4. The first real operation: embedding lookup.
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
