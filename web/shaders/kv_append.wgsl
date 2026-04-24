// KV cache append using state.T0 as the write position.
//
// Equivalent to:
//   copyBufferToBuffer(src, 0, cache, T0 * kvDim * 4, kvDim * 4)
// but the offset comes from a uniform, so the dispatch can be baked
// into a recorded command buffer (whereas copyBufferToBuffer offsets
// are fixed at encoding time).
//
// src:   (kvDim,)            — newly-computed K or V for this token
// cache: (max_T, kvDim)      — per-layer K_cache or V_cache
// state.T0: row to write     — cache.seqlens, changes per token

@group(0) @binding(0) var<storage, read>       src:   array<f32>;
@group(0) @binding(1) var<storage, read_write> cache: array<f32>;
@group(0) @binding(2) var<uniform>             p:     Params;

@group(1) @binding(0) var<uniform>             state: DecodeState;

struct Params     { kvDim: u32, _p0: u32, _p1: u32, _p2: u32 };
struct DecodeState { token_id: u32, T0: u32, _p0: u32, _p1: u32 };

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.kvDim) { return; }
    cache[state.T0 * p.kvDim + i] = src[i];
}
