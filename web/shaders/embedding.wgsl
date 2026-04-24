// Embedding lookup shader.
//
// Inputs (storage buffers):
//   wte:    f32 array of shape (vocab_size * n_embd), row-major.
//   out:    f32 array of length n_embd. Filled in by this shader.
//
// Uniforms (16-byte aligned):
//   token_id: u32     row to read
//   n_embd:   u32     embedding dimension (= cols of wte)
//   vocab:    u32     vocab size (= rows of wte)  — currently unused, kept for sanity
//   _pad:     u32
//
// Dispatch: ceil(n_embd / 64) workgroups of 64 threads each.
//   Each thread handles one element of the output row.

@group(0) @binding(0) var<storage, read>      wte:    array<f32>;
@group(0) @binding(1) var<storage, read_write> out:    array<f32>;
@group(1) @binding(0) var<uniform>             params: Uniforms;

struct Uniforms {
    token_id: u32,
    n_embd:   u32,
    vocab:    u32,
    _pad:     u32,
};

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_embd) {
        return;
    }
    // Row-major: element at (token_id, i) is at index token_id * n_embd + i.
    let src_idx = params.token_id * params.n_embd + i;
    out[i] = wte[src_idx];
}
