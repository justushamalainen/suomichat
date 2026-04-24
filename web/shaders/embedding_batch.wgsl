// Multi-token embedding lookup. Reads T rows of `wte` (one per token id)
// and writes them into a flat (T, n_embd) output buffer.
//
// wte:       (vocab, n_embd)
// token_ids: (T,)  u32
// out:       (T, n_embd)
//
// Dispatch: ceil(T * n_embd / 64) threads. Each thread copies one f32.

@group(0) @binding(0) var<storage, read>       wte:        array<f32>;
@group(0) @binding(1) var<storage, read>       token_ids:  array<u32>;
@group(0) @binding(2) var<storage, read_write> out:        array<f32>;
@group(1) @binding(0) var<uniform>             p:          Params;

struct Params {
    T:      u32,
    n_embd: u32,
    vocab:  u32,
    _p0:    u32,
};

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = p.T * p.n_embd;
    if (i >= total) { return; }
    let t = i / p.n_embd;
    let d = i % p.n_embd;
    let id = token_ids[t];
    out[i] = wte[id * p.n_embd + d];
}
