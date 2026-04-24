// Embedding lookup that reads `token_id` from a SHARED decode_state
// uniform buffer (so the bind group can be baked into a recorded
// command buffer; only writeBuffer is needed between submits).
//
// @group(0): static-per-call params and tensors
// @group(1): the dynamic decode state (token_id, T0)
//
// Out shape: (n_embd,) — one row of wte indexed by state.token_id.

@group(0) @binding(0) var<storage, read>       wte: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             p:   Params;

@group(1) @binding(0) var<uniform>             state: DecodeState;

struct Params {
    n_embd: u32,
    vocab:  u32,
    _p0:    u32,
    _p1:    u32,
};

struct DecodeState {
    token_id: u32,
    T0:       u32,
    _p0:      u32,
    _p1:      u32,
};

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n_embd) { return; }
    out[i] = wte[state.token_id * p.n_embd + i];
}
