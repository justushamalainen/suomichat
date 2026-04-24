// Rotary position embedding application.
//
// Input x is a flattened tensor of shape (N, D) where N = T*H
// (time × heads, flat) and D = head_dim. cos/sin are flat arrays of
// shape (rotary_seq_len, d) where d = D/2.
//
// Per-row position: row n has token index (n / H), so its position is
// T0 + (n / H). For T=1 (decode), H=N → all rows get position T0
// (backward compatible with the old single-position path).
// For T>1 (prefill), pass H = n_head/n_kv so each token's heads share
// a position, and consecutive tokens get consecutive positions.
//
// Rotation formulation (matches suomichat/gpt.py:apply_rotary_emb):
//   y1[i]   = x1[i] * cos[i] + x2[i] * sin[i]   for i in 0..d
//   y2[i]   = -x1[i] * sin[i] + x2[i] * cos[i]  (stored at offset d+i)
// where x1 = x[..., :d], x2 = x[..., d:].

@group(0) @binding(0) var<storage, read>       x:      array<f32>;
@group(0) @binding(1) var<storage, read>       cos_t:  array<f32>;
@group(0) @binding(2) var<storage, read>       sin_t:  array<f32>;
@group(0) @binding(3) var<storage, read_write> y:      array<f32>;
@group(1) @binding(0) var<uniform>             params: Params;

struct Params {
    N:  u32,   // total rows (T * H)
    D:  u32,   // head_dim
    d:  u32,   // D/2
    T0: u32,   // base position
    H:  u32,   // heads per token (so token = n / H, position = T0 + token)
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let total = params.N * params.D;
    if (flat >= total) { return; }

    let n = flat / params.D;
    let i = flat % params.D;
    let base = n * params.D;

    let pos = params.T0 + (n / params.H);
    let row = pos * params.d;

    if (i < params.d) {
        y[flat] = x[base + i] * cos_t[row + i]
                + x[base + i + params.d] * sin_t[row + i];
    } else {
        let ii = i - params.d;
        y[flat] = -x[base + ii] * sin_t[row + ii]
                + x[flat]       * cos_t[row + ii];
    }
}
