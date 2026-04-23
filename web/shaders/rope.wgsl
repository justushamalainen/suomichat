// Rotary position embedding application.
//
// Input x is a flattened tensor of shape (N, D) where N = B*T*H
// (flattened across batch, time, head) and D = head_dim.
// cos/sin are flat arrays of shape (rotary_seq_len, d) where d = D/2.
//
// For each (n, i) in (N, D):
//   t = T0 + (n / H)     — but for our T=1 path N=H and t=T0 for all n
//   For Phase 5 we simplify: pass T0 as uniform, assume all N share the
//   same position. Multi-position RoPE (T>1 prefill) comes later.
//
// Rotation formulation (matches suomichat/gpt.py:apply_rotary_emb):
//   y1[i]   = x1[i] * cos[i] + x2[i] * sin[i]   for i in 0..d
//   y2[i]   = -x1[i] * sin[i] + x2[i] * cos[i]  (stored at offset d+i)
// where x1 = x[..., :d], x2 = x[..., d:].

@group(0) @binding(0) var<storage, read>       x:      array<f32>;
@group(0) @binding(1) var<storage, read>       cos_t:  array<f32>;
@group(0) @binding(2) var<storage, read>       sin_t:  array<f32>;
@group(0) @binding(3) var<storage, read_write> y:      array<f32>;
@group(0) @binding(4) var<uniform>             params: Params;

struct Params {
    N:  u32,   // number of (b, t, h) positions
    D:  u32,   // head_dim
    d:  u32,   // D/2
    T0: u32,   // time-position offset into cos/sin
};

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let total = params.N * params.D;
    if (flat >= total) { return; }

    let n = flat / params.D;
    let i = flat % params.D;
    let base = n * params.D;

    // cos/sin rows are (rotary_seq_len, d). We read row T0.
    let row = params.T0 * params.d;

    if (i < params.d) {
        // First half gets y1 = x1 * cos + x2 * sin
        y[flat] = x[base + i] * cos_t[row + i]
                + x[base + i + params.d] * sin_t[row + i];
    } else {
        // Second half gets y2 = -x1 * sin + x2 * cos
        let ii = i - params.d;
        y[flat] = -x[base + ii] * sin_t[row + ii]
                + x[flat]       * cos_t[row + ii];
    }
}
