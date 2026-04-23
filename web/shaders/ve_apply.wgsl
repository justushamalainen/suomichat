// Value-residual application: v[i] += gate[h] * ve[i]
//
// Used inside attention. `v` and `ve` are flattened (M*n_kv*head_dim)
// arrays. `gate` is the per-head scalar (length n_kv when M=1, which is
// the inference case we support). For each element index i, the head
// it belongs to is (i / head_dim).
//
// Shader runs in-place on `v` (read_write). One thread per element.
// Inference assumes M=1 — multi-token batched inference would also need
// an n_kv parameter to compute h modulo n_kv.

@group(0) @binding(0) var<storage, read>       gate: array<f32>;
@group(0) @binding(1) var<storage, read>       ve:   array<f32>;
@group(0) @binding(2) var<storage, read_write> v:    array<f32>;
@group(0) @binding(3) var<uniform>             p:    Params;

struct Params {
    numel:    u32,
    head_dim: u32,
    _p0:      u32,
    _p1:      u32,
};

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.numel) { return; }
    let h = i / p.head_dim;
    v[i] = v[i] + gate[h] * ve[i];
}
