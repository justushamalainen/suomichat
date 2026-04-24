// Multi-token VE-apply: v[t, h, d] += gate[t, h] * ve[t, h, d]
//
// gate: (T, n_kv)
// ve:   (T, n_kv, head_dim)  flat
// v:    (T, n_kv, head_dim)  flat (read_write — modified in place)
//
// One thread per element of v. T=1 case handled by the existing
// ve_apply.wgsl; this shader is for T>1 prefill.

@group(0) @binding(0) var<storage, read>       gate: array<f32>;
@group(0) @binding(1) var<storage, read>       ve:   array<f32>;
@group(0) @binding(2) var<storage, read_write> v:    array<f32>;
@group(1) @binding(0) var<uniform>             p:    Params;

struct Params {
    numel:    u32,    // T * n_kv * head_dim
    head_dim: u32,
    n_kv:     u32,
    _p0:      u32,
};

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.numel) { return; }
    let t = i / (p.n_kv * p.head_dim);
    let h = (i % (p.n_kv * p.head_dim)) / p.head_dim;
    v[i] = v[i] + gate[t * p.n_kv + h] * ve[i];
}
