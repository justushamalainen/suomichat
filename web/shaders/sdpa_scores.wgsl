// Attention scores (T_q=1): scores[h, tk] = (q[h] · K_cache[tk, h_kv]) / sqrt(hd)
//
// q:        (nH * hd)            single query per head
// k_cache:  (max_T * nKV * hd)   GPU-resident K cache, layout (t, h_kv, d)
// scores:   (nH * Tk)            output; only first Tk positions written
//
// GQA mapping: h_kv = h * nKV / nH. For d12 nH==nKV so it's identity.
//
// Dispatch: (ceil(nH/8), ceil(Tk/8)). One thread per (h, tk).

@group(0) @binding(0) var<storage, read>       q:        array<f32>;
@group(0) @binding(1) var<storage, read>       k_cache:  array<f32>;
@group(0) @binding(2) var<storage, read_write> scores:   array<f32>;
@group(0) @binding(3) var<uniform>             params:   Params;

struct Params {
    nH:    u32,
    nKV:   u32,
    hd:    u32,
    Tk:    u32,
    scale: f32,    // 1 / sqrt(hd)
    _p0:   u32,
    _p1:   u32,
    _p2:   u32,
};

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h  = gid.x;
    let tk = gid.y;
    if (h >= params.nH || tk >= params.Tk) { return; }

    let h_kv = (h * params.nKV) / params.nH;
    let q_off = h * params.hd;
    let k_off = tk * params.nKV * params.hd + h_kv * params.hd;

    var s: f32 = 0.0;
    var d: u32 = 0u;
    loop {
        if (d >= params.hd) { break; }
        s = s + q[q_off + d] * k_cache[k_off + d];
        d = d + 1u;
    }
    scores[h * params.Tk + tk] = s * params.scale;
}
