// Attention weighted sum (T_q=1): out[h, d] = sum_tk attn[h, tk] * V_cache[tk, h_kv, d]
//
// attn:     (nH * Tk)           softmaxed attention weights
// v_cache:  (max_T * nKV * hd)  GPU-resident V cache, layout (t, h_kv, d)
// out:      (nH * hd)
//
// GQA mapping: h_kv = h * nKV / nH.
//
// Dispatch: (ceil(nH/8), ceil(hd/8)). One thread per (h, d).

@group(0) @binding(0) var<storage, read>       attn:     array<f32>;
@group(0) @binding(1) var<storage, read>       v_cache:  array<f32>;
@group(0) @binding(2) var<storage, read_write> out:      array<f32>;
@group(0) @binding(3) var<uniform>             params:   Params;

struct Params {
    nH:  u32,
    nKV: u32,
    hd:  u32,
    Tk:  u32,
};

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    let d = gid.y;
    if (h >= params.nH || d >= params.hd) { return; }

    let h_kv = (h * params.nKV) / params.nH;

    var s: f32 = 0.0;
    var tk: u32 = 0u;
    loop {
        if (tk >= params.Tk) { break; }
        let v_off = tk * params.nKV * params.hd + h_kv * params.hd + d;
        s = s + attn[h * params.Tk + tk] * v_cache[v_off];
        tk = tk + 1u;
    }
    out[h * params.hd + d] = s;
}
