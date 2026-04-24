// Attention weighted-value sum, T queries × Tk keys.
//
// out[t, h, d] = sum_tk attn[t, h, tk] * V_cache[tk, h_kv, d]
//
// attn:     (T * nH * Tk)        post-softmax weights (causal-masked
//                                positions are exactly 0)
// v_cache:  (max_T * nKV * hd)
// out:      (T * nH * hd)
//
// Dispatch: ceil(T*nH / 8) × ceil(hd / 8). One thread per (t, h, d).

@group(0) @binding(0) var<storage, read>       attn:     array<f32>;
@group(0) @binding(1) var<storage, read>       v_cache:  array<f32>;
@group(0) @binding(2) var<storage, read_write> out:      array<f32>;
@group(0) @binding(3) var<uniform>             params:   Params;

struct Params {
    nH:  u32,
    nKV: u32,
    hd:  u32,
    Tk:  u32,
    T:   u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qIdx = gid.x;
    let d    = gid.y;
    if (qIdx >= params.T * params.nH || d >= params.hd) { return; }

    let t    = qIdx / params.nH;
    let h    = qIdx % params.nH;
    let h_kv = (h * params.nKV) / params.nH;

    var sum: f32 = 0.0;
    var tk: u32 = 0u;
    loop {
        if (tk >= params.Tk) { break; }
        let v_off = (tk * params.nKV + h_kv) * params.hd + d;
        sum = sum + attn[(t * params.nH + h) * params.Tk + tk] * v_cache[v_off];
        tk = tk + 1u;
    }
    out[(t * params.nH + h) * params.hd + d] = sum;
}
