// Attention scores with causal mask, T queries × Tk keys.
//
// scores[t, h, tk] = (q[t, h] · K_cache[tk, h_kv]) / sqrt(hd)
// Causal: query at absolute position (T0 + t) can attend to keys at
// absolute positions 0..(T0 + t). Out-of-range slots get -inf so softmax
// weighs them to zero.
//
// T=1 (decode) is a special case: T0 = cache.seqlens, Tk = T0 + 1, t = 0.
// All Tk slots are valid (tk <= T0+0 = T0 = Tk-1). No mask kicks in.
// T>1 (prefill) requires the mask.
//
// q:        (T * nH * hd)
// k_cache:  (max_T * nKV * hd)   layout (t, h_kv, d)
// scores:   (T * nH * Tk)
//
// Dispatch: ceil(T*nH / 8) × ceil(Tk / 8). One thread per (t*nH+h, tk).

@group(0) @binding(0) var<storage, read>       q:        array<f32>;
@group(0) @binding(1) var<storage, read>       k_cache:  array<f32>;
@group(0) @binding(2) var<storage, read_write> scores:   array<f32>;
@group(0) @binding(3) var<uniform>             params:   Params;

struct Params {
    nH:    u32,
    nKV:   u32,
    hd:    u32,
    Tk:    u32,
    T:     u32,
    T0:    u32,
    scale: f32,
    _p0:   u32,
};

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qIdx = gid.x;       // 0..T*nH
    let tk   = gid.y;       // 0..Tk
    if (qIdx >= params.T * params.nH || tk >= params.Tk) { return; }

    let t    = qIdx / params.nH;
    let h    = qIdx % params.nH;
    let h_kv = (h * params.nKV) / params.nH;

    var s: f32;
    let absPos = params.T0 + t;
    if (tk > absPos) {
        s = -3.4e38;     // -inf
    } else {
        let q_off = (t * params.nH + h) * params.hd;
        let k_off = (tk * params.nKV + h_kv) * params.hd;
        var sum: f32 = 0.0;
        var d: u32 = 0u;
        loop {
            if (d >= params.hd) { break; }
            sum = sum + q[q_off + d] * k_cache[k_off + d];
            d = d + 1u;
        }
        s = sum * params.scale;
    }
    scores[(t * params.nH + h) * params.Tk + tk] = s;
}
