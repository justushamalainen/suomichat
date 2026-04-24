// Attention scores with causal mask, T queries × Tk keys.
//
// scores[t, h, tk] = (q[t, h] · K_cache[tk, h_kv]) / sqrt(hd)
// Causal: query at absolute position (T0 + t) attends to keys at
// absolute positions 0..(T0 + t). Out-of-range slots get -inf so
// softmax weighs them to zero.
//
// `Tk` is the DISPATCHED / scores-stride dimension — usually equal to
// `Tk_valid`, but can be larger for command-buffer replay (where the
// dispatch shape must be fixed across all decode steps). Slots in
// `[Tk_valid, Tk)` get -inf scores so they contribute nothing.

@group(0) @binding(0) var<storage, read>       q:        array<f32>;
@group(0) @binding(1) var<storage, read>       k_cache:  array<f32>;
@group(0) @binding(2) var<storage, read_write> scores:   array<f32>;
@group(1) @binding(0) var<uniform>             params:   Params;

struct Params {
    nH:        u32,
    nKV:       u32,
    hd:        u32,
    Tk:        u32,        // dispatched / scores stride
    T:         u32,
    T0:        u32,
    scale:     f32,
    Tk_valid:  u32,        // tk in [0, Tk_valid) may have non-(-inf) scores
};

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qIdx = gid.x;
    let tk   = gid.y;
    if (qIdx >= params.T * params.nH || tk >= params.Tk) { return; }

    let t    = qIdx / params.nH;
    let h    = qIdx % params.nH;
    let h_kv = (h * params.nKV) / params.nH;

    var s: f32;
    let absPos = params.T0 + t;
    if (tk >= params.Tk_valid || tk > absPos) {
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
