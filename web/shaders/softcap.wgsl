// Logit softcap: y = cap * tanh(x / cap)
//
// suomichat uses softcap=15 at the final lm_head output. Smoothly
// saturates values to ±cap without a hard clip. This is element-wise,
// one thread per element, identical structure to sigmoid.

@group(0) @binding(0) var<storage, read>       x:      array<f32>;
@group(0) @binding(1) var<storage, read_write> y:      array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;

struct Params {
    n:   u32,
    cap: f32,
    _p0: u32,
    _p1: u32,
};

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    y[i] = params.cap * tanh(x[i] / params.cap);
}
