// Smear apply: x[i] = x[i] + gate_scalar[0] * prev[i]
//
// gate_scalar is a 1-element f32 buffer (the smear lambda * sigmoid(gate)
// computed earlier). Avoids downloading that scalar to CPU mid-forward
// and re-uploading a broadcast Tensor.
//
// Used inside forwardT for the smear gate on decode steps 2+.

@group(0) @binding(0) var<storage, read>       gate_scalar: array<f32>;
@group(0) @binding(1) var<storage, read>       prev:        array<f32>;
@group(0) @binding(2) var<storage, read_write> x:           array<f32>;
@group(1) @binding(0) var<uniform>             p:           Params;

struct Params {
    n:   u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let g = gate_scalar[0];
    x[i] = x[i] + g * prev[i];
}
