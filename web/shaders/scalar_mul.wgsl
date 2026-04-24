// Scalar multiply: c[i] = alpha * a[i]
//
// Uniform moved to @group(1) for the ring-buffer dispatch path.

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read_write> c: array<f32>;

@group(1) @binding(0) var<uniform>             params: Params;

struct Params { n: u32, alpha: f32, _p0: u32, _p1: u32 };

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    c[i] = params.alpha * a[i];
}
