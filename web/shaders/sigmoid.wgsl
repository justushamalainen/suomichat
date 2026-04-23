// Sigmoid: c[i] = 1 / (1 + exp(-a[i]))

@group(0) @binding(0) var<storage, read>       a:      array<f32>;
@group(0) @binding(1) var<storage, read_write> c:      array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;

struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, };

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    c[i] = 1.0 / (1.0 + exp(-a[i]));
}
