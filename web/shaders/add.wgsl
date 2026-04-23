// Element-wise add: c[i] = a[i] + b[i]

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             params: Params;

struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, };

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    c[i] = a[i] + b[i];
}
