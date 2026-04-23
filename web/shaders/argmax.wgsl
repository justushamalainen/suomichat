// Argmax over a 1D tensor.
//
// Input  x:   f32 array of length n
// Output out: u32 array of length 1 (the index of the max)
// Uniform: n
//
// Single workgroup of 256 threads. Each thread scans its strided slice
// of x and tracks (best_value, best_index). Then a tree reduction picks
// the global winner. Sufficient for vocab=32K (each thread does ~128
// comparisons).

@group(0) @binding(0) var<storage, read>       x:   array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform>             p:   Params;

struct Params {
    n:   u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

const WG: u32 = 256u;
var<workgroup> part_v: array<f32, 256>;
var<workgroup> part_i: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;

    // Per-thread strided scan
    var v: f32 = -3.4e38;
    var i: u32 = 0u;
    var k: u32 = tid;
    loop {
        if (k >= p.n) { break; }
        let xv = x[k];
        if (xv > v) { v = xv; i = k; }
        k = k + WG;
    }
    part_v[tid] = v;
    part_i[tid] = i;
    workgroupBarrier();

    // Tree reduction: keep the max-valued thread's index.
    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            if (part_v[tid + stride] > part_v[tid]) {
                part_v[tid] = part_v[tid + stride];
                part_i[tid] = part_i[tid + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (tid == 0u) { out[0] = part_i[0]; }
}
