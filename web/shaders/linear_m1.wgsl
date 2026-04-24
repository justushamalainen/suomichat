// Linear layer specialised for M=1. Simplest possible structure:
// one thread per output, vec4 inner loop.
//
// y = x @ weight.T     x (1, K), w (N, K), y (1, N)
//
// Each thread:  y[n] = Σ dot(vec4 x[i], vec4 w[n, i])  for i in 0..K/4
//
// Dispatch: ceil(N / 256) workgroups of 256 threads. K must be a
// multiple of 4 (caller falls back otherwise).
//
// Tradeoffs vs naive fp32 scalar matmul:
//   + 4× fewer memory transactions via vec4 loads
//   + 256 threads/WG concurrency (vs 8 active of 64 in @workgroup_size(8,8))
//   - Threads' B reads are still strided by K bytes (not coalesced within
//     a warp). An ORT-style K-split tiled kernel solves this but showed
//     2.8× regression at our shapes due to barrier + shared-mem overhead
//     — kept as shaders/linear_m1_ksplit.wgsl for reference.

@group(0) @binding(0) var<storage, read>       x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       w: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform>             params: Params;

struct Params {
    K:    u32,   // scalar K (must be multiple of 4)
    N:    u32,
    _p0:  u32,
    _p1:  u32,
};

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= params.N) { return; }
    let k_vec = params.K / 4u;
    var s4: vec4<f32> = vec4<f32>(0.0);
    var k: u32 = 0u;
    loop {
        if (k >= k_vec) { break; }
        let x4 = x[k];
        let w4 = w[n * k_vec + k];
        s4 = s4 + x4 * w4;
        k = k + 1u;
    }
    y[n] = s4.x + s4.y + s4.z + s4.w;
}
