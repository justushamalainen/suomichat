// Naive matmul: C (M×N) = A (M×K) @ B (K×N), all row-major f32.
//
// Dispatch: ceil(M/8) × ceil(N/8) workgroups of (8, 8) threads.
// Each thread computes one output element C[m, n] by looping over K.
// This is the dumbest correct version — no shared memory, no tiling,
// no vec4 loads. The goal is readability.
//
// Uniforms:
//   M, K, N: dimensions
//   _pad:    16-byte alignment

@group(0) @binding(0) var<storage, read>       a:      array<f32>;
@group(0) @binding(1) var<storage, read>       b:      array<f32>;
@group(0) @binding(2) var<storage, read_write> c:      array<f32>;
@group(1) @binding(0) var<uniform>             params: Params;

struct Params {
    M:   u32,
    K:   u32,
    N:   u32,
    _p:  u32,
};

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = gid.x;
    let n = gid.y;
    if (m >= params.M || n >= params.N) {
        return;
    }
    var s: f32 = 0.0;
    var k: u32 = 0u;
    loop {
        if (k >= params.K) { break; }
        // A[m, k] at row-major index m*K + k
        // B[k, n] at row-major index k*N + n
        s = s + a[m * params.K + k] * b[k * params.N + n];
        k = k + 1u;
    }
    c[m * params.N + n] = s;
}
