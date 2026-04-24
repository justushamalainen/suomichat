// Linear layer without bias: y = x @ weight.T
//
// Shapes:
//   x:      (M, Kstride)   row-major; we use only the FIRST K elements per row
//   weight: (N, K)         row-major (PyTorch convention)
//   y:      (M, N)
//
// `Kstride` defaults to K (set by JS) — backwards-compatible. When the
// caller wants to consume only a prefix of a wider row (e.g. ve_gate
// reads x[..., :12] from x of shape (T, n_embd)), pass Kstride=n_embd.

@group(0) @binding(0) var<storage, read>       x:      array<f32>;
@group(0) @binding(1) var<storage, read>       w:      array<f32>;
@group(0) @binding(2) var<storage, read_write> y:      array<f32>;
@group(1) @binding(0) var<uniform>             params: Params;

struct Params {
    M:        u32,
    K:        u32,
    N:        u32,
    Kstride:  u32,
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
        s = s + x[m * params.Kstride + k] * w[n * params.K + k];
        k = k + 1u;
    }
    y[m * params.N + n] = s;
}
