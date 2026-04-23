// Linear layer without bias: y = x @ weight.T
//
// Matches torch.nn.Linear(in_features, out_features, bias=False) where
// weight has PyTorch's native shape (out_features, in_features).
//
// Shapes:
//   x:      (M, K)       row-major
//   weight: (N, K)       row-major (PyTorch convention)
//   y:      (M, N)
//
// We avoid needing a separate matmul shader for weights by indexing
// weight[n, k] at offset (n * K + k) — i.e. reading "along" the
// out-feature dimension, which is equivalent to (x @ weight.T).

@group(0) @binding(0) var<storage, read>       x:      array<f32>;
@group(0) @binding(1) var<storage, read>       w:      array<f32>;
@group(0) @binding(2) var<storage, read_write> y:      array<f32>;
@group(0) @binding(3) var<uniform>             params: Params;

struct Params {
    M:   u32,   // rows of x
    K:   u32,   // in_features
    N:   u32,   // out_features
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
        s = s + x[m * params.K + k] * w[n * params.K + k];
        k = k + 1u;
    }
    y[m * params.N + n] = s;
}
