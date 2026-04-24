// Fused MLP block — c_fc + relu² + c_proj in ONE dispatch.
//
// Equivalent PyTorch:
//   h = x @ c_fc.T           # (1, hidden)
//   h = F.relu(h).square()
//   y = h @ c_proj.T         # (1, n_embd)
//
// Single workgroup of 256 threads. Phase 1 each thread strides over
// `hidden`, computing relu²(c_fc · x) into workgroup-shared `h`.
// Phase 2 (after barrier) each thread strides over `n_embd`, computing
// c_proj · h.
//
// HIDDEN_MAX is hardcoded to 1536 (d6: 4 * n_embd = 4 * 384). For d12
// (n_embd 768, hidden 3072) raise it to 3072 — but workgroup memory is
// a shader-compile constant, so multi-model support means recompiling
// the shader per model size.

@group(0) @binding(0) var<storage, read>       x:        array<f32>;
@group(0) @binding(1) var<storage, read>       w_c_fc:   array<f32>;
@group(0) @binding(2) var<storage, read>       w_c_proj: array<f32>;
@group(0) @binding(3) var<storage, read_write> y:        array<f32>;
@group(0) @binding(4) var<uniform>             p:        Params;

struct Params {
    n_embd: u32,
    hidden: u32,
    _p0:    u32,
    _p1:    u32,
};

const WG:         u32 = 256u;
const HIDDEN_MAX: u32 = 1536u;
var<workgroup> h: array<f32, 1536>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;

    // Phase 1: h[i] = relu(sum_d x[d] * c_fc[i, d])^2
    var i: u32 = tid;
    loop {
        if (i >= p.hidden) { break; }
        var s: f32 = 0.0;
        var d: u32 = 0u;
        loop {
            if (d >= p.n_embd) { break; }
            s = s + x[d] * w_c_fc[i * p.n_embd + d];
            d = d + 1u;
        }
        let r = max(s, 0.0);
        h[i] = r * r;
        i = i + WG;
    }
    workgroupBarrier();

    // Phase 2: y[j] = sum_k h[k] * c_proj[j, k]
    var j: u32 = tid;
    loop {
        if (j >= p.n_embd) { break; }
        var s: f32 = 0.0;
        var k: u32 = 0u;
        loop {
            if (k >= p.hidden) { break; }
            s = s + h[k] * w_c_proj[j * p.hidden + k];
            k = k + 1u;
        }
        y[j] = s;
        j = j + WG;
    }
}
