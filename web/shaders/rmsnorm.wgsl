// RMSNorm.
//
// Equivalent to PyTorch:
//   y = x / sqrt(mean(x^2) + eps)     (no learnable gain — suomichat uses
//                                       F.rms_norm without a weight tensor)
//
// We use one workgroup of 64 threads to handle one row of length n.
// The threads cooperatively compute sum-of-squares via a parallel
// reduction in shared memory, then everyone uses the same scale to
// normalize their strided slice of elements.
//
// Inputs:
//   in:  f32 array of length n     (one row)
//   out: f32 array of length n     (filled in)
// Uniforms:
//   n:   u32   length of the row
//   eps: f32   numerical-stability constant (typically 1e-5)
//
// Dispatch: 1 workgroup. (Multi-row batching = dispatch multiple
// workgroups + index by workgroup_id; we'll add that later.)

@group(0) @binding(0) var<storage, read>       in:     array<f32>;
@group(0) @binding(1) var<storage, read_write> out:    array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;

struct Params {
    n:    u32,
    eps:  f32,
    _p0:  u32,
    _p1:  u32,
};

const WG: u32 = 64u;
var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;

    // 1. Each thread sums squares of its strided slice.
    var s: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= params.n) { break; }
        let v = in[i];
        s = s + v * v;
        i = i + WG;
    }
    partial[tid] = s;
    workgroupBarrier();

    // 2. Tree reduction across the workgroup. Halve the active set each step.
    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            partial[tid] = partial[tid] + partial[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // 3. partial[0] now holds sum-of-squares. Compute scale.
    let mean_sq = partial[0] / f32(params.n);
    let scale = inverseSqrt(mean_sq + params.eps);

    // 4. Each thread writes its strided slice scaled.
    var j: u32 = tid;
    loop {
        if (j >= params.n) { break; }
        out[j] = in[j] * scale;
        j = j + WG;
    }
}
