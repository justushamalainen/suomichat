// RMSNorm. Multi-row: one workgroup per row, 64 threads per workgroup.
//
// Equivalent to PyTorch:
//   y[row] = x[row] / sqrt(mean(x[row]^2) + eps)
//
// Inputs:
//   in:  f32 array of length rows*n
//   out: f32 array of length rows*n
// Uniforms:
//   n:    u32   row length
//   eps:  f32   numerical-stability constant (typically 1e-5)
//
// Dispatch: `rows` workgroups. Each workgroup of 64 threads computes
// sum-of-squares for its row via a tree reduction in shared memory,
// then writes the normalized row.

@group(0) @binding(0) var<storage, read>       in:     array<f32>;
@group(0) @binding(1) var<storage, read_write> out:    array<f32>;
@group(1) @binding(0) var<uniform>             params: Params;

struct Params {
    n:    u32,
    eps:  f32,
    _p0:  u32,
    _p1:  u32,
};

const WG: u32 = 64u;
var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id)        wid: vec3<u32>) {
    let tid  = lid.x;
    let base = wid.x * params.n;

    // 1. Each thread sums squares of its strided slice of the row.
    var s: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if (i >= params.n) { break; }
        let v = in[base + i];
        s = s + v * v;
        i = i + WG;
    }
    partial[tid] = s;
    workgroupBarrier();

    // 2. Tree reduction across the workgroup.
    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            partial[tid] = partial[tid] + partial[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let mean_sq = partial[0] / f32(params.n);
    let scale   = inverseSqrt(mean_sq + params.eps);

    // 3. Each thread writes its strided slice scaled.
    var j: u32 = tid;
    loop {
        if (j >= params.n) { break; }
        out[base + j] = in[base + j] * scale;
        j = j + WG;
    }
}
