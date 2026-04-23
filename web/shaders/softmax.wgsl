// Row-wise softmax with workgroup-shared reductions.
//
// Input  x: f32 array, shape (rows, n)   row-major
// Output y: same shape
// One workgroup (64 threads) per row. Two reductions:
//   1. max across the row (for numerical stability)
//   2. sum of exp(x - max)
// Then each thread writes y[i] = exp(x[i] - max) / sum for its strided slice.

@group(0) @binding(0) var<storage, read>       x:      array<f32>;
@group(0) @binding(1) var<storage, read_write> y:      array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;

struct Params {
    rows: u32,
    n:    u32,
    _p0:  u32,
    _p1:  u32,
};

const WG: u32 = 64u;
var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid:  vec3<u32>,
        @builtin(workgroup_id)         wid: vec3<u32>) {
    let row = wid.x;
    if (row >= params.rows) { return; }
    let tid = lid.x;
    let base = row * params.n;

    // 1. Find max via tree reduction
    var m: f32 = -3.4e38;   // -inf for f32
    var i: u32 = tid;
    loop {
        if (i >= params.n) { break; }
        m = max(m, x[base + i]);
        i = i + WG;
    }
    partial[tid] = m;
    workgroupBarrier();
    var s: u32 = WG / 2u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { partial[tid] = max(partial[tid], partial[tid + s]); }
        workgroupBarrier();
        s = s / 2u;
    }
    let row_max = partial[0];
    workgroupBarrier();

    // 2. Sum exp(x - max)
    var sum: f32 = 0.0;
    i = tid;
    loop {
        if (i >= params.n) { break; }
        sum = sum + exp(x[base + i] - row_max);
        i = i + WG;
    }
    partial[tid] = sum;
    workgroupBarrier();
    s = WG / 2u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { partial[tid] = partial[tid] + partial[tid + s]; }
        workgroupBarrier();
        s = s / 2u;
    }
    let row_sum = partial[0];
    workgroupBarrier();

    // 3. Write normalised output
    i = tid;
    loop {
        if (i >= params.n) { break; }
        y[base + i] = exp(x[base + i] - row_max) / row_sum;
        i = i + WG;
    }
}
