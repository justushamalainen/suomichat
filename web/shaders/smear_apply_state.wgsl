// Fused smear apply for the record/replay decode path.
// In ONE dispatch:
//   1. prev_out[i] = x[i]                    (snapshot current pre-smear
//                                             — becomes "prev" for the
//                                             NEXT decode step)
//   2. x[i]        = x[i] + gate[0] * prev_in[i]
//                                            (apply smear from previous
//                                             step's snapshot)
//
// Because each thread handles ONE element and reads x[i] before writing
// it, the in-place update is safe within a thread; no barriers needed.
//
// JS schedules two bind groups, one per parity:
//   even step: prev_in = prev[0], prev_out = prev[1]
//   odd  step: prev_in = prev[1], prev_out = prev[0]
//
// On the very first decode step prev_in must already hold something
// sensible (the prefill writes the last prompt token's x_norm into
// prev[0] so step 0 [even parity] smears against it).

@group(0) @binding(0) var<storage, read>       gate:     array<f32>;   // length 1
@group(0) @binding(1) var<storage, read>       prev_in:  array<f32>;
@group(0) @binding(2) var<storage, read_write> prev_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> x:        array<f32>;
@group(0) @binding(4) var<uniform>             p:        Params;

struct Params {
    n:   u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let v = x[i];
    prev_out[i] = v;
    x[i]        = v + gate[0] * prev_in[i];
}
