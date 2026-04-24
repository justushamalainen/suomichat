// Multi-token smear apply (PyTorch prefill semantics, gpt.py:438):
//   out[0]   = x[0]                                      (position 0 unchanged)
//   out[t]   = x[t] + gates[t-1] * x[t-1]   for t >= 1
// where gates is length T (gates[0] is unused — the gate for position 0
// has no previous embedding to combine with) but we write all T to keep
// the layout simple; gates[t] = smear_lambda * sigmoid(smear_gate(x[t,:24])).

@group(0) @binding(0) var<storage, read>       gates: array<f32>;
@group(0) @binding(1) var<storage, read>       x:     array<f32>;
@group(0) @binding(2) var<storage, read_write> out:   array<f32>;
@group(1) @binding(0) var<uniform>             p:     Params;

struct Params {
    T:       u32,
    n_embd:  u32,
    _p0:     u32,
    _p1:     u32,
};

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = p.T * p.n_embd;
    if (i >= total) { return; }
    let t = i / p.n_embd;
    let d = i % p.n_embd;
    if (t == 0u) {
        out[i] = x[i];
    } else {
        out[i] = x[i] + gates[t] * x[(t - 1u) * p.n_embd + d];
    }
}
