# Fused-kernels plan

Branch: `web/perf-fused`. Goal: collapse the ~200 dispatches per forward into
~10 by writing big WGSL shaders that do whole logical units in one dispatch.

## Why this is the right lever

PyTorch beats us 100× because each of its op dispatches costs ~5–50 µs vs
WebGPU's ~400 µs. We can't shrink WebGPU's per-dispatch cost — but we can
cut the dispatch count. CUDA-Graphs / torch.compile / flash-attn all do
the same thing: fuse small ops into one big kernel.

## Target architecture

| Phase | Fuses | Dispatches before → after | Expected savings |
|---|---|---|---|
| F1 | MLP: c_fc + relu² + c_proj | 3 → 1 | 6 layers × 2 dispatches = ~8 ms |
| F2 | Attention: Q/K/V + RoPE + QKnorm + SDPA + c_proj | 19 → 1 (decode) | 6 × 18 = ~70 ms |
| F3 | Full block: residual + norm + attn + add + norm + mlp + add | ~30 → 1 | 6 × 29 = ~110 ms |
| F4 (stretch) | Whole forward in one shader (single workgroup loop over layers) | ~200 → 1 | targets <20 ms/forward |

Each phase ships a new `<name>_fused.wgsl` shader + a `<name>FusedT` JS
function + a test that compares to existing `<name>BlockT` numerically
(atol depends on accumulation order — should be ≤ 1e-3 abs).

## Constraints

- WGSL workgroup memory: ≤ 16 KB (most adapters). For d6, n_embd=384 →
  intermediates fit comfortably. For d12 (n_embd=768) some intermediates
  approach the limit.
- WGSL workgroup size: max 256 threads on most platforms.
- WGSL has no inter-workgroup sync within a dispatch. So multi-layer
  fusion in ONE dispatch is single-workgroup only (loses parallelism).
  We accept that for the stretch phase only.

## Acceptance per phase

- Numerical: existing test for the block remains green; new fused-block
  test compares fused output to non-fused (atol 5e-4 abs).
- 30/30 still green.
- Bench drops by the expected savings (within ~30%).

## Verification

After each phase:
```
SUOMICHAT_BASE_DIR=/home/janitor/llm-training/data-fi-v2 .venv/bin/python web/test_runtime.py
```
must show 30/30 + the bench number.
