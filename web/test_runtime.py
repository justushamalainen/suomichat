#!/usr/bin/env python
"""
Headless-Chrome test harness for the WebGPU runtime.

Uses Playwright to launch Chromium with WebGPU enabled, navigates to
the local dev server, runs a JavaScript test, captures the result,
and compares against a PyTorch reference.

Prereqs:
    uv pip install playwright
    python -m playwright install --with-deps chromium

Run:
    cd web
    python convert.py --model-tag d12   # one-time, ~30s
    python server.py &                  # in another terminal
    python test_runtime.py

The test harness uses a special test.html page that exposes the
runtime ops as window functions, so we can call them from Python.
"""
import argparse
import asyncio
import json
import sys
import os
from pathlib import Path

import torch
from playwright.async_api import async_playwright

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from suomichat.checkpoint_manager import load_model


# ----------------------------------------------------------------------
# Reference (PyTorch) computations — one helper per WebGPU op we test
# ----------------------------------------------------------------------
def ref_embedding(model, token_id: int):
    """PyTorch reference for embedding lookup. Returns a (n_embd,) tensor."""
    return model.transformer.wte.weight[token_id].detach().cpu().float()


def ref_rmsnorm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """PyTorch reference for RMSNorm (no learnable weight, last-dim normalised)."""
    return torch.nn.functional.rms_norm(x, (x.size(-1),), eps=eps)


def ref_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch reference for row-major fp32 matmul: C = A @ B."""
    return a @ b


def ref_add(a, b):         return a + b
def ref_mul(a, b):         return a * b
def ref_scalar_mul(a, alpha): return alpha * a
def ref_relu2(a):          return torch.nn.functional.relu(a).square()
def ref_sigmoid(a):        return torch.sigmoid(a)


def ref_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """PyTorch reference matching suomichat.gpt.apply_rotary_emb exactly."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


def ref_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """PyTorch reference for nn.Linear without bias: y = x @ weight.T"""
    return torch.nn.functional.linear(x, weight)


# ----------------------------------------------------------------------
# Headless Chrome driver
# ----------------------------------------------------------------------
CHROMIUM_FLAGS = [
    # WebGPU is gated behind these flags in 2026-04 Chromium builds.
    "--enable-unsafe-webgpu",
    # Needed for timestamp-query feature at full nanosecond resolution
    # (default Chromium quantises timestamps to 100 µs).
    "--enable-webgpu-developer-features",
    "--enable-features=Vulkan",
    "--use-vulkan",
    "--enable-features=VulkanFromANGLE",
    # Headless Chrome on a headless Linux host needs these to find the GPU.
    "--ignore-gpu-blocklist",
    "--disable-gpu-sandbox",
    "--no-sandbox",
    # Reduce noise.
    "--disable-dev-shm-usage",
]


async def run_browser_op(op_name: str, args: dict, *, base_url: str = "http://localhost:9876"):
    """Launch headless Chrome, navigate to test.html, call an op, return its result.

    The page exposes a `window.run(opName, args)` helper that returns a
    JSON-serialisable result. We capture it via page.evaluate.
    """
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(args=CHROMIUM_FLAGS, headless=True)
        ctx = await browser.new_context()
        page = await ctx.new_page()
        page.on("console", lambda msg: print(f"[browser {msg.type}] {msg.text}"))
        page.on("pageerror", lambda exc: print(f"[browser error] {exc}"))

        await page.goto(f"{base_url}/test.html")

        # Wait for `window.runtimeReady` set by test.html after WebGPU + weights load
        await page.wait_for_function("window.runtimeReady === true", timeout=120_000)

        result = await page.evaluate(
            "async ({op, args}) => await window.run(op, args)",
            {"op": op_name, "args": args},
        )

        await browser.close()
        return result


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
async def test_embedding(model, token_id: int, atol: float = 1e-5):
    print(f"--- test: embedding[{token_id}] ---")
    ref = ref_embedding(model, token_id).numpy()
    got = await run_browser_op("embedding", {"tokenId": token_id})
    got = torch.tensor(got, dtype=torch.float32).numpy()

    if got.shape != ref.shape:
        print(f"  SHAPE MISMATCH: ref={ref.shape}, got={got.shape}")
        return False
    diff = abs(ref - got).max()
    print(f"  max abs diff: {diff:.3e}  (tol {atol:.0e})")
    print(f"  ref[:8]: {ref[:8]}")
    print(f"  got[:8]: {got[:8]}")
    if diff < atol:
        print("  ✓ PASS")
        return True
    else:
        print("  ✗ FAIL")
        return False


async def test_matmul(seed: int = 0, M: int = 1, K: int = 768, N: int = 768, atol: float = 5e-4):
    print(f"--- test: matmul ({M}x{K}) @ ({K}x{N})  seed={seed} ---")
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(M, K, dtype=torch.float32, generator=g)
    b = torch.randn(K, N, dtype=torch.float32, generator=g)
    ref = ref_matmul(a, b).numpy()

    got = await run_browser_op("matmul", {
        "a": a.flatten().tolist(),
        "aShape": [M, K],
        "b": b.flatten().tolist(),
        "bShape": [K, N],
    })
    got = torch.tensor(got, dtype=torch.float32).view(M, N).numpy()

    if got.shape != ref.shape:
        print(f"  SHAPE MISMATCH: ref={ref.shape}, got={got.shape}")
        return False
    diff = abs(ref - got).max()
    rel = diff / (abs(ref).max() + 1e-12)
    print(f"  max abs diff: {diff:.3e}  rel: {rel:.3e}  (tol {atol:.0e})")
    print(f"  ref[0, :6]: {ref[0, :6]}")
    print(f"  got[0, :6]: {got[0, :6]}")
    if diff < atol:
        print("  ✓ PASS")
        return True
    print("  ✗ FAIL")
    return False


async def _run_elem(op: str, pyref, args_js: dict, ref_args: tuple, atol: float):
    print(f"--- test: {op} ---")
    ref = pyref(*ref_args).numpy()
    got = await run_browser_op(op, args_js)
    got = torch.tensor(got, dtype=torch.float32).numpy()
    if got.shape != ref.shape:
        print(f"  SHAPE MISMATCH: ref={ref.shape}, got={got.shape}")
        return False
    diff = abs(ref - got).max()
    print(f"  max abs diff: {diff:.3e}  (tol {atol:.0e})")
    if diff <= atol:
        print("  ✓ PASS")
        return True
    print(f"  ref[:4]: {ref[:4]}")
    print(f"  got[:4]: {got[:4]}")
    print("  ✗ FAIL")
    return False


async def test_elementwise(seed: int = 0, n: int = 768):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, dtype=torch.float32, generator=g)
    b = torch.randn(n, dtype=torch.float32, generator=g)
    alpha = 3.5
    results = [
        await _run_elem("add", ref_add, {"a": a.tolist(), "b": b.tolist()}, (a, b), 0),
        await _run_elem("mul", ref_mul, {"a": a.tolist(), "b": b.tolist()}, (a, b), 0),
        await _run_elem("scalar_mul", ref_scalar_mul, {"a": a.tolist(), "alpha": alpha}, (a, alpha), 0),
        await _run_elem("relu2", ref_relu2, {"a": a.tolist()}, (a,), 0),
        await _run_elem("sigmoid", ref_sigmoid, {"a": a.tolist()}, (a,), 1e-6),  # exp drift
    ]
    return all(results)


async def test_softmax(seed: int = 0, rows: int = 6, n: int = 64, atol: float = 1e-5):
    print(f"--- test: softmax (rows={rows}, n={n}, seed={seed}) ---")
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, n, dtype=torch.float32, generator=g)
    ref = torch.softmax(x, dim=-1).numpy()
    got = await run_browser_op("softmax", {
        "x": x.flatten().tolist(), "rows": rows, "n": n,
    })
    got = torch.tensor(got, dtype=torch.float32).view(rows, n).numpy()
    diff = abs(ref - got).max()
    row_sums = got.sum(axis=-1)
    print(f"  max abs diff: {diff:.3e}  (tol {atol:.0e})")
    print(f"  row sums (should all be ~1): min={row_sums.min():.6f} max={row_sums.max():.6f}")
    if diff <= atol:
        print("  ✓ PASS")
        return True
    print("  ✗ FAIL")
    return False


async def test_full_forward(model, token_id: int = 100, atol: float = 0.3):
    """Run the full 12-layer forward for a single token. Compare to
    PyTorch's first-position logits from a 2-token forward (so the
    model's T>1 assertion doesn't trip; smear doesn't touch position 0).

    Acceptance: (a) argmax must match exactly (the functional property
    that matters for greedy sampling), AND (b) absolute logit diff
    under `atol`. 0.3 absolute covers the 12-layer compound fp32
    accumulation drift through 32K-col lm_head matmul + softcap;
    relative error stays ~1%.
    """
    print(f"--- test: full_forward token_id={token_id} ---")
    tokens = torch.tensor([[token_id, token_id]])
    with torch.no_grad():
        logits_all = model(tokens).cpu().float().numpy()
    ref = logits_all[0, 0]

    got = await run_browser_op("full_forward", {"tokenId": token_id})
    got = torch.tensor(got, dtype=torch.float32).numpy()

    if got.shape != ref.shape:
        print(f"  SHAPE MISMATCH: ref={ref.shape}, got={got.shape}")
        return False

    diff = abs(ref - got).max()
    rel = diff / (abs(ref).max() + 1e-12)
    ref_argmax = int(ref.argmax())
    got_argmax = int(got.argmax())
    ref_top3 = ref.argsort()[-3:][::-1].tolist()
    got_top3 = got.argsort()[-3:][::-1].tolist()
    print(f"  max abs diff: {diff:.3e}  rel: {rel:.3e}  (tol {atol:.2f})")
    print(f"  argmax — ref: {ref_argmax}  got: {got_argmax}  {'MATCH' if ref_argmax == got_argmax else 'MISMATCH'}")
    print(f"  top-3 ref: {ref_top3}  got: {got_top3}")
    ok = diff <= atol and ref_argmax == got_argmax
    if ok:
        print("  ✓ PASS")
        return True
    print("  ✗ FAIL")
    return False


async def test_tokenizer_encode(prompt_text: str = "Moi! Kuka olet?"):
    """JS BPE encoder vs PyTorch tiktoken encode_ordinary. Exact match."""
    print(f"--- test: tokenizer_encode {prompt_text!r} ---")
    from suomichat.tokenizer import get_tokenizer
    tok = get_tokenizer()
    ref = tok.enc.encode_ordinary(prompt_text)
    got = await run_browser_op("tokenizer_encode", {"text": prompt_text})
    print(f"  ref: {ref}")
    print(f"  got: {got}")
    if ref == got:
        print("  ✓ PASS")
        return True
    print("  ✗ FAIL")
    return False


async def test_tokenizer_render(prompt_text: str = "Moi! Kuka olet?"):
    """JS renderForCompletion vs PyTorch render_for_completion. Exact match."""
    print(f"--- test: tokenizer_render {prompt_text!r} ---")
    from suomichat.tokenizer import get_tokenizer
    tok = get_tokenizer()
    conv = {"messages": [{"role": "user", "content": prompt_text},
                          {"role": "assistant", "content": ""}]}
    ref = tok.render_for_completion(conv)
    got = await run_browser_op("tokenizer_render", {
        "messages": [{"role": "user", "content": prompt_text}],
    })
    print(f"  ref: {ref}")
    print(f"  got: {got}")
    if ref == got:
        print("  ✓ PASS")
        return True
    print("  ✗ FAIL")
    return False


async def test_tokenizer_decode(prompt_text: str = "Moi! Kuka olet?"):
    """Decode-roundtrip: encode in PyTorch, decode in JS, compare strings."""
    print(f"--- test: tokenizer_decode {prompt_text!r} ---")
    from suomichat.tokenizer import get_tokenizer
    tok = get_tokenizer()
    ids = tok.enc.encode_ordinary(prompt_text)
    ref = tok.enc.decode(ids)
    got = await run_browser_op("tokenizer_decode", {"ids": ids, "skipSpecial": True})
    print(f"  ids: {ids}")
    print(f"  ref: {ref!r}")
    print(f"  got: {got!r}")
    if ref == got:
        print("  ✓ PASS")
        return True
    print("  ✗ FAIL")
    return False


async def test_chat_e2e(model, prompt_text: str = "Moi! Kuka olet?", max_new: int = 8):
    """End-to-end: render chat prompt, generate in browser, compare token
    sequences against PyTorch greedy reference. Also decode and print so
    we see what the model actually said."""
    print(f"--- test: chat_e2e {prompt_text!r} max_new={max_new} ---")
    from suomichat.tokenizer import get_tokenizer
    tok = get_tokenizer()
    conv = {"messages": [{"role": "user", "content": prompt_text}, {"role": "assistant", "content": ""}]}
    prompt_ids = tok.render_for_completion(conv)

    # PyTorch greedy reference
    ref_ids = list(prompt_ids)
    with torch.no_grad():
        for _ in range(max_new):
            t = torch.tensor([ref_ids])
            logits = model(t)[0, -1].cpu().float().numpy()
            ref_ids.append(int(logits.argmax()))

    # WebGPU generation
    got_ids = await run_browser_op("greedy_generate", {
        "promptTokens": list(prompt_ids),
        "maxNew": max_new,
        "maxSeqLen": len(prompt_ids) + max_new + 4,
    })

    # Decode for visibility
    got_tail = got_ids[len(prompt_ids):]
    ref_tail = ref_ids[len(prompt_ids):]
    text = tok.decode(got_tail)
    print(f"  prompt: {prompt_ids}")
    print(f"  ref tail: {ref_tail}")
    print(f"  got tail: {got_tail}")
    print(f"  decoded: {text!r}")

    matched = 0
    for i in range(max_new):
        if ref_tail[i] == got_tail[i]: matched += 1
        else: break
    required = (max_new + 1) // 2
    print(f"  matched {matched}/{max_new} (required >= {required})")
    if matched >= required:
        print("  ✓ PASS")
        return True
    print("  ✗ FAIL")
    return False


async def test_greedy_generate(model, prompt_tokens=(100, 7), max_new: int = 8):
    """Greedy autoregressive generation. Compares the WebGPU output token
    sequence against PyTorch ref (forward + argmax loop).

    Acceptance: exact token-for-token match.

    Note: small drift in fp32 accumulation can flip an argmax late in the
    sequence (0.3 abs logit diff vs ref). We tolerate that — but only by
    measuring how many initial tokens match, and require >= ceil(max_new/2).
    """
    print(f"--- test: greedy_generate prompt={list(prompt_tokens)} max_new={max_new} ---")

    # PyTorch reference: same loop the runtime does.
    ref_tokens = list(prompt_tokens)
    with torch.no_grad():
        for _ in range(max_new):
            t = torch.tensor([ref_tokens])
            logits = model(t)[0, -1].cpu().float().numpy()
            ref_tokens.append(int(logits.argmax()))

    got_tokens = await run_browser_op("greedy_generate", {
        "promptTokens": list(prompt_tokens),
        "maxNew": max_new,
        "maxSeqLen": len(prompt_tokens) + max_new + 1,
    })

    print(f"  ref: {ref_tokens}")
    print(f"  got: {got_tokens}")

    # Count consecutive matches starting from prompt end
    n_prompt = len(prompt_tokens)
    matched = 0
    for i in range(max_new):
        if ref_tokens[n_prompt + i] == got_tokens[n_prompt + i]:
            matched += 1
        else:
            break
    required = (max_new + 1) // 2
    print(f"  matched {matched}/{max_new} consecutive (required >= {required})")
    if matched >= required:
        print("  ✓ PASS")
        return True
    print("  ✗ FAIL")
    return False


async def test_forward_batch(model, prompt_tokens=(100, 7, 42, 200), atol: float = 0.5):
    """Multi-token prefill: forwardBatchT([t0..tN-1]) vs PyTorch model([[t0..tN-1]]).
    Compares last-position logits (which is what we'd sample for the next token).
    Acceptance: argmax matches + abs diff under tol."""
    print(f"--- test: forward_batch tokens={list(prompt_tokens)} ---")
    import numpy as np
    tokens = torch.tensor([list(prompt_tokens)])
    with torch.no_grad():
        ref_logits = model(tokens).cpu().float().numpy()
    ref = ref_logits[0, -1]   # last position
    got = await run_browser_op("forward_batch", {"tokenIds": list(prompt_tokens)})
    got = np.array(got, dtype=np.float32)
    diff = float(np.abs(ref - got).max())
    rel  = diff / (float(np.abs(ref).max()) + 1e-12)
    ref_top3 = ref.argsort()[-3:][::-1].tolist()
    got_top3 = got.argsort()[-3:][::-1].tolist()
    print(f"  max abs diff: {diff:.3e}  rel: {rel:.3e}  (tol {atol:.2f})")
    print(f"  argmax — ref: {int(ref.argmax())}  got: {int(got.argmax())}  {'MATCH' if ref.argmax()==got.argmax() else 'MISMATCH'}")
    print(f"  top-3 ref: {ref_top3}  got: {got_top3}")
    ok = (diff <= atol) and (int(ref.argmax()) == int(got.argmax()))
    if ok:
        print("  ✓ PASS")
        return True
    print("  ✗ FAIL")
    return False


async def test_kv_cache_two_tokens(model, t0: int = 100, t1: int = 7, atol: float = 0.5):
    """Two-step decode with KV cache + smear gate (smear fires on step 2).
    Compares the WebGPU position-1 logits to PyTorch's logits[0, 1]
    from a single 2-token forward (which exercises the same SDPA, smear,
    x0 paths but in prefill form).

    Acceptance: argmax must match exactly + abs diff under `atol`.
    `atol` here is more permissive than single-token (0.3) because the
    two paths differ structurally — PyTorch prefill computes both
    positions in one batched matmul, while WebGPU decode runs each token
    through its own per-step SDPA over a growing KV cache. Pure fp32
    accumulation order accounts for ~1% relative drift across 12 layers.
    """
    print(f"--- test: kv_cache_two_tokens t0={t0} t1={t1} ---")
    tokens = torch.tensor([[t0, t1]])
    with torch.no_grad():
        logits_all = model(tokens).cpu().float().numpy()
    ref = logits_all[0, 1]    # position-1 logits

    got = await run_browser_op("two_tokens", {"t0": t0, "t1": t1, "maxSeqLen": 32})
    got = torch.tensor(got, dtype=torch.float32).numpy()

    if got.shape != ref.shape:
        print(f"  SHAPE MISMATCH: ref={ref.shape}, got={got.shape}")
        return False

    diff = abs(ref - got).max()
    rel = diff / (abs(ref).max() + 1e-12)
    ref_argmax = int(ref.argmax())
    got_argmax = int(got.argmax())
    ref_top3 = ref.argsort()[-3:][::-1].tolist()
    got_top3 = got.argsort()[-3:][::-1].tolist()
    print(f"  max abs diff: {diff:.3e}  rel: {rel:.3e}  (tol {atol:.2f})")
    print(f"  argmax — ref: {ref_argmax}  got: {got_argmax}  {'MATCH' if ref_argmax == got_argmax else 'MISMATCH'}")
    print(f"  top-3 ref: {ref_top3}  got: {got_top3}")
    ok = diff <= atol and ref_argmax == got_argmax
    if ok:
        print("  ✓ PASS")
        return True
    print("  ✗ FAIL")
    return False


async def test_transformer_block(model, layer_idx: int = 0, seed: int = 0, atol: float = 5e-4):
    """Full block: resid_lambda mix + attn + mlp + residuals. Against model.transformer.h[i]."""
    from suomichat.gpt import has_ve
    print(f"--- test: transformer_block layer {layer_idx} ---")
    g = torch.Generator().manual_seed(seed)
    n_embd = model.config.n_embd
    n_layer = model.config.n_layer

    x = torch.randn(1, 1, n_embd, dtype=torch.float32, generator=g)
    x0 = torch.randn(1, 1, n_embd, dtype=torch.float32, generator=g)

    T0 = 0
    cos = model.cos[:, T0:T0+1]
    sin = model.sin[:, T0:T0+1]

    # Compute the outer-forward mix manually (this is in GPT.forward, not Block.forward)
    residL = model.resid_lambdas[layer_idx].item()
    x0L    = model.x0_lambdas[layer_idx].item()
    x_mixed = residL * x + x0L * x0

    if has_ve(layer_idx, n_layer):
        ve_tensor = model.value_embeds[str(layer_idx)](torch.tensor([[0]]))
        ve_arg = ve_tensor.flatten().tolist()
    else:
        ve_tensor = None
        ve_arg = None

    with torch.no_grad():
        ref = model.transformer.h[layer_idx](x_mixed, ve_tensor, (cos, sin), (-1, 0), None).cpu().float().numpy()

    got = await run_browser_op("block", {
        "x": x.flatten().tolist(),
        "x0": x0.flatten().tolist(),
        "layerIdx": layer_idx,
        "T0": T0,
        "ve": ve_arg,
    })
    got = torch.tensor(got, dtype=torch.float32).view(ref.shape).numpy()
    diff = abs(ref - got).max()
    print(f"  max abs diff: {diff:.3e}  (tol {atol:.0e})")
    if diff <= atol:
        print("  ✓ PASS")
        return True
    print(f"  ref[0, 0, :6]: {ref[0, 0, :6]}")
    print(f"  got[0, 0, :6]: {got[0, 0, :6]}")
    print("  ✗ FAIL")
    return False


async def test_attention_block(model, layer_idx: int = 0, seed: int = 0, atol: float = 5e-4):
    """Verify attentionBlock against model.transformer.h[i].attn() for T=1."""
    from suomichat.gpt import has_ve
    print(f"--- test: attention_block layer {layer_idx} ---")
    g = torch.Generator().manual_seed(seed)
    n_embd = model.config.n_embd
    n_layer = model.config.n_layer
    head_dim = n_embd // model.config.n_head

    x = torch.randn(1, 1, n_embd, dtype=torch.float32, generator=g)
    T0 = 0
    cos = model.cos[:, T0:T0+1]
    sin = model.sin[:, T0:T0+1]

    # value embedding path (Phase 7): active on alternating layers
    layer_has_ve = has_ve(layer_idx, n_layer)
    if layer_has_ve:
        fake_token = torch.tensor([[0]])  # any token id; just need the shape
        ve_tensor = model.value_embeds[str(layer_idx)](fake_token)  # (1, 1, n_kv_head*head_dim)
        ve_arg = ve_tensor.flatten().tolist()
    else:
        ve_tensor = None
        ve_arg = None

    # Window size: full context for our Phase 6 (we bypass sliding window)
    with torch.no_grad():
        ref = model.transformer.h[layer_idx].attn(
            x, ve_tensor, (cos, sin), (-1, 0), None
        ).cpu().float().numpy()

    got = await run_browser_op("attention", {
        "x": x.flatten().tolist(),
        "layerIdx": layer_idx,
        "T0": T0,
        "ve": ve_arg,
    })
    got = torch.tensor(got, dtype=torch.float32).view(ref.shape).numpy()

    diff = abs(ref - got).max()
    rel = diff / (abs(ref).max() + 1e-12)
    print(f"  max abs diff: {diff:.3e}  rel: {rel:.3e}  (tol {atol:.0e})")
    print(f"  has_ve: {layer_has_ve}")
    if diff <= atol:
        print("  ✓ PASS")
        return True
    print(f"  ref[0, 0, :6]: {ref[0, 0, :6]}")
    print(f"  got[0, 0, :6]: {got[0, 0, :6]}")
    print("  ✗ FAIL")
    return False


async def test_mlp_block(model, layer_idx: int = 0, seed: int = 0, atol: float = 5e-4):
    """Verify our mlpBlock against model.transformer.h[i].mlp on a random x."""
    print(f"--- test: mlp_block layer {layer_idx} ---")
    g = torch.Generator().manual_seed(seed)
    n_embd = model.config.n_embd
    x = torch.randn(1, 1, n_embd, dtype=torch.float32, generator=g)
    with torch.no_grad():
        ref = model.transformer.h[layer_idx].mlp(x).cpu().float().numpy()

    got = await run_browser_op("mlp", {"x": x.flatten().tolist(), "layerIdx": layer_idx})
    got = torch.tensor(got, dtype=torch.float32).view(ref.shape).numpy()
    diff = abs(ref - got).max()
    print(f"  max abs diff: {diff:.3e}  (tol {atol:.0e})")
    if diff <= atol:
        print("  ✓ PASS")
        return True
    print(f"  ref[0, 0, :6]: {ref[0, 0, :6]}")
    print(f"  got[0, 0, :6]: {got[0, 0, :6]}")
    print("  ✗ FAIL")
    return False


async def test_linear(model, weight_name: str, seed: int = 0, atol: float = 5e-4):
    """Verify our `linear` shader against nn.Linear using a real model weight."""
    print(f"--- test: linear '{weight_name}' ---")
    w = dict(model.named_parameters())[weight_name].detach().cpu().float()
    N, K = w.shape
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, K, dtype=torch.float32, generator=g)
    ref = ref_linear(x, w).numpy()

    got = await run_browser_op("linear", {
        "x": x.flatten().tolist(),
        "M": 1, "K": K,
        "weightName": weight_name,
    })
    got = torch.tensor(got, dtype=torch.float32).view(1, N).numpy()
    diff = abs(ref - got).max()
    print(f"  max abs diff: {diff:.3e}  (tol {atol:.0e})")
    if diff <= atol:
        print("  ✓ PASS")
        return True
    print(f"  ref[0, :6]: {ref[0, :6]}")
    print(f"  got[0, :6]: {got[0, :6]}")
    print("  ✗ FAIL")
    return False


async def test_sdpa_t(model, T: int = 4, seed: int = 0, atol: float = 5e-4):
    """T>1 SDPA with causal mask. Compare to PyTorch
    F.scaled_dot_product_attention with is_causal=True."""
    print(f"--- test: sdpa_t (T={T}) ---")
    import numpy as np
    g = torch.Generator().manual_seed(seed)
    head_dim = model.config.n_embd // model.config.n_head
    nH       = model.config.n_head
    nKV      = model.config.n_kv_head
    # Random Q (T, nH, hd), K/V (T, nKV, hd)
    q = torch.randn(T, nH,  head_dim, dtype=torch.float32, generator=g)
    k = torch.randn(T, nKV, head_dim, dtype=torch.float32, generator=g)
    v = torch.randn(T, nKV, head_dim, dtype=torch.float32, generator=g)
    # PyTorch reference: causal attention. SDPA expects (B, H, T, hd).
    # Need to expand K, V to nH heads if GQA (here nH==nKV for d6 so identity).
    q_t = q.transpose(0, 1).unsqueeze(0)         # (1, nH, T, hd)
    k_t = k.transpose(0, 1).unsqueeze(0)         # (1, nKV, T, hd)
    v_t = v.transpose(0, 1).unsqueeze(0)
    # Repeat K, V if GQA
    if nKV != nH:
        k_t = k_t.repeat_interleave(nH // nKV, dim=1)
        v_t = v_t.repeat_interleave(nH // nKV, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
    ref = ref.squeeze(0).transpose(0, 1).flatten().numpy()   # (T, nH, hd) flat

    got = await run_browser_op("sdpa_t", {
        "T": T, "nH": nH, "nKV": nKV, "hd": head_dim,
        "q": q.flatten().tolist(),
        "k": k.flatten().tolist(),
        "v": v.flatten().tolist(),
    })
    got = np.array(got, dtype=np.float32)
    diff = float(np.abs(ref - got).max())
    print(f"  max abs diff: {diff:.3e} (tol {atol:.0e})")
    if diff <= atol:
        print("  ✓ PASS")
        return True
    print("  ✗ FAIL")
    return False


async def test_rope_multi(model, T: int = 4, T0: int = 5, seed: int = 0, atol: float = 1e-5):
    """Multi-position RoPE: T tokens, n_head heads each, positions T0..T0+T-1."""
    print(f"--- test: rope_multi (T={T}, T0={T0}) ---")
    g = torch.Generator().manual_seed(seed)
    head_dim = model.config.n_embd // model.config.n_head
    n_head   = model.config.n_head
    # x shape (1, T, n_head, head_dim)
    x = torch.randn(1, T, n_head, head_dim, dtype=torch.float32, generator=g)
    # PyTorch ref: per-token RoPE
    ref_chunks = []
    for t in range(T):
        cos = model.cos[:, T0+t:T0+t+1]
        sin = model.sin[:, T0+t:T0+t+1]
        rot = ref_rope(x[:, t:t+1], cos, sin)
        ref_chunks.append(rot.flatten())
    ref = torch.cat(ref_chunks).numpy()

    # Browser: pass flat (N=T*n_head, D=head_dim) with H=n_head
    got = await run_browser_op("rope", {
        "x": x.flatten().tolist(),
        "N": T * n_head, "D": head_dim, "T0": T0, "H": n_head,
    })
    got = torch.tensor(got, dtype=torch.float32).numpy()
    diff = abs(ref - got).max()
    print(f"  max abs diff: {diff:.3e} (tol {atol:.0e})")
    if diff <= atol:
        print("  ✓ PASS")
        return True
    print("  ✗ FAIL")
    return False


async def test_rope(model, seed: int = 0, T0: int = 0, atol: float = 1e-5):
    print(f"--- test: rope (T0={T0}) ---")
    g = torch.Generator().manual_seed(seed)
    # d12 config: n_head=6, head_dim = n_embd/n_head = 768/6 = 128
    head_dim = model.config.n_embd // model.config.n_head
    n_heads = model.config.n_head

    # Shape matches attention code: (B=1, T=1, H=nHeads, D=head_dim)
    x = torch.randn(1, 1, n_heads, head_dim, dtype=torch.float32, generator=g)
    cos = model.cos[:, T0:T0+1]   # (1, 1, 1, d)
    sin = model.sin[:, T0:T0+1]
    ref = ref_rope(x, cos, sin).detach().cpu().float().numpy().flatten()

    # Browser: pass flat (N=H, D=head_dim)
    got = await run_browser_op("rope", {
        "x": x.flatten().tolist(),
        "N": n_heads, "D": head_dim, "T0": T0,
    })
    got = torch.tensor(got, dtype=torch.float32).numpy()

    if got.shape != ref.shape:
        print(f"  SHAPE MISMATCH: ref={ref.shape}, got={got.shape}")
        return False
    diff = abs(ref - got).max()
    print(f"  max abs diff: {diff:.3e}  (tol {atol:.0e})")
    print(f"  ref[:6]: {ref[:6]}")
    print(f"  got[:6]: {got[:6]}")
    if diff <= atol:
        print("  ✓ PASS")
        return True
    print("  ✗ FAIL")
    return False


async def test_rmsnorm(seed: int = 0, n: int = 768, atol: float = 1e-5):
    print(f"--- test: rmsnorm (n={n}, seed={seed}) ---")
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dtype=torch.float32, generator=g)
    ref = ref_rmsnorm(x).numpy()

    got = await run_browser_op("rmsnorm", {"input": x.tolist(), "eps": 1e-5})
    got = torch.tensor(got, dtype=torch.float32).numpy()

    if got.shape != ref.shape:
        print(f"  SHAPE MISMATCH: ref={ref.shape}, got={got.shape}")
        return False
    diff = abs(ref - got).max()
    rel = diff / (abs(ref).max() + 1e-12)
    print(f"  max abs diff: {diff:.3e}  rel: {rel:.3e}  (tol {atol:.0e})")
    print(f"  ref[:8]: {ref[:8]}")
    print(f"  got[:8]: {got[:8]}")
    if diff < atol:
        print("  ✓ PASS")
        return True
    else:
        print("  ✗ FAIL")
        return False


# ----------------------------------------------------------------------
BASELINE_MS_PER_TOKEN = 140.0   # forwardT pre-batching, RTX 6000 Ada, d6 SFT
BASELINE_LEGACY_MS    = 180.0   # _forward_legacy reference

async def run_bench(N: int, token_id: int):
    new_b = await run_browser_op("bench_forward", {"tokenId": token_id, "N": N, "legacy": False})
    old_b = await run_browser_op("bench_forward", {"tokenId": token_id, "N": N, "legacy": True})
    summary = {
        "N": N,
        "token_id": token_id,
        "forwardT_ms_per_token": round(new_b["msPerToken"], 2),
        "legacy_ms_per_token":   round(old_b["msPerToken"], 2),
        "baseline_ms_per_token": BASELINE_MS_PER_TOKEN,
        "speedup_vs_legacy":     round(old_b["ms"] / new_b["ms"], 2),
        "speedup_vs_baseline":   round(BASELINE_MS_PER_TOKEN / new_b["msPerToken"], 2),
    }
    return summary


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", default="d6")
    ap.add_argument("--source", default="sft")
    ap.add_argument("--token-id", type=int, default=100)
    ap.add_argument("--base-url", default="http://localhost:9876")
    ap.add_argument("--bench-only", action="store_true",
                    help="Skip correctness tests; just run the bench and print JSON.")
    ap.add_argument("--profile", action="store_true",
                    help="GPU-side per-dispatch profile via timestamp-query; prints histogram.")
    ap.add_argument("--bench-n", type=int, default=50,
                    help="Iterations per bench run (default 50).")
    args = ap.parse_args()

    print(f"Loading {args.source} model {args.model_tag} for reference computation...")
    model, tokenizer, meta = load_model(
        args.source, device=torch.device("cpu"), phase="eval", model_tag=args.model_tag,
    )

    if getattr(args, "profile", False):
        # Need a model load to populate the page first
        await test_embedding(model, args.token_id)
        print("\n--- GPU profile (N=20 decode forwards) ---")
        rep = await run_browser_op("bench_profile", {"tokenId": args.token_id, "N": 20})
        print(f"wall:        {rep['wall_ms_per_token']:.2f} ms/token   ({rep['wall_ms']:.0f} ms total / {rep['N']} iters)")
        print(f"gpu-active:  {rep['gpu_total_ms_per_token']:.2f} ms/token   (sum of per-dispatch GPU time)")
        print(f"dispatches:  {rep['dispatches_per_token']:.0f} per token")
        print(f"gpu-idle:    {rep['wall_ms_per_token'] - rep['gpu_total_ms_per_token']:.2f} ms/token   (wall - gpu-active)\n")
        rows = sorted(rep["histogram"].items(), key=lambda kv: -kv[1]["total_ms"])
        n_tok = rep["N"]
        print(f"{'shader':<22} {'calls':>8} {'total_ms':>12} {'ms/call':>10} {'ms/tok':>10}")
        print("-" * 66)
        for shader, h in rows:
            print(f"{shader:<22} {h['count']:>8} {h['total_ms']:>12.2f} {h['avg_ms']:>10.3f} {h['total_ms']/n_tok:>10.3f}")
        sys.exit(0)

    if args.bench_only:
        # Need a model load to populate the page first — call any cheap op.
        await test_embedding(model, args.token_id)
        summary = await run_bench(args.bench_n, args.token_id)
        print(json.dumps(summary, indent=2))
        sys.exit(0)

    passed = []
    passed.append(await test_embedding(model, args.token_id))
    passed.append(await test_rmsnorm(seed=0, n=768))
    passed.append(await test_rmsnorm(seed=1, n=768))
    passed.append(await test_matmul(seed=0, M=1, K=768, N=768))       # shape: Q projection on one token
    passed.append(await test_matmul(seed=1, M=768, K=768, N=2048))    # shape: MLP c_fc
    passed.append(await test_matmul(seed=2, M=2, K=3, N=4))           # tiny sanity check
    passed.append(await test_elementwise(seed=0, n=768))              # add/mul/scalar_mul/relu2/sigmoid
    passed.append(await test_rope(model, seed=0, T0=0))
    passed.append(await test_rope(model, seed=1, T0=17))              # non-zero offset
    passed.append(await test_rope_multi(model, T=4, T0=5))           # T>1 prefill
    passed.append(await test_sdpa_t(model, T=4))                     # T>1 SDPA causal mask
    passed.append(await test_sdpa_t(model, T=8))
    passed.append(await test_linear(model, "transformer.h.0.attn.c_q.weight"))  # (768, 768)
    passed.append(await test_linear(model, "transformer.h.0.mlp.c_fc.weight"))  # (3072, 768) d12: 4*n_embd
    passed.append(await test_mlp_block(model, layer_idx=0))
    passed.append(await test_mlp_block(model, layer_idx=model.config.n_layer - 1))   # last layer sanity
    passed.append(await test_attention_block(model, layer_idx=0))               # no value embeds
    passed.append(await test_attention_block(model, layer_idx=1))               # has value embeds
    passed.append(await test_transformer_block(model, layer_idx=0))
    passed.append(await test_transformer_block(model, layer_idx=1))
    passed.append(await test_full_forward(model, token_id=100))
    passed.append(await test_full_forward(model, token_id=7))
    passed.append(await test_softmax(seed=0, rows=6, n=64))
    passed.append(await test_softmax(seed=1, rows=6, n=1))    # trivial 1-element softmax = 1.0
    passed.append(await test_kv_cache_two_tokens(model, t0=100, t1=7))
    passed.append(await test_kv_cache_two_tokens(model, t0=42, t1=42))  # repeat token edge
    passed.append(await test_forward_batch(model, prompt_tokens=(100, 7, 42, 200)))   # T=4 prefill
    passed.append(await test_greedy_generate(model, prompt_tokens=(100, 7), max_new=8))
    passed.append(await test_tokenizer_decode("Moi! Kuka olet?"))
    passed.append(await test_tokenizer_encode("Moi! Kuka olet?"))
    passed.append(await test_tokenizer_encode("hei maailma"))
    passed.append(await test_tokenizer_encode("Suomen pääkaupunki on Helsinki."))   # ä, multi-byte UTF-8
    passed.append(await test_tokenizer_render("Moi! Kuka olet?"))
    passed.append(await test_chat_e2e(model, "Moi! Kuka olet?", max_new=8))

    # Bench. Structured summary so the autonomous loop can detect regressions.
    print("--- bench ---")
    summary = await run_bench(args.bench_n, 100)
    print(json.dumps(summary, indent=2))

    print("--- bench: prefill batched vs stepwise ---")
    prefill = await run_browser_op("bench_prefill", {"N": 8, "iters": 10})
    print(json.dumps(prefill, indent=2))
    if prefill["speedup"] >= 3.0:
        print("  ✓ PASS (>= 3x prefill speedup)")
    else:
        print(f"  ✗ FAIL prefill speedup {prefill['speedup']:.2f}x < 3x target")

    n_pass = sum(passed)
    n_total = len(passed)
    print(f"\n{n_pass}/{n_total} tests passed")
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    asyncio.run(main())
