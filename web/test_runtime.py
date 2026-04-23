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


# ----------------------------------------------------------------------
# Headless Chrome driver
# ----------------------------------------------------------------------
CHROMIUM_FLAGS = [
    # WebGPU is gated behind these flags in 2026-04 Chromium builds.
    "--enable-unsafe-webgpu",
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
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", default="d12")
    ap.add_argument("--source", default="sft")
    ap.add_argument("--token-id", type=int, default=100)
    ap.add_argument("--base-url", default="http://localhost:9876")
    args = ap.parse_args()

    print(f"Loading {args.source} model {args.model_tag} for reference computation...")
    model, tokenizer, meta = load_model(
        args.source, device=torch.device("cpu"), phase="eval", model_tag=args.model_tag,
    )

    passed = []
    passed.append(await test_embedding(model, args.token_id))
    passed.append(await test_rmsnorm(seed=0, n=768))
    passed.append(await test_rmsnorm(seed=1, n=768))
    passed.append(await test_matmul(seed=0, M=1, K=768, N=768))       # shape: Q projection on one token
    passed.append(await test_matmul(seed=1, M=768, K=768, N=2048))    # shape: MLP c_fc
    passed.append(await test_matmul(seed=2, M=2, K=3, N=4))           # tiny sanity check
    passed.append(await test_elementwise(seed=0, n=768))              # add/mul/scalar_mul/relu2/sigmoid

    # Future: test_rope, test_attention, test_full_forward, etc.

    n_pass = sum(passed)
    n_total = len(passed)
    print(f"\n{n_pass}/{n_total} tests passed")
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    asyncio.run(main())
