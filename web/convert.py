#!/usr/bin/env python
"""
Convert a suomichat PyTorch checkpoint into a browser-friendly format:

  weights.bin     flat fp32 binary, one tensor after another
  tensors.json    manifest: name -> {offset, shape, dtype}
  config.json     model architecture (depth, heads, dims, etc.)

Why two files?
  - weights.bin is huge (~1 GB for d12). Fetched as a single ArrayBuffer.
  - tensors.json is tiny. Tells JS where each weight starts and what shape it is.
  - config.json mirrors the PyTorch GPTConfig so JS can build the same model.

Usage:
    cd web
    python convert.py --model-tag d12 --source sft

Outputs land next to this script.
"""
import argparse
import json
import os
import sys
import struct
from pathlib import Path

# Run from web/ — find suomichat root one level up
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import torch
from suomichat.checkpoint_manager import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", default="d12", help="e.g. d12, d26")
    ap.add_argument("--source", default="sft", choices=["base", "sft"])
    ap.add_argument("--out-dir", default=str(HERE), help="where to write weights.bin etc.")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                    help="output dtype. float32 is easiest to debug; float16 halves the file size.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.source} model {args.model_tag}...")
    model, tokenizer, meta = load_model(args.source, device=torch.device("cpu"),
                                         phase="eval", model_tag=args.model_tag)
    cfg = meta["model_config"]
    print(f"  config: {cfg}")

    out_dtype = torch.float32 if args.dtype == "float32" else torch.float16
    bytes_per = 4 if args.dtype == "float32" else 2

    # Stream tensors to weights.bin and build the manifest.
    weights_path = out_dir / "weights.bin"
    tensors = {}
    offset = 0
    n_params = 0
    print(f"Writing {weights_path} ({args.dtype})...")
    with open(weights_path, "wb") as f:
        for name, param in model.named_parameters():
            t = param.detach().to(dtype=out_dtype).contiguous().cpu()
            tensors[name] = {
                "offset": offset,
                "shape": list(t.shape),
                "dtype": args.dtype,
            }
            data = t.numpy().tobytes()
            f.write(data)
            offset += len(data)
            n_params += t.numel()

        # Buffers (rotary cache etc.) — also useful in browser
        for name, buf in model.named_buffers():
            t = buf.detach().to(dtype=out_dtype).contiguous().cpu()
            tensors[name] = {
                "offset": offset,
                "shape": list(t.shape),
                "dtype": args.dtype,
                "_kind": "buffer",
            }
            data = t.numpy().tobytes()
            f.write(data)
            offset += len(data)

    print(f"  total: {n_params:,} params = {offset / 1024**2:.1f} MB at {args.dtype}")

    # tensors.json
    with open(out_dir / "tensors.json", "w") as f:
        json.dump({"total_size": offset, "bytes_per_element": bytes_per, "tensors": tensors}, f, indent=2)
    print(f"  manifest: {out_dir / 'tensors.json'}")

    # config.json — same shape as GPTConfig dataclass
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  config: {out_dir / 'config.json'}")

    print("Done.")


if __name__ == "__main__":
    main()
