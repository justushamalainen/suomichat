#!/usr/bin/env python
"""
Export the suomichat tokenizer to a browser-loadable JSON.

Output (web/tokenizer.json):
  {
    "n_vocab": int,
    "vocab_b64": [str, str, ...],          # id -> base64-encoded bytes
    "special_tokens": {name: id, ...},     # e.g. "<|bos|>" -> 0
    "bos_id": int,
    "eot_id": int,
  }

The encoder side (BPE merges + split regex) is intentionally NOT exported
yet — Phase 14a uses pre-tokenized prompts. Phase 14b will add the
encoder export and a JS BPE implementation.

Usage:
    SUOMICHAT_BASE_DIR=/home/janitor/llm-training/data-fi-v2 \\
        python web/export_tokenizer.py
"""
import argparse
import base64
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from suomichat.tokenizer import get_tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(Path(__file__).parent / "tokenizer.json"))
    args = p.parse_args()

    tok = get_tokenizer()
    enc = tok.enc
    n_vocab = enc.n_vocab

    vocab_b64 = []
    for tid in range(n_vocab):
        try:
            b = enc.decode_single_token_bytes(tid)
        except Exception:
            b = b""
        vocab_b64.append(base64.b64encode(b).decode("ascii"))

    # Special token name -> id
    special = {}
    for name in enc.special_tokens_set:
        try:
            special[name] = enc.encode_single_token(name)
        except Exception:
            pass

    # First non-special id (= len of mergeable_ranks). Anything < this is
    # a regular BPE merge; anything >= this is a special token.
    n_merges = len(enc._mergeable_ranks)

    out = {
        "n_vocab": n_vocab,
        "vocab_b64": vocab_b64,
        "n_merges": n_merges,
        "special_tokens": special,
        "bos_id": tok.bos_token_id,
        "eot_id": special.get("<|endoftext|>") or special.get("<|bos|>"),
        "pat_str": enc._pat_str,    # GPT-4-style split regex (Python re syntax)
    }
    Path(args.out).write_text(json.dumps(out))
    print(f"wrote {args.out}  ({n_vocab} tokens, {len(special)} special)")


if __name__ == "__main__":
    main()
