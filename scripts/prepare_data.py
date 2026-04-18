#!/usr/bin/env python
"""
Download Finnish HuggingFace datasets and repackage into suomichat parquet shards.

Writes shards to $SUOMICHAT_BASE_DIR/base_data_climbmix/shard_XXXXX.parquet
(reuses the default suomichat dir name so suomichat/dataset.py finds them with
no URL-swap needed — just export SUOMICHAT_BASE_DIR to the Finnish data dir.)

Mix (by character budget):
    ~70% Finnish-NLP/Fineweb2_Finnish_fineweb_edu_predicted (edu_score>=2.0)
    ~20% Finnish-NLP/wikipedia_20230501_fi_cleaned (upsample 3x)
    ~10% Finnish-NLP/Reddit_fi_2006_2022

Usage:
    python prepare_finnish_data.py --num-shards 30          # default smoke-test size
    python prepare_finnish_data.py --num-shards 150         # full d20 run

Idempotent: skips any shard_XXXXX.parquet that already exists.
"""
import argparse
import os
import random
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

CHARS_PER_SHARD = 250_000_000
ROW_GROUP_SIZE = 1024
SHARD_FMT = "shard_{:05d}.parquet"

SOURCES = [
    # (hf_path, weight, split, text_col, filter_fn, upsample)
    # Field is `pred` (edu-score prediction, 0-5 scale). Filter >= 2.0 keeps ~top half.
    ("Finnish-NLP/Fineweb2_Finnish_fineweb_edu_predicted", 0.70, "train", "text",
     lambda r: (r.get("pred") or 0) >= 2.0, 1),
    ("Finnish-NLP/wikipedia_20230501_fi_cleaned", 0.20, "train", "text", None, 3),
    ("Finnish-NLP/Reddit_fi_2006_2022", 0.10, "train", "body", None, 1),
]


def stream_docs(hf_path, split, text_col, filter_fn, upsample, seed=42):
    """Yield document strings from a streaming HF dataset.
    Auto-reopens the stream on transient HTTP errors (ChunkedEncodingError etc.)
    to survive long-running jobs. The shuffle seed is perturbed per attempt so
    we don't loop on the exact same broken chunk.
    """
    attempt = 0
    while True:
        attempt += 1
        print(f"[{hf_path}] opening stream (split={split}, upsample={upsample}, attempt={attempt})", flush=True)
        try:
            ds = load_dataset(hf_path, split=split, streaming=True)
            ds = ds.shuffle(seed=seed + attempt, buffer_size=10_000)
            for row in ds:
                if filter_fn is not None and not filter_fn(row):
                    continue
                text = row.get(text_col)
                if not text or len(text) < 200:
                    continue
                for _ in range(upsample):
                    yield text
            # clean EOF
            return
        except Exception as e:
            print(f"[{hf_path}] stream error (attempt {attempt}): {type(e).__name__}: {str(e)[:160]}", flush=True)
            if attempt >= 20:
                print(f"[{hf_path}] giving up after {attempt} attempts", flush=True)
                return
            time.sleep(min(5 * attempt, 60))


def interleaved_mix(sources, rng):
    """Round-robin-ish weighted interleave of multiple doc streams."""
    iters = []
    for hf_path, weight, split, text_col, filt, upsample in sources:
        it = stream_docs(hf_path, split, text_col, filt, upsample)
        iters.append([it, weight, hf_path, True])  # [iter, weight, name, alive]
    while any(x[3] for x in iters):
        alive = [x for x in iters if x[3]]
        total = sum(x[1] for x in alive)
        r = rng.random() * total
        acc = 0.0
        for entry in alive:
            acc += entry[1]
            if r <= acc:
                try:
                    yield next(entry[0])
                except StopIteration:
                    entry[3] = False
                break


def write_shard(out_dir, idx, docs):
    path = os.path.join(out_dir, SHARD_FMT.format(idx))
    tmp = path + ".tmp"
    table = pa.Table.from_pydict({"text": docs})
    pq.write_table(table, tmp, row_group_size=ROW_GROUP_SIZE, compression="zstd")
    os.replace(tmp, path)
    print(f"  wrote {path} ({len(docs)} docs)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-shards", type=int, default=30)
    ap.add_argument("--out-dir", type=str,
                    default=os.path.join(
                        os.environ.get("SUOMICHAT_BASE_DIR",
                                       "/home/janitor/llm-training/data-fi"),
                        "base_data_climbmix"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    # Figure out which shards still need to be written (idempotent)
    todo = [i for i in range(args.num_shards)
            if not os.path.exists(os.path.join(args.out_dir, SHARD_FMT.format(i)))]
    if not todo:
        print(f"All {args.num_shards} shards already present in {args.out_dir}")
        return
    print(f"Will write {len(todo)} shards (of {args.num_shards}) to {args.out_dir}")

    stream = interleaved_mix(SOURCES, rng)

    current_docs = []
    current_chars = 0
    shard_cursor = todo[0]
    remaining = set(todo)
    start = time.time()

    try:
        while remaining:
            try:
                doc = next(stream)
            except StopIteration:
                print("Warning: source stream exhausted before target shard count", flush=True)
                break
            current_docs.append(doc)
            current_chars += len(doc)
            if current_chars >= CHARS_PER_SHARD:
                if shard_cursor in remaining:
                    write_shard(args.out_dir, shard_cursor, current_docs)
                    remaining.remove(shard_cursor)
                    elapsed = time.time() - start
                    print(f"  [{shard_cursor+1}/{args.num_shards}] elapsed {elapsed/60:.1f} min",
                          flush=True)
                shard_cursor += 1
                current_docs = []
                current_chars = 0
                # skip forward if later shards already exist
                while shard_cursor < args.num_shards and shard_cursor not in remaining:
                    shard_cursor += 1
        # flush trailing partial shard only if it meets threshold
        if current_docs and current_chars >= CHARS_PER_SHARD * 0.5 and shard_cursor in remaining:
            write_shard(args.out_dir, shard_cursor, current_docs)
    except KeyboardInterrupt:
        print("Interrupted; partial shard discarded.", flush=True)
        sys.exit(130)

    print(f"Done. Shards in {args.out_dir}")


if __name__ == "__main__":
    main()
