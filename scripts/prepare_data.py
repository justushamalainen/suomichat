#!/usr/bin/env python
"""
Download Finnish HuggingFace datasets and repackage into suomichat parquet shards.

Writes shards to $SUOMICHAT_BASE_DIR/base_data_climbmix/shard_XXXXX.parquet

ClimbMix-fi blend (Chinchilla-optimal for d20, ~18B tokens / 346 shards):
    35% Finnish Wikipedia      — knowledge-dense, ~6 epochs (epoch-looped)
    35% FineWeb2-fi (edu>=2.0) — high-quality web
    15% HPLT 1.2-fi            — long-form web content
    10% mC4-fi                 — diverse web crawl
     5% Reddit-fi              — informal register, epoch-looped

Usage:
    python -m scripts.prepare_data --num-shards 30    # d12 smoke test
    python -m scripts.prepare_data --num-shards 80    # d20 ratio=6
    python -m scripts.prepare_data --num-shards 346   # d20 Chinchilla

Idempotent: skips shards that already exist.
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
    # (hf_path, weight, split, text_col, filter_fn, min_doc_len)
    ("Finnish-NLP/Fineweb2_Finnish_fineweb_edu_predicted", 0.35, "train", "text",
     lambda r: (r.get("pred") or 0) >= 2.0, 200),
    ("Finnish-NLP/wikipedia_20230501_fi_cleaned", 0.35, "train", "text",
     None, 200),
    ("Finnish-NLP/HPLT_1.2_fi_cleaned", 0.15, "train", "text",
     None, 300),
    ("Finnish-NLP/mc4_fi_cleaned", 0.10, "train", "text",
     None, 200),
    ("Finnish-NLP/Reddit_fi_2006_2022", 0.05, "train", "body",
     None, 100),
]


def stream_docs_looping(hf_path, split, text_col, filter_fn, min_doc_len, seed=42):
    """Yield documents from a HF dataset, looping with reshuffled epochs on exhaustion.

    Unlike simple inline upsample (doc,doc,doc,...), this cycles through the
    entire dataset multiple times with different shuffle orders. Each epoch
    sees every document once in a new random order — much better for training.

    Auto-reopens on transient HTTP errors (ChunkedEncodingError etc.).
    """
    epoch = 0
    error_count = 0
    while True:
        epoch += 1
        epoch_seed = seed + epoch * 1000
        print(f"[{hf_path}] epoch {epoch} (seed={epoch_seed})", flush=True)
        try:
            ds = load_dataset(hf_path, split=split, streaming=True)
            ds = ds.shuffle(seed=epoch_seed, buffer_size=10_000)
            doc_count = 0
            for row in ds:
                if filter_fn is not None and not filter_fn(row):
                    continue
                text = row.get(text_col)
                if not text or len(text) < min_doc_len:
                    continue
                doc_count += 1
                yield text
            error_count = 0
            print(f"[{hf_path}] epoch {epoch} done ({doc_count:,} docs)", flush=True)
        except Exception as e:
            error_count += 1
            print(f"[{hf_path}] stream error (epoch {epoch}, error #{error_count}): "
                  f"{type(e).__name__}: {str(e)[:160]}", flush=True)
            if error_count >= 20:
                print(f"[{hf_path}] too many errors, stopping", flush=True)
                return
            time.sleep(min(5 * error_count, 60))


def interleaved_mix(sources, rng):
    """Weighted interleave of looping doc streams.

    Each source loops indefinitely (epoch after epoch). The mixer picks
    sources by weight — smaller datasets naturally cycle more epochs to
    fill their weight budget. The caller stops when enough shards are written.
    """
    iters = []
    for hf_path, weight, split, text_col, filt, min_len in sources:
        it = stream_docs_looping(hf_path, split, text_col, filt, min_len)
        iters.append({"iter": it, "weight": weight, "name": hf_path, "alive": True})

    while any(e["alive"] for e in iters):
        alive = [e for e in iters if e["alive"]]
        total_w = sum(e["weight"] for e in alive)
        r = rng.random() * total_w
        acc = 0.0
        for entry in alive:
            acc += entry["weight"]
            if r <= acc:
                try:
                    yield next(entry["iter"])
                except StopIteration:
                    entry["alive"] = False
                    print(f"[{entry['name']}] exhausted (all epochs failed)", flush=True)
                break


def write_shard(out_dir, idx, docs):
    path = os.path.join(out_dir, SHARD_FMT.format(idx))
    tmp = path + ".tmp"
    table = pa.Table.from_pydict({"text": docs})
    pq.write_table(table, tmp, row_group_size=ROW_GROUP_SIZE, compression="zstd")
    os.replace(tmp, path)
    print(f"  wrote {path} ({len(docs):,} docs)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Prepare Finnish pretraining data (ClimbMix-fi)")
    ap.add_argument("--num-shards", type=int, default=30,
                    help="Number of shards to create (30=d12 smoke, 80=d20 ratio6, 346=d20 Chinchilla)")
    ap.add_argument("--out-dir", type=str,
                    default=os.path.join(
                        os.environ.get("SUOMICHAT_BASE_DIR",
                                       os.path.expanduser("~/.cache/suomichat")),
                        "base_data_climbmix"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    todo = [i for i in range(args.num_shards)
            if not os.path.exists(os.path.join(args.out_dir, SHARD_FMT.format(i)))]
    if not todo:
        print(f"All {args.num_shards} shards already present in {args.out_dir}")
        return

    est_tokens = args.num_shards * CHARS_PER_SHARD / 4.83
    print(f"ClimbMix-fi: {len(todo)} new shards (of {args.num_shards}) → ~{est_tokens/1e9:.1f}B tokens")
    print(f"Output: {args.out_dir}")
    mix_desc = ", ".join(s[0].split("/")[-1] + f" {s[1]*100:.0f}%" for s in SOURCES)
    print(f"Mix: {mix_desc}")
    print()

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
                print("Warning: all source streams exhausted", flush=True)
                break
            current_docs.append(doc)
            current_chars += len(doc)
            if current_chars >= CHARS_PER_SHARD:
                if shard_cursor in remaining:
                    write_shard(args.out_dir, shard_cursor, current_docs)
                    remaining.remove(shard_cursor)
                    elapsed = time.time() - start
                    done = args.num_shards - len(remaining)
                    rate = elapsed / done if done else 0
                    eta = rate * len(remaining) / 60
                    print(f"  [{done}/{args.num_shards}] elapsed {elapsed/60:.1f} min, "
                          f"ETA {eta:.0f} min", flush=True)
                shard_cursor += 1
                current_docs = []
                current_chars = 0
                while shard_cursor < args.num_shards and shard_cursor not in remaining:
                    shard_cursor += 1
        if current_docs and current_chars >= CHARS_PER_SHARD * 0.5 and shard_cursor in remaining:
            write_shard(args.out_dir, shard_cursor, current_docs)
    except KeyboardInterrupt:
        print("Interrupted; partial shard discarded.", flush=True)
        sys.exit(130)

    elapsed = time.time() - start
    print(f"Done in {elapsed/60:.1f} min. Shards in {args.out_dir}")


if __name__ == "__main__":
    main()
