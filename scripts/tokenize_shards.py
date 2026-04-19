"""
Pre-tokenize raw text shards into token shards for faster training.

Reads:  $SUOMICHAT_BASE_DIR/base_data_climbmix/shard_XXXXX.parquet  (text column)
Writes: $SUOMICHAT_BASE_DIR/base_data_tokenized/shard_XXXXX.parquet (tokens column)

The dataloader auto-detects tokenized shards and skips on-the-fly tokenization,
giving ~10% training speedup and enabling pre-computed data reuse across runs.

Usage:
    python -m scripts.tokenize_shards              # tokenize all shards
    python -m scripts.tokenize_shards --workers 8   # parallel (CPU-bound)
"""
import argparse
import os
import time
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

from suomichat.common import get_base_dir
from suomichat.tokenizer import get_tokenizer

ROW_GROUP_SIZE = 1024


def tokenize_one_shard(args_tuple):
    """Tokenize a single shard file. Designed for multiprocessing.Pool."""
    src_path, dst_path, tokenizer_dir = args_tuple

    if os.path.exists(dst_path):
        return f"skip {os.path.basename(dst_path)}"

    # Each worker loads its own tokenizer (not picklable across processes)
    from suomichat.tokenizer import RustBPETokenizer
    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
    bos = tokenizer.get_bos_token_id()

    pf = pq.ParquetFile(src_path)
    all_token_lists = []

    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx)
        texts = rg.column("text").to_pylist()
        # Batch tokenize with BOS prepended (matches dataloader behavior)
        token_lists = tokenizer.encode(texts, prepend=bos, num_threads=1)
        all_token_lists.extend(token_lists)

    # Write tokenized shard
    tmp = dst_path + ".tmp"
    table = pa.Table.from_pydict({"tokens": all_token_lists})
    pq.write_table(table, tmp, row_group_size=ROW_GROUP_SIZE, compression="zstd")
    os.replace(tmp, dst_path)

    n_tokens = sum(len(t) for t in all_token_lists)
    return f"wrote {os.path.basename(dst_path)} ({len(all_token_lists):,} docs, {n_tokens:,} tokens)"


def main():
    ap = argparse.ArgumentParser(description="Pre-tokenize text shards for faster training")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    base_dir = get_base_dir()
    src_dir = os.path.join(base_dir, "base_data_climbmix")
    dst_dir = os.path.join(base_dir, "base_data_tokenized")
    tokenizer_dir = os.path.join(base_dir, "tokenizer")

    if not os.path.exists(tokenizer_dir):
        print("Error: tokenizer not found. Run `python -m scripts.tok_train` first.")
        return

    os.makedirs(dst_dir, exist_ok=True)

    src_files = sorted(f for f in os.listdir(src_dir)
                       if f.endswith(".parquet") and not f.endswith(".tmp"))
    if not src_files:
        print(f"No shards found in {src_dir}")
        return

    tasks = []
    for fname in src_files:
        tasks.append((
            os.path.join(src_dir, fname),
            os.path.join(dst_dir, fname),
            tokenizer_dir,
        ))

    existing = sum(1 for _, dst, _ in tasks if os.path.exists(dst))
    todo = len(tasks) - existing
    print(f"Tokenizing {todo} shards ({existing} already done) using {args.workers} workers")
    print(f"Source: {src_dir}")
    print(f"Output: {dst_dir}")

    t0 = time.time()
    with Pool(processes=args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(tokenize_one_shard, tasks)):
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(tasks)}] {result} ({elapsed/60:.1f} min)")

    elapsed = time.time() - t0
    print(f"Done in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
