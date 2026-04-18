"""
Finnish pretraining dataset — parquet shards of Finnish web text.

Data is prepared by `scripts/prepare_data.py` which downloads from HuggingFace
(FineWeb2 Finnish, Wikipedia Finnish, Reddit Finnish) and repackages into
250M-char shards in the same format as nanochat's ClimbMix.

If shards don't exist yet, run:
    python -m scripts.prepare_data --num-shards 30   # d12 smoke
    python -m scripts.prepare_data --num-shards 80   # d20 run
    python -m scripts.prepare_data --num-shards 200  # d24+ run
"""

import os
import argparse
import pyarrow.parquet as pq

from suomichat.common import get_base_dir

# Data lives in $SUOMICHAT_BASE_DIR/base_data_climbmix/
# (reuses the dir name so nanochat-compatible code paths work)
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data_climbmix")


def list_parquet_files(data_dir=None, warn_on_legacy=False):
    """Looks into a data dir and returns full paths to all parquet files."""
    data_dir = DATA_DIR if data_dir is None else data_dir
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Run: python -m scripts.prepare_data --num-shards 30")
        return []
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths


def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups.
    - split: "train" or "val". The last parquet file is val.
    - start/step: for DDP sharding (start=rank, step=world_size)
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    if not parquet_paths:
        raise FileNotFoundError("No data shards found. Run: python -m scripts.prepare_data --num-shards 30")
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Finnish pretraining data status")
    args = parser.parse_args()
    paths = list_parquet_files()
    print(f"Found {len(paths)} shards in {DATA_DIR}")
    if paths:
        total_size = sum(os.path.getsize(p) for p in paths)
        print(f"Total size: {total_size / 1e9:.2f} GB")
