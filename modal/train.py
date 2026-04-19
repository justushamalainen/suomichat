"""
Run SuomiChat training on Modal.com with H100 GPUs.

Setup (once):
    pip install modal
    modal setup
    # Upload data (done separately via modal volume put)

Train:
    python modal/train.py train                        # 1x H100, d12, ~2.5h, ~$8
    python modal/train.py train --depth 26 --gpus 8    # 8x H100, d26, ~3h, ~$76

Download results:
    python modal/train.py download

Resources: Only Volumes persist. Container + GPU are ephemeral — released
immediately after training. No orphaned resources.
"""
import sys
import os
import subprocess

import modal

# --- Modal resources ---
# Only two persistent resources: data volume and checkpoint volume.
# Everything else (container, GPU) is ephemeral and auto-deleted.
app = modal.App("suomichat")

data_vol = modal.Volume.from_name("suomichat-data", create_if_missing=True)
checkpoint_vol = modal.Volume.from_name("suomichat-checkpoints", create_if_missing=True)

DATA_PATH = "/data"
CHECKPOINT_PATH = "/checkpoints"

# Bake local suomichat code into the container image.
# Updated on every deploy — no git repo needed.
SUOMICHAT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("curl", "build-essential")
    .pip_install(
        "torch==2.9.1",
        "datasets>=4.0.0",
        "pyarrow",
        "wandb",
        "tiktoken",
        "rustbpe",
        "tokenizers",
        "fastapi",
        "uvicorn",
        "psutil",
        "kernels",
    )
    .env({
        "SUOMICHAT_BASE_DIR": DATA_PATH,
        "OMP_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/app/suomichat",
    })
    # add_local_dir must be LAST — Modal adds these at container startup, not build time
    .add_local_dir(SUOMICHAT_ROOT, remote_path="/app/suomichat",
                   ignore=[".venv", "__pycache__", ".git", "*.pyc", "uv.lock"])
)


@app.function(image=image, volumes={DATA_PATH: data_vol}, gpu="h100", timeout=300, cpu=2, memory=4096)
def diagnose() -> str:
    """Quick diagnostic to test environment. Returns results as string."""
    import subprocess
    lines = []
    lines.append(f"CWD: {os.getcwd()}")
    lines.append(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")
    lines.append(f"ls /app/suomichat: {sorted(os.listdir('/app/suomichat'))[:15]}")
    lines.append(f"ls /data: {os.listdir('/data')}")
    r = subprocess.run([sys.executable, "-c", "from suomichat.gpt import GPT; print('import OK')"],
                       capture_output=True, text=True, cwd="/app/suomichat")
    lines.append(f"Import stdout: {r.stdout.strip()}")
    lines.append(f"Import stderr: {r.stderr[-500:]}")
    r2 = subprocess.run([sys.executable, "-c", "import torch; print(f'CUDA: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"],
                        capture_output=True, text=True)
    lines.append(f"CUDA: {r2.stdout.strip()}")
    return "\n".join(lines)


def _setup_dirs():
    """Symlink checkpoint dirs to persistent volume."""
    os.chdir("/app/suomichat")

    # Debug: show what's available
    print(f"CWD: {os.getcwd()}", flush=True)
    print(f"ls /app/suomichat: {os.listdir('/app/suomichat')[:20]}", flush=True)
    print(f"ls {DATA_PATH}: {os.listdir(DATA_PATH)}", flush=True)

    for subdir in ["base_checkpoints", "chatsft_checkpoints"]:
        src = os.path.join(CHECKPOINT_PATH, subdir)
        dst = os.path.join(DATA_PATH, subdir)
        os.makedirs(src, exist_ok=True)
        if not os.path.exists(dst):
            os.symlink(src, dst)

    tok_dir = os.path.join(DATA_PATH, "base_data_tokenized")
    if not os.path.isdir(tok_dir):
        raise FileNotFoundError(f"No tokenized data at {tok_dir}. Upload data first.")
    n = len([f for f in os.listdir(tok_dir) if f.endswith(".parquet")])
    print(f"Found {n} tokenized shards")
    return n


def _train_cmd(depth, ratio, nproc=1):
    """Build pretrain command."""
    use_fp8 = depth >= 24
    # Batch size must make (bs × seq × nproc) divide total_batch_size evenly.
    # With seq=1024 and 8 GPUs: bs must be power-of-2 (16, 32, 64).
    # Start conservative, check VRAM, increase if headroom exists.
    # Match nanochat speedrun config: bs=16, seq=2048 (default), fp8 for d24+
    use_fp8 = depth >= 24
    bs = 16  # nanochat speedrun uses bs=16 on 8xH100

    if nproc > 1:
        prefix = ["torchrun", "--standalone", f"--nproc_per_node={nproc}", "-m", "scripts.base_train", "--"]
    else:
        prefix = [sys.executable, "-m", "scripts.base_train"]

    cmd = prefix + [
        f"--depth={depth}",
        f"--device-batch-size={bs}",
        f"--target-param-data-ratio={ratio}",
        "--run=dummy",
        "--core-metric-every=999999",
        "--sample-every=-1",
        "--save-every=-1",
    ]
    # seq=2048 is the default — don't override (we were using 1024 before)
    if use_fp8:
        cmd.append("--fp8")
    return cmd


def _sft_cmd(nproc=1):
    """Build SFT command."""
    if nproc > 1:
        prefix = ["torchrun", "--standalone", f"--nproc_per_node={nproc}", "-m", "scripts.chat_sft", "--"]
    else:
        prefix = [sys.executable, "-m", "scripts.chat_sft"]

    return prefix + [
        "--device-batch-size=16",
        "--max-seq-len=2048",
        "--load-optimizer=0",
        "--run=dummy",
    ]


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol, CHECKPOINT_PATH: checkpoint_vol},
    gpu="h100",
    timeout=7200,
    cpu=4,
    memory=8192,
)
def sft_only():
    """Run SFT only on existing pretrained checkpoint (1x H100)."""
    _setup_dirs()
    print(">>> Finnish SFT (346K rows)...", flush=True)
    rc = subprocess.call(_sft_cmd(nproc=1))
    if rc != 0:
        raise RuntimeError(f"SFT failed with exit code {rc}")
    checkpoint_vol.commit()
    data_vol.commit()
    print(">>> SFT Done!", flush=True)


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol, CHECKPOINT_PATH: checkpoint_vol},
    gpu="h100",
    timeout=86400,
    cpu=4,
    memory=8192,
)
def train_single(depth: int = 12, ratio: int = 8):
    """Train SuomiChat on a single H100."""
    _setup_dirs()
    subprocess.run([sys.executable, "scripts/gen_identity.py"], check=True)

    print(f">>> Pretraining d{depth} ratio={ratio}...", flush=True)
    cmd = _train_cmd(depth, ratio, nproc=1)
    print(f"CMD: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    checkpoint_vol.commit()

    print(">>> Finnish SFT...", flush=True)
    subprocess.run(_sft_cmd(nproc=1), check=True)
    checkpoint_vol.commit()
    data_vol.commit()
    print(">>> Done!")


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol, CHECKPOINT_PATH: checkpoint_vol},
    gpu="h100:8",
    timeout=86400,
    cpu=16,
    memory=32768,
)
def train_multi(depth: int = 20, ratio: int = 8):
    """Train SuomiChat on 8x H100 with torchrun."""
    _setup_dirs()
    subprocess.run([sys.executable, "scripts/gen_identity.py"], check=True)

    cmd = _train_cmd(depth, ratio, nproc=8)
    print(f"CMD: {' '.join(cmd)}", flush=True)
    # Stream output directly so Modal logs show training progress
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"Pretrain failed with exit code {rc}")
    checkpoint_vol.commit()

    print(">>> Finnish SFT on 8x H100...", flush=True)
    rc = subprocess.call(_sft_cmd(nproc=8))
    if rc != 0:
        raise RuntimeError(f"SFT failed with exit code {rc}")
    checkpoint_vol.commit()
    data_vol.commit()
    print(">>> Done!")


# --- CLI ---
def cli():
    import argparse
    parser = argparse.ArgumentParser(description="SuomiChat Modal training")
    sub = parser.add_subparsers(dest="command")

    tr = sub.add_parser("train", help="Train on Modal")
    tr.add_argument("--depth", type=int, default=12)
    tr.add_argument("--ratio", type=int, default=8)
    tr.add_argument("--gpus", type=int, default=1, choices=[1, 8])

    sub.add_parser("sft", help="Run SFT only on existing checkpoint (1x H100)")
    sub.add_parser("diagnose", help="Run diagnostics on Modal container")
    sub.add_parser("download", help="Download checkpoints from Modal")

    args = parser.parse_args()

    if args.command == "sft":
        with app.run():
            sft_only.remote()

    elif args.command == "diagnose":
        with app.run():
            result = diagnose.remote()
            print(result)

    elif args.command == "train":
        with app.run():
            if args.gpus == 1:
                train_single.remote(depth=args.depth, ratio=args.ratio)
            else:
                train_multi.remote(depth=args.depth, ratio=args.ratio)

    elif args.command == "download":
        print("Download checkpoints with:")
        print("  modal volume get suomichat-checkpoints / ./checkpoints/")

    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
