"""
Run SuomiChat training on Modal.com with H100 GPUs.

Setup (once):
    pip install modal
    modal setup
    # Upload data (done separately via modal volume put)

Train (single dial: --depth; GPU count auto-picked from depth, override with --gpus):
    python modal/train.py train --depth 12             # 1x H100 (depth<22 default)
    python modal/train.py train --depth 24             # 8x H100 (depth>=22 default)
    python modal/train.py train --depth 20 --gpus 1    # force single GPU

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
    .apt_install("curl", "build-essential", "git")
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
        # End-of-training FIN-bench eval. Pin matches pyproject.toml's
        # `eval` extra so local and Modal runs use the same harness.
        "lm_eval @ git+https://github.com/LumiOpen/lm-evaluation-harness.git@6e0a60abb74cf098bc5511b25caf66566586f9f0",
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


def _resolve_config(depth, nproc):
    """Resolve the training config inside the Modal container by calling
    suomichat.hardware.recommend_config — this picks bs and fp8 from the
    container's actual VRAM detection, not a hardcoded table here."""
    sys.path.insert(0, "/app/suomichat")
    from suomichat.hardware import detect_hardware, recommend_config
    hw = detect_hardware()
    cfg = recommend_config(depth, hw=hw)
    # nproc is fixed by the @app.function decoration, not autoconfig
    cfg["nproc"] = nproc
    return cfg


def _train_cmd(depth, ratio, nproc=1):
    """Build pretrain command using hardware-derived bs and fp8."""
    cfg = _resolve_config(depth, nproc)
    print(f"  config: bs={cfg['device_batch_size']} fp8={cfg['fp8']} max_seq={cfg['max_seq_len']} ratio={ratio}", flush=True)

    if nproc > 1:
        prefix = ["torchrun", "--standalone", f"--nproc_per_node={nproc}", "-m", "scripts.base_train", "--"]
    else:
        prefix = [sys.executable, "-m", "scripts.base_train"]

    cmd = prefix + [
        f"--depth={depth}",
        f"--device-batch-size={cfg['device_batch_size']}",
        f"--target-param-data-ratio={ratio}",
        f"--max-seq-len={cfg['max_seq_len']}",
        "--run=dummy",
        "--sample-every=-1",
        "--save-every=-1",
    ]
    if cfg["fp8"]:
        cmd.append("--fp8")
    return cmd


def _sft_cmd(depth, nproc=1):
    """Build SFT command, also using hardware-derived bs."""
    cfg = _resolve_config(depth, nproc)
    if nproc > 1:
        prefix = ["torchrun", "--standalone", f"--nproc_per_node={nproc}", "-m", "scripts.chat_sft", "--"]
    else:
        prefix = [sys.executable, "-m", "scripts.chat_sft"]

    return prefix + [
        f"--device-batch-size={cfg['device_batch_size']}",
        f"--max-seq-len={cfg['max_seq_len']}",
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
def sft_only(depth: int = 12):
    """Run SFT only on existing pretrained checkpoint (1x H100). Pass --depth
    so hardware autoconfig knows what batch size to use."""
    _setup_dirs()
    print(">>> Finnish SFT...", flush=True)
    rc = subprocess.call(_sft_cmd(depth, nproc=1))
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
    subprocess.run(_sft_cmd(depth, nproc=1), check=True)
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
    rc = subprocess.call(_sft_cmd(depth, nproc=8))
    if rc != 0:
        raise RuntimeError(f"SFT failed with exit code {rc}")
    checkpoint_vol.commit()
    data_vol.commit()
    print(">>> Done!")


def _default_gpus_for_depth(depth):
    """Single H100 fits up to ~d20; deeper models want 8 H100s for speed
    (and d24+ benefits a lot from FP8 + extra VRAM headroom)."""
    return 8 if depth >= 22 else 1


# --- CLI ---
def cli():
    import argparse
    parser = argparse.ArgumentParser(description="SuomiChat Modal training")
    sub = parser.add_subparsers(dest="command")

    tr = sub.add_parser("train", help="Train on Modal")
    tr.add_argument("--depth", type=int, default=12)
    tr.add_argument("--ratio", type=int, default=8)
    tr.add_argument("--gpus", type=int, default=None, choices=[1, 8],
                    help="GPU count (default: 1 for d<22, 8 for d>=22)")

    s = sub.add_parser("sft", help="Run SFT only on existing checkpoint (1x H100)")
    s.add_argument("--depth", type=int, default=12,
                   help="model depth (used for batch-size autoconfig)")

    sub.add_parser("diagnose", help="Run diagnostics on Modal container")
    sub.add_parser("download", help="Download checkpoints from Modal")

    args = parser.parse_args()

    if args.command == "sft":
        with app.run():
            sft_only.remote(depth=args.depth)

    elif args.command == "diagnose":
        with app.run():
            result = diagnose.remote()
            print(result)

    elif args.command == "train":
        gpus = args.gpus if args.gpus else _default_gpus_for_depth(args.depth)
        print(f"Dispatching to {gpus}× H100 for depth={args.depth}")
        with app.run():
            if gpus == 1:
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
