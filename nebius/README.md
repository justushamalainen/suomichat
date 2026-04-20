# SuomiChat on Nebius

Two paths, pick one:

- **SkyPilot (recommended)** — declarative YAML, named clusters,
  autostop, cross-cloud. Use `nebius/suomichat.yaml`.
- **Raw SSH** — provision a VM, SSH in, run `nebius/train.sh`. More
  control, less automation.

## Why Nebius?

- **EU-based** (Dutch parent); relevant if data residency matters
- **Direct H100/H200 access**; no per-function cold-start overhead
- **Cheaper per GPU-hour** than Modal on some SKUs

The training code is GPU-provider-agnostic — anything that gives you a
Linux box with 1-8× NVIDIA H100/H200 works. This guide is specifically
for Nebius, but the same `nebius/train.sh` runs unchanged on Lambda
Labs, CoreWeave, RunPod, or any bare-metal node.

## Cost reference (Nebius on-demand pricing, 2026-04)

| Config | Cost/hour | d12 | d20 | d24 | d26 |
|---|---|---|---|---|---|
| 1× H100 80GB | ~$2.40 | ~1h / $2 | ~10h / $24 | — | — |
| 8× H100 80GB | ~$19.20 | — | ~1.5h / $29 | ~2.5h / $48 | ~3h / $58 |
| 8× H200 141GB | ~$24.00 | — | — | ~2.0h / $48 | ~2.5h / $60 |

(Spot instances are typically 30-50% cheaper if you accept preemption.)

Blackwell (B200/B300) is supported by `suomichat/hardware.py` (FP8
enabled, bs table tuned for 192-288 GB VRAM) but not the primary
target today — FA3 wheels are still Hopper-only, so SDPA fallback
costs ~30% on attention. Revisit when FA3 ships Blackwell builds.

## Path A: SkyPilot (recommended)

### Setup (one-time, on your laptop)

```bash
# Nebius CLI + credentials
curl -fsSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
nebius-setup.sh            # prompts for tenant/project

# SkyPilot with Nebius provider
pip install "skypilot-nightly[nebius]"
sky check                  # should show Nebius: ENABLED
```

### Create Object Storage buckets (one-time)

```bash
nebius storage bucket create suomichat-data
nebius storage bucket create suomichat-checkpoints
```

Buckets persist across cluster teardown, so pre-tokenized data survives.

### Launch training

From the suomichat repo root:

```bash
sky launch -c suomichat nebius/suomichat.yaml --env DEPTH=24
```

SkyPilot provisions a Nebius 8× H100 node, mounts your buckets at
`/workspace/data` and `/workspace/checkpoints`, runs the full pipeline.

### Re-run on the same cluster (skips setup)

```bash
sky exec suomichat nebius/suomichat.yaml --env DEPTH=26 --env WANDB_RUN=d26_run1
```

### Monitor

```bash
sky logs suomichat        # streaming logs
sky status                # cluster state + idle time
```

### Autostop on idle (cost control)

```bash
sky autostop suomichat -i 30    # shut down after 30 min idle
```

Cluster will stop when idle for 30 min; restart on next `sky launch` /
`sky exec`. Object Storage buckets persist.

### Tear down

```bash
sky down suomichat
```

## Path B: Raw SSH

### 1. Install the Nebius CLI on your laptop

```bash
curl -fsSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
nebius configure  # paste your credentials
```

### 2. Create a VM with GPUs

Via the web console (easier): pick a `gpu-h100-sxm` or `gpu-h200-sxm`
preset, 8 GPUs, Ubuntu 22.04, enough RAM (≥128 GB), enough disk (≥500
GB NVMe), and upload your SSH public key.

Via CLI:
```bash
nebius compute instance create \
    --parent-id "$NEBIUS_PROJECT_ID" \
    --name suomichat-train \
    --preset gpu-h100-sxm-8gpu \
    --boot-disk 'size=500gb,type=NETWORK_SSD_NON_REPLICATED,image-family=ubuntu-22-04' \
    --ssh-key-file ~/.ssh/id_rsa.pub
```

### 3. SSH in and clone the repo

```bash
nebius compute instance get-ssh-connection suomichat-train
# ssh -i ~/.ssh/id_rsa ubuntu@<public-ip>

# Inside the VM:
sudo apt update && sudo apt install -y git build-essential
git clone <suomichat-repo-url> suomichat && cd suomichat
```

### 4. Run training

One command, autodetects everything:

```bash
bash nebius/train.sh --depth 24
```

Flags (all optional):

- `--depth N` — model depth (default: 12). See `README.md` for size/tier guidance.
- `--num-shards N` — Finnish pretraining shards to download (default depends on depth: 30 for d12, 241 for d24/d26).
- `--sft-file PATH` — use a custom SFT jsonl (default: `$SUOMICHAT_BASE_DIR/sft_train.jsonl` if present, else FinnishAlpaca fallback).
- `--skip-data` — skip steps 1-4 (useful on re-runs where data is already cached).

Env vars:

- `SUOMICHAT_BASE_DIR` — data/checkpoint root (default: `~/.cache/suomichat`)
- `WANDB_RUN` — wandb run name (default: `dummy` = no wandb logging)
- `TOK_WORKERS` — pre-tokenization worker count (default: 8)
- `SUOMICHAT_BS` / `SUOMICHAT_NPROC` — override autoconfig

## Persisting data across runs

Nebius VMs are long-lived, but if you tear down and recreate:

1. **Upload once** to a Nebius Object Storage bucket before teardown:
    ```bash
    nebius storage cp -r ~/.cache/suomichat/base_data_tokenized \
        s3://<bucket>/suomichat-data/base_data_tokenized
    nebius storage cp -r ~/.cache/suomichat/tokenizer \
        s3://<bucket>/suomichat-data/tokenizer
    ```
2. **Download on new VM**:
    ```bash
    nebius storage cp -r s3://<bucket>/suomichat-data ~/.cache/suomichat
    bash nebius/train.sh --depth 24 --skip-data
    ```

This saves ~1h of tokenizer training + pre-tokenization on each new VM.

## Downloading results

After training:

```bash
# From your laptop (push from VM):
scp -r ubuntu@<vm-ip>:~/.cache/suomichat/chatsft_checkpoints ./my-checkpoints/

# Or upload to object storage and pull down:
nebius storage cp -r ~/.cache/suomichat/chatsft_checkpoints \
    s3://<bucket>/suomichat-runs/$(date +%Y%m%d-d$DEPTH)/
```

## Tearing down

Don't forget this step — Nebius VMs bill while running.

```bash
nebius compute instance delete suomichat-train
```

## Gotchas

- **Disk space**: d24 Chinchilla needs 241 shards × ~250 MB ≈ 60 GB for
  raw + another 60 GB for pre-tokenized. Provision ≥500 GB.
- **FA3**: installed automatically via `kernels` lib on Hopper (SM 9.0+)
  at first training step. Cold-start delay ~30 s.
- **FP8**: auto-enabled at depth ≥ 20 on Hopper. Set `--fp8` manually
  for explicit on; there's no `--no-fp8` flag — edit `suomichat/hardware.py`
  if you need to force off.
- **Multi-node**: `nebius/train.sh` assumes a single-node machine. For
  cross-node H100 pods, set up `torchrun --nnodes` manually; the script
  doesn't handle that today.
