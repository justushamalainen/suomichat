"""
Hardware autodetect + training-config recommendation.

User picks `--depth`. This module derives everything else:

    device_batch_size — largest power of 2 expected to fit VRAM at this depth
    fp8               — only on Hopper+ (SM 9.0+) and only for d>=24
    max_seq_len       — locked to 2048 (per NOTES.md: lower causes SFT NaN)
    nproc             — number of visible GPUs (override with SUOMICHAT_NPROC)

Usage from Python:

    from suomichat.hardware import detect_hardware, recommend_config
    hw = detect_hardware()
    cfg = recommend_config(depth=12, hw=hw)
    # cfg = {"device_batch_size": 16, "fp8": False, "max_seq_len": 2048, "nproc": 1, ...}

CLI (used by runs/suomichat.sh and modal/train.py):

    python -m suomichat.hardware --depth 12
    # prints "device_batch_size=16 fp8=0 max_seq_len=2048 nproc=1"
"""
import os
import sys
import argparse


def detect_hardware():
    """Return a dict describing the visible compute."""
    info = {
        "gpu_count": 0,
        "gpu_name": None,
        "vram_gb": 0,
        "sm_major": 0,
        "sm_minor": 0,
        "has_fa3": False,   # Flash Attention 3 Hopper-only wheel (SM 9.0 exactly)
        "has_fp8": False,   # FP8 tensor cores: Hopper (SM 9.0) and Blackwell (SM 10+)
        "is_hopper": False,
        "is_blackwell": False,
        "is_ada": False,
    }
    try:
        import torch
    except ImportError:
        return info
    if not torch.cuda.is_available():
        return info
    info["gpu_count"] = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0)
    info["gpu_name"] = props.name
    info["vram_gb"] = props.total_memory / (1024 ** 3)
    info["sm_major"] = props.major
    info["sm_minor"] = props.minor
    info["is_hopper"] = (props.major == 9)
    info["is_blackwell"] = (props.major >= 10)
    info["is_ada"] = (props.major == 8 and props.minor == 9)
    # FA3 today: varunneal/flash-attention-3 wheels are Hopper-only.
    # Blackwell will fall back to PyTorch SDPA until FA3 gets a Blackwell build.
    info["has_fa3"] = info["is_hopper"]
    # FP8 tensor cores: Hopper and Blackwell both have them.
    info["has_fp8"] = info["is_hopper"] or info["is_blackwell"]
    return info


# Empirically validated batch sizes per (depth, has_fa3, vram_gb) bucket.
# Memory model: without FA3, attention materializes an explicit O(seq²) mask
# so activation memory roughly doubles vs Hopper. This table is conservative —
# it leaves headroom for activations + grad buffers + Muon optimizer state.
#
# Numbers are device_batch_size (per-GPU). seq_len is fixed at 2048.
#
# Format: depth_bucket -> {(fa3, vram_gb_bucket): bs}
# vram_gb_bucket is one of "lt40", "40to60", "60to100", "ge100", "ge200".
# "ge200" = Blackwell B200/B300 (192-288 GB) where VRAM is abundant enough
# to offset FA3's absence (SDPA's O(seq²) mask still fits comfortably).
_BS_TABLE = {
    # depth 6 — tiny, fits anywhere with big batch
    6:  {(True, "lt40"): 32, (True, "40to60"): 64, (True, "60to100"): 64, (True, "ge100"): 64, (True, "ge200"): 64,
         (False, "lt40"): 32, (False, "40to60"): 32, (False, "60to100"): 32, (False, "ge100"): 32, (False, "ge200"): 64},
    # depth 12 — ~17GB at bs=16 on H100 with FA3
    12: {(True, "lt40"): 8,  (True, "40to60"): 16, (True, "60to100"): 32, (True, "ge100"): 32, (True, "ge200"): 64,
         (False, "lt40"): 4,  (False, "40to60"): 16, (False, "60to100"): 24, (False, "ge100"): 24, (False, "ge200"): 48},
    # depth 16 — ~30GB at bs=16 on H100
    16: {(True, "lt40"): 4,  (True, "40to60"): 8,  (True, "60to100"): 16, (True, "ge100"): 32, (True, "ge200"): 48,
         (False, "lt40"): 2,  (False, "40to60"): 8,  (False, "60to100"): 16, (False, "ge100"): 16, (False, "ge200"): 32},
    # depth 20 — ~45GB at bs=16 on H100
    20: {(True, "lt40"): 2,  (True, "40to60"): 4,  (True, "60to100"): 16, (True, "ge100"): 16, (True, "ge200"): 32,
         (False, "lt40"): 1,  (False, "40to60"): 4,  (False, "60to100"): 8,  (False, "ge100"): 16, (False, "ge200"): 32},
    # depth 24 — needs FP8 on 80GB H100; can't fit on Ada at any bs.
    # bs=32 confirmed working on 80GB H100 with FP8 per NOTES.md:20-22;
    # bs=64 OOMs at peak 78/80 GB. On B300 288GB the FA3-less SDPA path
    # still fits because VRAM is so abundant.
    24: {(True, "lt40"): 1,  (True, "40to60"): 2,  (True, "60to100"): 32, (True, "ge100"): 32, (True, "ge200"): 48,
         (False, "lt40"): 0,  (False, "40to60"): 0,  (False, "60to100"): 0,  (False, "ge100"): 0,  (False, "ge200"): 16},
    # depth 26+ — H100 (80GB) tight; B300 (288GB) is roomy even without FA3
    26: {(True, "lt40"): 0,  (True, "40to60"): 0,  (True, "60to100"): 16, (True, "ge100"): 32, (True, "ge200"): 48,
         (False, "lt40"): 0,  (False, "40to60"): 0,  (False, "60to100"): 0,  (False, "ge100"): 0,  (False, "ge200"): 16},
}


def _vram_bucket(vram_gb):
    if vram_gb < 40:
        return "lt40"
    if vram_gb < 60:
        return "40to60"
    if vram_gb < 100:
        return "60to100"
    if vram_gb < 200:
        return "ge100"
    return "ge200"  # Blackwell B200 (192 GB) or B300 (288 GB)


def _nearest_depth(depth):
    """Snap to the nearest tabulated depth bucket."""
    keys = sorted(_BS_TABLE.keys())
    for k in keys:
        if depth <= k:
            return k
    return keys[-1]


def recommend_config(depth, hw=None):
    """Return a dict of recommended training flags for the given depth + hw."""
    if hw is None:
        hw = detect_hardware()

    # Batch size from table
    if hw["gpu_count"] > 0:
        bucket = _vram_bucket(hw["vram_gb"])
        bs = _BS_TABLE[_nearest_depth(depth)][(hw["has_fa3"], bucket)]
    else:
        bs = 1  # CPU/no-GPU: smoke-test only

    if bs == 0:
        # Tabulated as "won't fit"; fall back to bs=1 + a loud warning
        print(f"WARNING: depth={depth} likely won't fit on {hw['gpu_name']} ({hw['vram_gb']:.0f}GB). Forcing bs=1.", file=sys.stderr)
        bs = 1

    # FP8: Hopper+ AND deep enough for the throughput win to outweigh the
    # numerical overhead. Per dev/LOG.md history, FP8 at d<=12 is net zero or
    # slightly negative (grouped_mm dispatch overhead eats the speedup). At
    # d>=20 the speedup is typically 10-40% depending on batch size and
    # attention cost; at d>=24 it's closer to 2× on matmul-heavy layers.
    # Users can override via --fp8 (force on) or --no-fp8 isn't wired today
    # — edit this line or skip the hardware module if needed.
    fp8 = hw["has_fp8"] and depth >= 20

    # nproc: use all visible GPUs unless overridden
    nproc_env = os.environ.get("SUOMICHAT_NPROC")
    nproc = int(nproc_env) if nproc_env else max(1, hw["gpu_count"])

    # Allow user to override bs via env (escape hatch for OOM debugging)
    bs_env = os.environ.get("SUOMICHAT_BS")
    if bs_env:
        bs = int(bs_env)

    return {
        "device_batch_size": bs,
        "fp8": fp8,
        "max_seq_len": 2048,           # locked: SFT inheritance NaN issue
        "target_param_data_ratio": 8,  # speed-optimized; bump for production
        "nproc": nproc,
        "_hw_summary": f"{hw['gpu_count']}× {hw['gpu_name'] or 'CPU'} ({hw['vram_gb']:.0f}GB, SM {hw['sm_major']}.{hw['sm_minor']}, FA3={hw['has_fa3']}, FP8={hw['has_fp8']})",
    }


def _emit_shell(cfg):
    """Print config as KEY=VALUE lines for `eval` in bash."""
    print(f"BS={cfg['device_batch_size']}")
    print(f"FP8={'1' if cfg['fp8'] else '0'}")
    print(f"MAX_SEQ_LEN={cfg['max_seq_len']}")
    print(f"NPROC={cfg['nproc']}")
    print(f"RATIO={cfg['target_param_data_ratio']}")
    print(f"HW_SUMMARY=\"{cfg['_hw_summary']}\"")


def main():
    ap = argparse.ArgumentParser(description="SuomiChat hardware autodetect + recommended training config.")
    ap.add_argument("--depth", type=int, required=True, help="model depth (12, 20, 24, ...)")
    ap.add_argument("--shell", action="store_true", help="emit KEY=VALUE lines for eval in bash")
    args = ap.parse_args()
    cfg = recommend_config(args.depth)
    if args.shell:
        _emit_shell(cfg)
    else:
        print(f"Hardware: {cfg['_hw_summary']}")
        for k, v in cfg.items():
            if not k.startswith("_"):
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
