"""
Finnish SFT task — loads instruction data for Finnish chat fine-tuning.

Primary source: `datacrunch/finnish_alpaca` (guaranteed Finnish, Alpaca schema).
Secondary (optional): `LumiOpen/poro2-instruction-collection` filtered to Finnish
(skipped if unavailable or schema differs; gracefully degrades).
"""

import torch.distributed as dist
from datasets import load_dataset
from tasks.common import Task


def _finnish_ratio(text: str) -> float:
    if not text:
        return 0.0
    chars = [c for c in text if c.isalpha()]
    if not chars:
        return 0.0
    fi = sum(1 for c in chars if c in "äöÄÖåÅ")
    return fi / len(chars)


def _load_dataset_rank0_gated(*args, **kwargs):
    """Rank 0 triggers the HF download; other ranks wait on a barrier
    and then read from the shared cache.

    Without this, 8 ranks on 8-GPU runs race to download the same dataset
    in parallel — 8× HF API traffic + potential FileLock contention in
    HF's download code. With the gate, only rank 0 hits the network on
    cold cache; the others get instant-cache hits after the barrier.

    On single-GPU runs (DDP not initialized), this is just load_dataset.
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        if rank == 0:
            result = load_dataset(*args, **kwargs)
            dist.barrier()
            return result
        else:
            dist.barrier()
            return load_dataset(*args, **kwargs)
    return load_dataset(*args, **kwargs)


class FinnishAlpaca(Task):
    """
    datacrunch/finnish_alpaca — Alpaca-style Finnish instruction dataset.
    Schema: {instruction, input, output}.
    """
    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        self.ds = _load_dataset_rank0_gated(
            "datacrunch/finnish_alpaca", split=split,
        ).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        instr = row.get("instruction", "") or ""
        inp = row.get("input", "") or ""
        out = row.get("output", "") or ""
        user = instr if not inp else f"{instr}\n\n{inp}"
        return {
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": out},
            ]
        }


class Poro2InstructFi(Task):
    """
    LumiOpen/poro2-instruction-collection — mixed EN/FI instruction data.
    Filters rows where the assistant answer looks Finnish (ä/ö density > 0.5%).
    If the dataset or schema is not available, constructor raises and caller
    can skip.
    """
    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        self.ds = _load_dataset_rank0_gated(
            "LumiOpen/poro2-instruction-collection", split=split,
        ).shuffle(seed=42)
        # Pre-filter indices to Finnish-looking rows (one pass, cheap-ish)
        self.indices = []
        for i, row in enumerate(self.ds):
            msgs = row.get("messages") or []
            if not msgs:
                continue
            content = " ".join(m.get("content", "") for m in msgs if m.get("role") == "assistant")
            if _finnish_ratio(content) >= 0.005:
                self.indices.append(i)
        self.length = len(self.indices)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[self.indices[index]]
        msgs = row["messages"]
        # Normalize: ensure user/assistant alternation, string content
        clean = []
        for m in msgs:
            role = m.get("role")
            content = m.get("content")
            if role in ("user", "assistant", "system") and isinstance(content, str):
                clean.append({"role": role, "content": content})
        return {"messages": clean}
