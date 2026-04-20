"""
lm-evaluation-harness model wrapper for suomichat checkpoints.

Two entry points:

1. As a registered lm_eval model — loads from a checkpoint path:
       python -m lm_eval --model suomichat \
           --model_args source=sft,device=cuda \
           --tasks goldenswag_ht_fi_cf_fbv2_p0 \
           --batch_size 1

2. As an in-process call — wraps an already-loaded model:
       from scripts.lm_eval_wrapper import run_finbench
       results = run_finbench(model, tokenizer, meta, tasks=[...])

Requires lm-evaluation-harness installed (LumiOpen fork has Finnish tasks).
"""
import sys
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from typing import List, Optional, Tuple
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.api.instance import Instance


def _ddp_rank_world():
    """Return (rank, world_size). Works pre-DDP-init as (0, 1)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


# Default Finnish CORE suite — kept here so both run_finbench.py CLI and
# the in-process training-loop call agree on what "FIN-CORE" means.
#
# MCF (multiple-choice format): the model is shown labeled options
# A/B/C/D and scored on which letter has highest loglikelihood. This is
# more robust than CF (continuation form) for small models — a 286M
# model on CF general_knowledge scored 0.13 (sub-random) because it
# picked fluent short words like "kaksi" over correct longer answers.
# MCF eliminates that fluency bias because every choice is a single
# letter token. See diagnostic results in program.md (2026-04-20).
FINNISH_CORE_TASKS = [
    "goldenswag_ht_fi_mcf_fbv2_p0",
    "scandisent_fi_mcf_fbv2_p0",
    "sib200_fi_mcf_fbv2_p0",
    "belebele_fin_mcf_fbv2_p0",
    "finbench_general_knowledge_mcf_fbv2_p0",
    "finbench_analogies_mcf_fbv2_p0",
    # Note: cause_and_effect has no MCF variant; arc_challenge_fi_mcf
    # was tried but its dataset config "ARC-C-fi-HT" is not available
    # in the default TurkuNLP/FIN-bench split. Keeping at 6 tasks for now.
]

# CF variants kept for reference / opt-in comparison runs.
FINNISH_CORE_TASKS_CF = [
    "goldenswag_ht_fi_cf_fbv2_p0",
    "scandisent_fi_cf_fbv2_p0",
    "sib200_fi_cf_fbv2_p0",
    "belebele_fin_cf_fbv2_p0",
    "FIN-bench_general_knowledge_multiple_choice",
    "FIN-bench_analogies_multiple_choice",
    "FIN-bench_cause_and_effect_multiple_choice",
]


@register_model("suomichat")
class NanochatLM(TemplateLM):
    """lm-eval model adapter for suomichat. Either loads from a checkpoint
    (CLI path) or wraps an already-loaded model (training-loop path)."""

    def __init__(
        self,
        source: Optional[str] = None,
        device: str = "cuda",
        model_tag: Optional[str] = None,
        step: Optional[int] = None,
        batch_size: int = 1,
        max_gen_toks: int = 256,
        # Pre-loaded model path: pass these instead of `source`.
        model=None,
        tokenizer=None,
        meta=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        from suomichat.common import COMPUTE_DTYPE

        if model is not None:
            assert tokenizer is not None and meta is not None, \
                "When passing model directly, tokenizer and meta are also required"
            self._device = next(model.parameters()).device
            self._model = model
            self._tokenizer = tokenizer
            self._meta = meta
        else:
            assert source is not None, "Either source=... or model=... must be provided"
            from suomichat.checkpoint_manager import load_model
            self._device = torch.device(device)
            self._model, self._tokenizer, self._meta = load_model(
                source,
                device=self._device,
                phase="eval",
                model_tag=model_tag,
                step=int(step) if step is not None else None,
            )

        self._model.eval()
        self._dtype = COMPUTE_DTYPE
        self._max_length = self._meta["model_config"]["sequence_len"]
        self._batch_size = int(batch_size)
        self._max_gen_toks = int(max_gen_toks)
        self._bos = self._tokenizer.get_bos_token_id()
        self._vocab_size = self._tokenizer.get_vocab_size()

    @property
    def eot_token_id(self):
        return self._bos

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        return self._tokenizer.encode(string)

    def tok_decode(self, tokens, **kwargs) -> str:
        return self._tokenizer.decode(tokens)

    def _score_one(self, ctx_toks: List[int], cont_toks: List[int]) -> Tuple[float, bool]:
        """Forward + loglikelihood computation for one (ctx, cont) pair."""
        # Degenerate empty request: no context and no continuation.
        if not ctx_toks and not cont_toks:
            return 0.0, True

        full_toks = ctx_toks + cont_toks
        if len(full_toks) > self._max_length:
            full_toks = full_toks[-self._max_length:]
            cont_len = min(len(cont_toks), self._max_length - 1)
        else:
            cont_len = len(cont_toks)

        inp = torch.tensor([full_toks], dtype=torch.long, device=self._device)
        with torch.no_grad():
            logits = self._model(inp)
        logits = logits.float()
        log_probs = F.log_softmax(logits, dim=-1)

        start = len(full_toks) - cont_len - 1
        end = len(full_toks) - 1

        cont_log_probs = 0.0
        is_greedy = True
        for pos in range(start, end):
            target = full_toks[pos + 1]
            lp = log_probs[0, pos, target].item()
            cont_log_probs += lp
            greedy_tok = log_probs[0, pos].argmax().item()
            if greedy_tok != target:
                is_greedy = False
        return cont_log_probs, is_greedy

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        """Rank-sharded loglikelihood. Every rank sees the full request list
        (lm_eval calls this once per rank), but each rank only *computes* the
        slice requests[rank::world_size]. Results are aggregated via all_reduce.

        Matches the pattern nanochat used for CORE eval: data-parallel
        evaluation reuses the N-GPU compute already allocated for training.
        Saves ~$5/run on 8×H100 vs. the previous rank-0-only approach.
        """
        rank, world_size = _ddp_rank_world()
        n = len(requests)

        # Per-request result buffers — only filled on the owner rank
        lp_buf = torch.zeros(n, dtype=torch.float64, device=self._device)
        greedy_buf = torch.zeros(n, dtype=torch.float64, device=self._device)

        # tqdm only on rank 0; others show a silent bar
        iter_range = range(rank, n, world_size)
        show_tqdm = (rank == 0) and not disable_tqdm
        for i in tqdm(iter_range, disable=not show_tqdm,
                      total=len(iter_range), desc=f"loglikelihood (rank {rank})"):
            (_ctx_str, _cont_str), ctx_toks, cont_toks = requests[i]
            lp, ig = self._score_one(ctx_toks, cont_toks)
            lp_buf[i] = lp
            greedy_buf[i] = 1.0 if ig else 0.0

        # Aggregate across ranks. Each slot was written by exactly one rank;
        # sum reduction equals the full per-request result vector.
        if world_size > 1:
            dist.all_reduce(lp_buf, op=dist.ReduceOp.SUM)
            dist.all_reduce(greedy_buf, op=dist.ReduceOp.SUM)

        return [(lp_buf[i].item(), greedy_buf[i].item() > 0.5) for i in range(n)]

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        """Same rank-sharded pattern as _loglikelihood_tokens. FIN-CORE
        doesn't use this path but keep it consistent for correctness."""
        rank, world_size = _ddp_rank_world()
        n = len(requests)
        lp_buf = torch.zeros(n, dtype=torch.float64, device=self._device)

        iter_range = range(rank, n, world_size)
        show_tqdm = (rank == 0) and not disable_tqdm
        for i in tqdm(iter_range, disable=not show_tqdm,
                      total=len(iter_range), desc=f"loglikelihood_rolling (rank {rank})"):
            req = requests[i]
            string = req.args[0]
            tokens = self.tok_encode(string)
            if not tokens:
                continue
            total_lp = 0.0
            for start in range(0, len(tokens), self._max_length):
                chunk = tokens[start : start + self._max_length]
                inp = torch.tensor([chunk], dtype=torch.long, device=self._device)
                with torch.no_grad():
                    logits = self._model(inp).float()
                log_probs = F.log_softmax(logits, dim=-1)
                for pos in range(len(chunk) - 1):
                    target = chunk[pos + 1]
                    total_lp += log_probs[0, pos, target].item()
            lp_buf[i] = total_lp

        if world_size > 1:
            dist.all_reduce(lp_buf, op=dist.ReduceOp.SUM)

        return [lp_buf[i].item() for i in range(n)]

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        results = []
        for req in tqdm(requests, disable=disable_tqdm, desc="generate_until"):
            context = req.args[0]
            until = req.args[1].get("until", [])
            max_gen = req.args[1].get("max_gen_toks", self._max_gen_toks)

            tokens = self.tok_encode(context)
            if len(tokens) > self._max_length - 1:
                tokens = tokens[-(self._max_length - 1):]

            generated = []
            cur = list(tokens)
            for _ in range(max_gen):
                inp = torch.tensor([cur[-self._max_length:]], dtype=torch.long, device=self._device)
                with torch.no_grad():
                    logits = self._model(inp).float()
                next_tok = logits[0, -1].argmax().item()
                if next_tok == self._bos:
                    break
                generated.append(next_tok)
                cur.append(next_tok)
                # check stop sequences
                gen_text = self.tok_decode(generated)
                if any(s in gen_text for s in until):
                    break

            results.append(self.tok_decode(generated))
        return results


def run_finbench(model, tokenizer, meta, tasks=None, limit=None, batch_size=1):
    """Run FIN-bench evaluation in-process on an already-loaded model.

    Must be called on ALL ranks when DDP-initialized — requests are sharded
    inside NanochatLM._loglikelihood_tokens via rank/world_size, and results
    aggregated via all_reduce. Returning the aggregated dict is safe for
    every rank to receive; callers typically only log on rank 0.

    Returns a dict:
        {
            "per_task": {task_name: accuracy, ...},
            "composite": float,  # mean across tasks
            "raw": <full lm_eval results dict>,
        }
    """
    from lm_eval.evaluator import simple_evaluate

    tasks = tasks or FINNISH_CORE_TASKS
    lm = NanochatLM(model=model, tokenizer=tokenizer, meta=meta, batch_size=batch_size)
    raw = simple_evaluate(model=lm, tasks=tasks, limit=limit)

    per_task = {}
    for task, metrics in (raw.get("results") or {}).items():
        acc = metrics.get("acc,none") or metrics.get("acc_norm,none") or 0.0
        per_task[task] = float(acc)
    composite = sum(per_task.values()) / len(per_task) if per_task else 0.0
    return {"per_task": per_task, "composite": composite, "raw": raw}
