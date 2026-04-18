"""
lm-evaluation-harness model wrapper for suomichat checkpoints.

Usage:
    export SUOMICHAT_BASE_DIR=/home/janitor/llm-training/data-fi
    cd /home/janitor/llm-training/suomichat

    # Run FIN-bench-v2 tasks:
    python -m lm_eval --model suomichat \
        --model_args source=sft,device=cuda \
        --tasks finbench_v2_goldenswag,finbench_v2_arc_c \
        --batch_size 1

Requires: suomichat on sys.path, lm-evaluation-harness installed.
"""
import sys
import os
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.api.instance import Instance


@register_model("suomichat")
class NanochatLM(TemplateLM):

    def __init__(
        self,
        source: str = "sft",
        device: str = "cuda",
        model_tag: Optional[str] = None,
        step: Optional[int] = None,
        batch_size: int = 1,
        max_gen_toks: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        from suomichat.checkpoint_manager import load_model
        from suomichat.common import COMPUTE_DTYPE

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

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        results = []
        for (ctx_str, cont_str), ctx_toks, cont_toks in tqdm(
            requests, disable=disable_tqdm, desc="loglikelihood"
        ):
            full_toks = ctx_toks + cont_toks
            if len(full_toks) > self._max_length:
                full_toks = full_toks[-self._max_length:]
                cont_len = len(cont_toks)
            else:
                cont_len = len(cont_toks)

            inp = torch.tensor([full_toks], dtype=torch.long, device=self._device)
            with torch.no_grad():
                logits = self._model(inp)  # (1, T, V)
            logits = logits.float()
            log_probs = F.log_softmax(logits, dim=-1)

            # continuation starts at position len(ctx_toks) in the full sequence
            # but we need the logits at position t to predict token t+1
            start = len(full_toks) - cont_len - 1
            end = len(full_toks) - 1

            cont_log_probs = 0.0
            is_greedy = True
            for i, pos in enumerate(range(start, end)):
                target = full_toks[pos + 1]
                lp = log_probs[0, pos, target].item()
                cont_log_probs += lp
                greedy_tok = log_probs[0, pos].argmax().item()
                if greedy_tok != target:
                    is_greedy = False

            results.append((cont_log_probs, is_greedy))
        return results

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        results = []
        for req in tqdm(requests, disable=disable_tqdm, desc="loglikelihood_rolling"):
            string = req.args[0]
            tokens = self.tok_encode(string)
            if not tokens:
                results.append(0.0)
                continue

            total_lp = 0.0
            # process in chunks of max_length
            for start in range(0, len(tokens), self._max_length):
                chunk = tokens[start : start + self._max_length]
                inp = torch.tensor([chunk], dtype=torch.long, device=self._device)
                with torch.no_grad():
                    logits = self._model(inp).float()
                log_probs = F.log_softmax(logits, dim=-1)
                for pos in range(len(chunk) - 1):
                    target = chunk[pos + 1]
                    total_lp += log_probs[0, pos, target].item()

            results.append(total_lp)
        return results

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
