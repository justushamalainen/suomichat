#!/usr/bin/env python
"""
Run FIN-bench-v2 evaluation on a suomichat checkpoint using lm-evaluation-harness.

Usage:
    cd /home/janitor/llm-training/suomichat
    source .venv/bin/activate
    export SUOMICHAT_BASE_DIR=/home/janitor/llm-training/data-fi
    python /home/janitor/llm-training/run_finbench.py [--source sft] [--limit 0] [--tasks TASK1,TASK2]
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

# Register suomichat model BEFORE importing lm_eval evaluator
import lm_eval_wrapper  # noqa: F401 — registers @register_model("suomichat")

import lm_eval
from lm_eval.evaluator import simple_evaluate
from lm_eval.utils import make_table

FINNISH_CORE_TASKS = [
    "goldenswag_ht_fi_cf_fbv2_p0",
    "scandisent_fi_cf_fbv2_p0",
    "sib200_fi_cf_fbv2_p0",
    "belebele_fin_cf_fbv2_p0",
    "FIN-bench_general_knowledge_multiple_choice",
    "FIN-bench_analogies_multiple_choice",
    "FIN-bench_cause_and_effect_multiple_choice",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="sft", choices=["base", "sft", "rl"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="0 = full eval")
    ap.add_argument("--tasks", type=str, default=None,
                    help="Comma-separated task names. Default: Finnish CORE suite")
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    tasks = args.tasks.split(",") if args.tasks else FINNISH_CORE_TASKS
    limit = args.limit if args.limit > 0 else None

    print(f"Running FIN-bench evaluation: {len(tasks)} tasks, source={args.source}, limit={limit}")
    print(f"Tasks: {tasks}")

    results = simple_evaluate(
        model="suomichat",
        model_args=f"source={args.source},device={args.device},batch_size={args.batch_size}",
        tasks=tasks,
        limit=limit,
    )

    print("\n" + "=" * 80)
    print(make_table(results))
    print("=" * 80)

    # Print summary
    if "results" in results:
        scores = []
        for task, metrics in results["results"].items():
            acc = metrics.get("acc,none") or metrics.get("acc_norm,none") or 0.0
            print(f"  {task}: {acc:.4f}")
            scores.append(acc)
        if scores:
            composite = sum(scores) / len(scores)
            print(f"\n  FIN-CORE composite: {composite:.4f}")


if __name__ == "__main__":
    main()
