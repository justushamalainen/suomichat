#!/usr/bin/env python
"""
Run FIN-bench-v2 evaluation on a suomichat checkpoint using lm-evaluation-harness.

Usage:
    cd /path/to/suomichat
    source .venv/bin/activate
    python -m scripts.run_finbench [--source sft] [--limit 0] [--tasks TASK1,TASK2]
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Importing the wrapper has side effects: it registers @register_model("suomichat")
# AND exports FINNISH_CORE_TASKS — both must come from the same place to keep the
# default task list consistent between this CLI and the in-process call.
from scripts.lm_eval_wrapper import FINNISH_CORE_TASKS  # noqa: F401

from lm_eval.evaluator import simple_evaluate
from lm_eval.utils import make_table


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
