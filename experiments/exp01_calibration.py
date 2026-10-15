"""Experiment 1 — Is a System-1 decision model calibrated on mechanism-design ground truth?

For auctions, stable matching and a rule-based drone triage policy, ask typed
questions whose correct answers are computed in code, then score accuracy,
Brier, log-loss and ECE per task/question. Cost at n=40 per task: ~120 calls
of ~300 tokens = ~36K tokens = $0.002.

Run:  python experiments/exp01_calibration.py --n 40 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subq_lab.client import SubQClient  # noqa: E402
from subq_lab.metrics import report  # noqa: E402
from subq_lab.tasks import all_instances, is_correct, pairs_from_answer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="instances per task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/exp01")
    ap.add_argument("--mock", action="store_true", help="use the offline mock client")
    args = ap.parse_args()

    if args.mock:
        from subq_lab.mock import MockClient as Client
    else:
        Client = SubQClient
    client = Client()

    instances = all_instances(seed=args.seed, n_per_task=args.n)
    rows = []
    pairs: dict[str, list] = defaultdict(list)
    correct: dict[str, list] = defaultdict(list)

    for idx, inst in enumerate(instances):
        resp = client.decide(inst.state, inst.questions)
        for qname, q in inst.questions.items():
            ans = resp["answers"][qname]
            truth = inst.truth[qname]
            ok = is_correct(q, ans, truth)
            # Several blocking pairs can be correct; accept any of them.
            if qname == "blocking_pair" and inst.truth.get("all_blocking"):
                ok = ans.get("choice") in inst.truth["all_blocking"]
            key = f"{inst.task}/{qname}"
            pairs[key].extend(pairs_from_answer(q, ans, truth))
            correct[key].append(int(ok))
            rows.append(
                {
                    "i": idx,
                    "task": inst.task,
                    "question": qname,
                    "truth": truth,
                    "answer": ans,
                    "correct": ok,
                    "meta": inst.meta,
                }
            )
        if (idx + 1) % 20 == 0:
            print(f"[{idx + 1}/{len(instances)}] {client.summary()}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    summary = {}
    for key in sorted(pairs):
        rep = report(pairs[key])
        rep["top1_accuracy"] = sum(correct[key]) / len(correct[key])
        summary[key] = rep
    summary["_cost"] = client.summary()
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Exp01 calibration on mechanism-design ground truth ===")
    print(f"{'task/question':32} {'n':>4} {'top1':>6} {'brier':>7} {'logloss':>8} {'ece':>6}")
    for key, rep in summary.items():
        if key.startswith("_"):
            continue
        print(
            f"{key:32} {len(correct[key]):4d} {rep['top1_accuracy']:6.2f} "
            f"{rep['brier']:7.3f} {rep['log_loss']:8.3f} {rep['ece']:6.3f}"
        )
    print(client.summary())


if __name__ == "__main__":
    main()
