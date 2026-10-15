"""Experiment 2 — How sensitive is the probability to how the question is worded?

Same state, same ground truth, four phrasings of the same noul question
(plain / formal / with criteria / without criteria). Reports the spread
(std, max-min) of P(yes) across phrasings per instance and whether
re-wording ever flips the decision. Mirrors the 公理措辞对照 idea: the
"axiom" is fixed, only the surface form moves.

Run:  python experiments/exp02_wording.py --n 30
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subq_lab.client import SubQClient, noul  # noqa: E402
from subq_lab.tasks import auction_instances, matching_instances  # noqa: E402
import random  # noqa: E402

PHRASINGS = {
    "auction": {
        "plain": lambda f: noul(f"Does bidder {f} win?"),
        "formal": lambda f: noul(
            f"Under the stated allocation rule, is bidder {f} allocated the item?"
        ),
        "with_criteria": lambda f: noul(
            f"Does bidder {f} win the item?",
            true=f"{f} submitted the highest bid",
            false=f"another bidder submitted a higher bid than {f}",
        ),
        "negated": lambda f: noul(f"Is it false that bidder {f} loses the auction?"),
    },
    "matching": {
        "plain": lambda _: noul("Is the proposed matching stable?"),
        "formal": lambda _: noul(
            "Does the proposed matching admit no blocking pair, i.e. is it pairwise stable?"
        ),
        "with_criteria": lambda _: noul(
            "Is the proposed matching stable?",
            true="no man and woman both prefer each other to their assigned partners",
            false="some man and woman both prefer each other to their assigned partners",
        ),
        "negated": lambda _: noul("Is it false that some pair blocks the proposed matching?"),
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/exp02")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    if args.mock:
        from subq_lab.mock import MockClient as Client
    else:
        Client = SubQClient
    client = Client()
    rng = random.Random(args.seed)

    items = []
    for inst in auction_instances(rng, args.n):
        focal = inst.questions["focal_wins"]["instructions"].split()[2]
        items.append(("auction", inst.state, focal, inst.truth["focal_wins"]))
    for inst in matching_instances(rng, args.n):
        items.append(("matching", inst.state, None, inst.truth["is_stable"]))

    rows = []
    for i, (task, state, focal, truth) in enumerate(items):
        ps = {}
        for name, build in PHRASINGS[task].items():
            resp = client.decide(state, {"q": build(focal)})
            ps[name] = float(resp["answers"]["q"]["noul"])
        vals = list(ps.values())
        rows.append(
            {
                "i": i,
                "task": task,
                "truth": truth,
                "p": ps,
                "std": statistics.pstdev(vals),
                "range": max(vals) - min(vals),
                "flips": len({v >= 0.5 for v in vals}) > 1,
                "all_correct": all((v >= 0.5) == bool(truth) for v in vals),
            }
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    print("\n=== Exp02 wording sensitivity ===")
    for task in PHRASINGS:
        sub = [r for r in rows if r["task"] == task]
        print(
            f"{task:10} n={len(sub):3d}  mean std={statistics.mean(r['std'] for r in sub):.3f}  "
            f"mean range={statistics.mean(r['range'] for r in sub):.3f}  "
            f"flip rate={sum(r['flips'] for r in sub)/len(sub):.2f}  "
            f"all-phrasings-correct={sum(r['all_correct'] for r in sub)/len(sub):.2f}"
        )
        for name in PHRASINGS[task]:
            acc = sum((r["p"][name] >= 0.5) == bool(r["truth"]) for r in sub) / len(sub)
            print(f"    {name:14} acc={acc:.2f}")
    print(client.summary())


if __name__ == "__main__":
    main()
