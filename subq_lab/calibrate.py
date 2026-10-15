"""Post-hoc temperature scaling for decision-model answers.

Laya's docs note its checkpoints ship over-confident and recommend refitting a
temperature per question type. This module fits one temperature T per
(backend, question type) on a calibration split by grid search on log-loss,
then rescales probabilities: noul via logit/T, choice/score via p^(1/T)
renormalised.
"""
from __future__ import annotations

import math

from .tasks import score_probs

EPS = 1e-6
GRID = [round(0.25 * 1.1**k, 4) for k in range(50)]  # 0.25 .. ~26


def _dist(row: dict) -> tuple[list[float], int]:
    """Return (probabilities over outcomes, index of the true outcome)."""
    q, a, truth = row["q"], row["answer"], row["truth"]
    t = q["type"]
    if t == "noul":
        p = float(a["noul"])
        return [1 - p, p], int(bool(truth))
    if t == "choice":
        opts = list(q["criteria"])
        probs = a.get("probabilities") or {}
        dist = [float(probs.get(o, 0.0)) for o in opts]
        accepted = row.get("accepted") or [truth]
        idx = next((opts.index(x) for x in accepted if x in opts), 0)
        return dist, idx
    return score_probs(q, a), int(truth)


def rescale(dist: list[float], T: float) -> list[float]:
    logs = [math.log(max(p, EPS)) / T for p in dist]
    m = max(logs)
    ex = [math.exp(x - m) for x in logs]
    s = sum(ex)
    return [x / s for x in ex]


def nll(rows: list[dict], T: float) -> float:
    tot = 0.0
    for r in rows:
        d, y = _dist(r)
        tot += -math.log(max(rescale(d, T)[y], EPS))
    return tot / max(len(rows), 1)


def fit_temperature(rows: list[dict]) -> float:
    if not rows:
        return 1.0
    return min(GRID, key=lambda T: nll(rows, T))


def apply(row: dict, T: float) -> dict:
    """Return a copy of the row with its answer rescaled by T."""
    d, _ = _dist(row)
    d2 = rescale(d, T)
    q = row["q"]
    a = dict(row["answer"])
    if q["type"] == "noul":
        a["noul"] = d2[1]
    elif q["type"] == "choice":
        opts = list(q["criteria"])
        a["probabilities"] = dict(zip(opts, d2))
        a["choice"] = opts[max(range(len(d2)), key=d2.__getitem__)]
    else:
        a["probabilities"] = {str(i): p for i, p in enumerate(d2)}
    out = dict(row)
    out["answer"] = a
    return out
