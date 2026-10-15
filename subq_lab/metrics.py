"""Calibration metrics for probabilistic answers.

All functions take a list of (p, y) pairs: p = predicted probability that the
answer is "yes"/correct, y = 1 if the ground truth is yes/correct else 0.
"""
from __future__ import annotations

import math
from collections import defaultdict


def brier(pairs: list[tuple[float, int]]) -> float:
    return sum((p - y) ** 2 for p, y in pairs) / max(len(pairs), 1)


def log_loss(pairs: list[tuple[float, int]], eps: float = 1e-6) -> float:
    s = 0.0
    for p, y in pairs:
        p = min(max(p, eps), 1 - eps)
        s += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return s / max(len(pairs), 1)


def accuracy(pairs: list[tuple[float, int]], threshold: float = 0.5) -> float:
    return sum(int((p >= threshold) == bool(y)) for p, y in pairs) / max(len(pairs), 1)


def reliability(pairs: list[tuple[float, int]], bins: int = 10) -> list[dict]:
    """Per-bin mean confidence vs empirical frequency (for a reliability diagram)."""
    buckets: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for p, y in pairs:
        b = min(int(p * bins), bins - 1)
        buckets[b].append((p, y))
    rows = []
    for b in range(bins):
        items = buckets.get(b, [])
        if not items:
            continue
        rows.append(
            {
                "bin": f"[{b / bins:.1f},{(b + 1) / bins:.1f})",
                "n": len(items),
                "mean_conf": sum(p for p, _ in items) / len(items),
                "empirical": sum(y for _, y in items) / len(items),
            }
        )
    return rows


def ece(pairs: list[tuple[float, int]], bins: int = 10) -> float:
    """Expected calibration error."""
    rows = reliability(pairs, bins)
    n = max(len(pairs), 1)
    return sum(r["n"] / n * abs(r["mean_conf"] - r["empirical"]) for r in rows)


def report(pairs: list[tuple[float, int]], bins: int = 10) -> dict:
    return {
        "n": len(pairs),
        "accuracy": accuracy(pairs),
        "brier": brier(pairs),
        "log_loss": log_loss(pairs),
        "ece": ece(pairs, bins),
        "reliability": reliability(pairs, bins),
    }
