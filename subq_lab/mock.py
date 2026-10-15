"""Offline stand-in for DecisionClient so the pipeline can be tested for $0.

Answers are random but well-formed. `mock-sharp` returns over-confident
answers, `mock-soft` softer ones, so the figures have two distinct shapes to
draw. These numbers mean nothing; use them only to test the plumbing.
"""
from __future__ import annotations

import hashlib
import json
import random


class MockClient:
    def __init__(self, name: str = "mock"):
        self.backend = name
        self.calls = 0
        self.sharp = 4.0 if "sharp" in name else 1.0

    def _rng(self, state, questions):
        h = hashlib.sha256(json.dumps([self.backend, state, questions], sort_keys=True).encode()).hexdigest()
        return random.Random(int(h[:12], 16))

    def _dist(self, rng, n):
        w = [rng.random() ** self.sharp for _ in range(n)]
        s = sum(w) or 1.0
        return [x / s for x in w]

    def decide(self, state, questions, images=None):
        self.calls += 1
        rng = self._rng(state, questions)
        answers = {}
        for name, q in questions.items():
            t = q["type"]
            if t == "noul":
                p = rng.random()
                if self.sharp > 1:
                    p = rng.uniform(0.0, 0.08) if p < 0.5 else rng.uniform(0.92, 1.0)
                answers[name] = {"type": "noul", "noul": round(p, 3)}
            elif t == "choice":
                opts = list(q["criteria"])
                d = self._dist(rng, len(opts))
                probs = dict(zip(opts, d))
                best = max(probs, key=probs.get)
                answers[name] = {"type": "choice", "choice": best, "probabilities": probs, "confidence": probs[best]}
            else:
                lv = q["criteria"]
                d = self._dist(rng, len(lv))
                probs = {str(i): p for i, p in enumerate(d)}
                answers[name] = {
                    "type": "score",
                    "score": sum(i * p for i, p in enumerate(d)),
                    "legend": {str(i): l for i, l in enumerate(lv)},
                    "probabilities": probs,
                    "confidence": max(d),
                }
        lat = rng.uniform(20, 60) if "sharp" in self.backend else rng.uniform(150, 600)
        return {"model": self.backend, "answers": answers, "usage": {"input_tokens": 0}, "_latency_ms": lat}

    def summary(self) -> str:
        return f"[{self.backend}] MOCK calls: {self.calls}  spent: $0"
