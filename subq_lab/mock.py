"""Offline stand-in for SubQClient so the pipeline can be tested for $0.

Returns well-formed but uninformative answers (uniform / 0.5). Use --mock
on any experiment to check the plumbing before spending real credits.
"""
from __future__ import annotations

import random


class MockClient:
    def __init__(self, *_, **__):
        self.calls = 0
        self.rng = random.Random(0)

    def decide(self, state, questions, images=None):
        self.calls += 1
        answers = {}
        for name, q in questions.items():
            t = q["type"]
            if t == "noul":
                answers[name] = {"type": "noul", "noul": round(self.rng.random(), 3)}
            elif t == "choice":
                opts = list(q["criteria"])
                w = [self.rng.random() for _ in opts]
                s = sum(w)
                probs = {o: x / s for o, x in zip(opts, w)}
                best = max(probs, key=probs.get)
                answers[name] = {"type": "choice", "choice": best, "probabilities": probs, "confidence": probs[best]}
            else:
                lv = q["criteria"]
                w = [self.rng.random() for _ in lv]
                s = sum(w)
                probs = {str(i): x / s for i, x in enumerate(w)}
                sc = sum(i * p for i, p in enumerate(probs.values()))
                answers[name] = {
                    "type": "score",
                    "score": sc,
                    "legend": {str(i): l for i, l in enumerate(lv)},
                    "probabilities": probs,
                    "confidence": max(probs.values()),
                }
        return {"model": "mock", "answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0}}

    def summary(self) -> str:
        return f"MOCK calls: {self.calls}  spent: $0"
