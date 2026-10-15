"""One client for any `/v1/systemone` decision endpoint (SubQ, local Laya, ...).

SubQ (`subq-snap-preview`) and Laya's `laya-serve` both expose the same wire
format: POST {state, questions{name: {type, instructions, criteria}}} and get
back typed answers with probabilities. Only the base URL, model id, key and
price differ, so they live in BACKENDS.

Every call is cached by backend + body hash, so re-running an experiment never
re-spends. Latency is recorded per call for the speed comparison.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import requests


def _backends() -> dict[str, dict]:
    return {
        "subq": {
            "base_url": os.environ.get("SUBQ_BASE_URL", "https://api.subq.ai/v1"),
            "model": os.environ.get("SUBQ_MODEL", "subq-snap-preview"),
            "key_env": "SUBQ_API_KEY",
            "key_required": True,
            # docs.subq.ai/using-the-api/models (Oct 2026): $0.05 / 1M input, output free
            "usd_per_input_token": 0.05 / 1_000_000,
        },
        "laya": {
            "base_url": os.environ.get("LAYA_BASE_URL", "http://127.0.0.1:8000/v1"),
            # laya-serve model ids: english | multilingual | typed-decisions
            "model": os.environ.get("LAYA_MODEL", "english"),
            "key_env": "LAYA_API_KEY",
            "key_required": False,
            "usd_per_input_token": 0.0,  # runs on your own GPU/CPU
        },
        # Same local server, the checkpoint fine-tuned on typed decisions (1,024-token context).
        "laya-typed": {
            "base_url": os.environ.get("LAYA_BASE_URL", "http://127.0.0.1:8000/v1"),
            "model": "typed-decisions",
            "key_env": "LAYA_API_KEY",
            "key_required": False,
            "usd_per_input_token": 0.0,
        },
    }


BACKENDS = _backends()


class BudgetExceeded(RuntimeError):
    pass


class DecisionClient:
    def __init__(
        self,
        backend: str = "subq",
        cache_path: str | Path = "results/cache.jsonl",
        budget_usd: float | None = None,
        timeout: float = 120.0,
    ) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; choose from {list(BACKENDS)}")
        self.backend = backend
        cfg = BACKENDS[backend]
        self.base_url = cfg["base_url"].rstrip("/")
        self.model = cfg["model"]
        self.price = cfg["usd_per_input_token"]
        self.api_key = os.environ.get(cfg["key_env"])
        if cfg["key_required"] and not self.api_key:
            raise RuntimeError(f"Set {cfg['key_env']} (see .env.example)")
        self.budget_usd = float(
            budget_usd if budget_usd is not None else os.environ.get("SUBQ_BUDGET_USD", "0.50")
        )
        self.timeout = timeout
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec["response"]
        self.spent_usd = 0.0
        self.calls = 0
        self.cache_hits = 0
        self.input_tokens = 0

    def _key(self, body: dict) -> str:
        canon = json.dumps(
            {"backend": self.backend, "base_url": self.base_url, "body": body},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    def decide(
        self,
        state: Any,
        questions: dict[str, dict],
        images: list[dict] | None = None,
        max_retries: int = 4,
    ) -> dict:
        """POST {base_url}/systemone. Returns the parsed response plus `_latency_ms`."""
        if self.backend.startswith("laya"):
            questions = _fill_null_options(questions)
        body: dict[str, Any] = {"model": self.model, "state": state, "questions": questions}
        if images:
            body["images"] = images
        key = self._key(body)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        if self.price and self.spent_usd >= self.budget_usd:
            raise BudgetExceeded(
                f"spent ${self.spent_usd:.4f} >= budget ${self.budget_usd:.2f}; "
                "raise SUBQ_BUDGET_USD to continue"
            )
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        delay = 1.0
        for attempt in range(max_retries + 1):
            t0 = time.perf_counter()
            r = requests.post(f"{self.base_url}/systemone", headers=headers, json=body, timeout=self.timeout)
            latency_ms = (time.perf_counter() - t0) * 1000
            if r.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            if not r.ok:
                try:
                    detail = r.json().get("detail", r.text)
                except ValueError:
                    detail = r.text
                raise RuntimeError(f"{self.backend} {r.status_code}: {detail}")
            resp = r.json()
            break

        resp["_latency_ms"] = round(latency_ms, 1)
        toks = int((resp.get("usage") or {}).get("input_tokens", 0) or 0)
        self.input_tokens += toks
        self.spent_usd += toks * self.price
        self.calls += 1
        self._cache[key] = resp
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps({"key": key, "backend": self.backend, "body": body, "response": resp}, ensure_ascii=False)
                + "\n"
            )
        return resp

    def summary(self) -> str:
        return (
            f"[{self.backend}:{self.model}] API calls: {self.calls}  cache hits: {self.cache_hits}  "
            f"input tokens: {self.input_tokens}  spent: ${self.spent_usd:.5f}"
        )


def _fill_null_options(questions: dict[str, dict]) -> dict[str, dict]:
    """Laya encodes option text, so give `null` options their own name as a description."""
    out = {}
    for name, q in questions.items():
        q = dict(q)
        if q["type"] == "choice" and isinstance(q.get("criteria"), dict):
            q["criteria"] = {k: (v if v is not None else k) for k, v in q["criteria"].items()}
        out[name] = q
    return out


# Backwards-compatible name used by exp00.
def SubQClient(**kw) -> DecisionClient:  # noqa: N802
    return DecisionClient(backend="subq", **kw)


def make_client(backend: str):
    if backend == "mock":
        from .mock import MockClient

        return MockClient()
    if backend.startswith("mock-"):
        from .mock import MockClient

        return MockClient(name=backend)
    return DecisionClient(backend=backend)


# ---------------------------------------------------------------- question DSL
def noul(instructions: str, true: str | None = None, false: str | None = None) -> dict:
    q: dict[str, Any] = {"type": "noul", "instructions": instructions}
    crit = {}
    if true is not None:
        crit["true"] = true
    if false is not None:
        crit["false"] = false
    q["criteria"] = crit
    return q


def choice(instructions: str, options: dict[str, str | None]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": options}


def score(instructions: str, levels: list[str]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": levels}
