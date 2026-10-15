"""Thin client for the SubQ Decisions API with a budget guard and a JSONL cache.

Pricing (docs.subq.ai/using-the-api/models, Oct 2026): subq-snap-preview
charges $0.05 per 1M input tokens; output is free. Every call here is
cached by a hash of its body, so re-running an experiment never re-spends.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://api.subq.ai/v1"
MODEL = "subq-snap-preview"
USD_PER_INPUT_TOKEN = 0.05 / 1_000_000


class BudgetExceeded(RuntimeError):
    pass


class SubQClient:
    def __init__(
        self,
        api_key: str | None = None,
        cache_path: str | Path = "results/cache.jsonl",
        budget_usd: float | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("SUBQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set SUBQ_API_KEY (see .env.example)")
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

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _key(body: dict) -> str:
        canon = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    def decide(
        self,
        state: Any,
        questions: dict[str, dict],
        images: list[dict] | None = None,
        max_retries: int = 4,
    ) -> dict:
        """POST /v1/systemone. Returns the parsed JSON response (answers + usage)."""
        body: dict[str, Any] = {"model": MODEL, "state": state, "questions": questions}
        if images:
            body["images"] = images
        key = self._key(body)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        if self.spent_usd >= self.budget_usd:
            raise BudgetExceeded(
                f"spent ${self.spent_usd:.4f} >= budget ${self.budget_usd:.2f}; "
                "raise SUBQ_BUDGET_USD to continue"
            )

        delay = 1.0
        for attempt in range(max_retries + 1):
            r = requests.post(
                f"{BASE_URL}/systemone",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=body,
                timeout=self.timeout,
            )
            if r.status_code in (429, 503, 500, 502, 504) and attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            if not r.ok:
                try:
                    detail = r.json().get("detail", r.text)
                except ValueError:
                    detail = r.text
                raise RuntimeError(f"SubQ {r.status_code}: {detail}")
            resp = r.json()
            break

        toks = int(resp.get("usage", {}).get("input_tokens", 0))
        self.input_tokens += toks
        self.spent_usd += toks * USD_PER_INPUT_TOKEN
        self.calls += 1
        self._cache[key] = resp
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "body": body, "response": resp}, ensure_ascii=False) + "\n")
        return resp

    def summary(self) -> str:
        return (
            f"API calls: {self.calls}  cache hits: {self.cache_hits}  "
            f"input tokens: {self.input_tokens}  spent: ${self.spent_usd:.5f}"
        )


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
