"""Mechanism-design micro-tasks with programmatically computed ground truth.

Each generator returns a list of Instance(state, questions, truth) where
`truth` maps question name -> correct answer ("A"/"B"/..., True/False, or
a level index). Ground truth costs nothing, so the model's probabilities can
be scored at scale.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any

from .client import choice, noul, score


@dataclass
class Instance:
    task: str
    state: Any
    questions: dict[str, dict]
    truth: dict[str, Any]
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------- auctions
def _bids(rng: random.Random, n: int) -> dict[str, int]:
    names = ["A", "B", "C", "D", "E"][:n]
    vals = rng.sample(range(5, 60), n)  # distinct so the winner is unique
    return dict(zip(names, vals))


def auction_instances(rng: random.Random, n: int = 40) -> list[Instance]:
    out = []
    for i in range(n):
        k = rng.choice([3, 4])
        bids = _bids(rng, k)
        fmt = rng.choice(["first-price", "second-price"])
        ranked = sorted(bids.items(), key=lambda kv: -kv[1])
        winner, top = ranked[0]
        second = ranked[1][1]
        price = top if fmt == "first-price" else second
        focal = rng.choice(list(bids))
        state = {
            "mechanism": f"sealed-bid {fmt} auction, single indivisible item, highest bid wins",
            "bids": bids,
        }
        opts = {name: None for name in bids}
        price_opts = {f"p{v}": f"pays {v}" for v in sorted(bids.values())}
        questions = {
            "winner": choice("Which bidder wins the item?", opts),
            "price": choice("What price does the winner pay under this mechanism?", price_opts),
            "focal_wins": noul(
                f"Does bidder {focal} win the item?",
                true=f"{focal} has the highest bid",
                false=f"someone other than {focal} has the highest bid",
            ),
            "truthful_dominant": noul(
                "In this auction format, is bidding one's true value a weakly dominant strategy?",
                true="yes (second-price / Vickrey)",
                false="no (first-price: shading bids below value can be profitable)",
            ),
        }
        truth = {
            "winner": winner,
            "price": f"p{price}",
            "focal_wins": focal == winner,
            "truthful_dominant": fmt == "second-price",
        }
        out.append(Instance("auction", state, questions, truth, {"format": fmt, "k": k}))
    return out


# ----------------------------------------------------------- stable matching
def _prefs(rng: random.Random, n: int):
    men = [f"m{i+1}" for i in range(n)]
    women = [f"w{i+1}" for i in range(n)]
    mp = {m: rng.sample(women, n) for m in men}
    wp = {w: rng.sample(men, n) for w in women}
    return men, women, mp, wp


def _gale_shapley(men, women, mp, wp):
    free = list(men)
    nxt = {m: 0 for m in men}
    engaged: dict[str, str] = {}  # woman -> man
    rank = {w: {m: r for r, m in enumerate(wp[w])} for w in women}
    while free:
        m = free.pop(0)
        w = mp[m][nxt[m]]
        nxt[m] += 1
        if w not in engaged:
            engaged[w] = m
        elif rank[w][m] < rank[w][engaged[w]]:
            free.append(engaged[w])
            engaged[w] = m
        else:
            free.append(m)
    return {m: w for w, m in engaged.items()}


def _blocking_pairs(match: dict[str, str], mp, wp):
    inv = {w: m for m, w in match.items()}
    mr = {m: {w: r for r, w in enumerate(mp[m])} for m in mp}
    wr = {w: {m: r for r, m in enumerate(wp[w])} for w in wp}
    blocks = []
    for m in mp:
        for w in wp:
            if match[m] == w:
                continue
            if mr[m][w] < mr[m][match[m]] and wr[w][m] < wr[w][inv[w]]:
                blocks.append((m, w))
    return blocks


def matching_instances(rng: random.Random, n: int = 40) -> list[Instance]:
    out = []
    for i in range(n):
        size = 3
        men, women, mp, wp = _prefs(rng, size)
        stable = _gale_shapley(men, women, mp, wp)
        # Half the time show the stable matching, half the time a random perturbation.
        if rng.random() < 0.5:
            shown = dict(stable)
        else:
            perm = rng.sample(women, size)
            shown = dict(zip(men, perm))
        blocks = _blocking_pairs(shown, mp, wp)
        is_stable = len(blocks) == 0
        state = {
            "setting": "one-to-one two-sided matching (men/women), each ranks all on the other side, most preferred first",
            "men_preferences": mp,
            "women_preferences": wp,
            "proposed_matching": shown,
        }
        pair_opts = {f"{m}-{w}": None for m in men for w in women if shown[m] != w}
        pair_opts["none"] = "the matching is stable; no pair blocks"
        questions = {
            "is_stable": noul(
                "Is the proposed matching stable?",
                true="no man and woman both prefer each other to their assigned partners",
                false="at least one blocking pair exists",
            ),
            "blocking_pair": choice("Which (man, woman) pair blocks the proposed matching?", pair_opts),
            "gs_partner_m1": choice(
                "Under the man-proposing deferred-acceptance (Gale-Shapley) algorithm, whom is m1 matched with?",
                {w: None for w in women},
            ),
            "n_blocking": score(
                "How many blocking pairs does the proposed matching have?",
                ["0", "1", "2", "3 or more"],
            ),
        }
        truth = {
            "is_stable": is_stable,
            "blocking_pair": f"{blocks[0][0]}-{blocks[0][1]}" if blocks else "none",
            "gs_partner_m1": stable["m1"],
            "n_blocking": min(len(blocks), 3),
            "all_blocking": [f"{m}-{w}" for m, w in blocks],
        }
        out.append(Instance("matching", state, questions, truth, {"shown_is_gs": shown == stable}))
    return out


# ---------------------------------------------------------- drone triage
def drone_instances(rng: random.Random, n: int = 40) -> list[Instance]:
    """Rule-based landing-point congestion triage: ground truth from a fixed rule."""
    out = []
    for i in range(n):
        capacity = rng.choice([2, 3, 4])
        queued = rng.randint(0, 8)
        wind = rng.choice([3, 6, 9, 12, 15])  # m/s
        battery = rng.choice([12, 20, 35, 60])  # %
        load = queued / capacity
        # Rule: divert if load > 1.5 or wind >= 12 or (battery < 15 and load > 1)
        divert = load > 1.5 or wind >= 12 or (battery < 15 and load > 1.0)
        level = 0 if load <= 0.5 else 1 if load <= 1.0 else 2 if load <= 1.5 else 3
        state = {
            "landing_point": {"pads": capacity, "drones_queued": queued, "wind_m_s": wind},
            "inbound_drone": {"battery_pct": battery},
            "policy": "divert if queued/pads > 1.5, or wind >= 12 m/s, or (battery < 15% and queued/pads > 1)",
        }
        questions = {
            "divert": noul(
                "Under the stated policy, should the inbound drone be diverted to another landing point?",
                true="the policy's divert condition holds",
                false="the drone may continue to this landing point",
            ),
            "congestion": score(
                "How congested is the landing point (queued drones per pad)?",
                ["<=0.5 per pad", "0.5-1.0 per pad", "1.0-1.5 per pad", ">1.5 per pad"],
            ),
        }
        out.append(Instance("drone", state, questions, {"divert": divert, "congestion": level}))
    return out


GENERATORS = {
    "auction": auction_instances,
    "matching": matching_instances,
    "drone": drone_instances,
}


def all_instances(seed: int = 0, n_per_task: int = 40) -> list[Instance]:
    rng = random.Random(seed)
    inst = []
    for name, gen in GENERATORS.items():
        inst.extend(gen(rng, n_per_task))
    return inst


# --------------------------------------------------------------- scoring
def pairs_from_answer(q: dict, answer: dict, truth: Any) -> list[tuple[float, int]]:
    """Convert one answer into (p, y) pairs for calibration metrics."""
    t = answer.get("type", q["type"])
    if t == "noul":
        p = float(answer["noul"])
        return [(p, int(bool(truth)))]
    if t == "choice":
        probs = answer.get("probabilities", {})
        return [(float(p), int(opt == truth)) for opt, p in probs.items()]
    if t == "score":
        probs = answer.get("probabilities", {})
        return [(float(p), int(int(lvl) == int(truth))) for lvl, p in probs.items()]
    return []


def is_correct(q: dict, answer: dict, truth: Any) -> bool:
    t = answer.get("type", q["type"])
    if t == "noul":
        return (float(answer["noul"]) >= 0.5) == bool(truth)
    if t == "choice":
        return answer.get("choice") == truth
    if t == "score":
        return int(round(float(answer["score"]))) == int(truth)
    return False
