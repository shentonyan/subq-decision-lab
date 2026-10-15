"""Build every figure and a REPORT.md from results/exp01 and results/exp02.

Reads whatever backends have results (subq, laya, ...), so it works with one
backend or several. Temperature scaling is fitted on even-numbered items and
evaluated on odd-numbered ones, so the "scaled" numbers are out-of-sample.

Run:  python analysis/make_figures.py            (writes figures/ and results/REPORT.md)
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import logging  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subq_lab.calibrate import _dist, apply, fit_temperature  # noqa: E402

# ------------------------------------------------------------------ style
# Categorical slots from a CVD-validated palette, assigned to backends in a fixed order.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FIXED = {"subq": 0, "laya": 1, "laya-typed": 2}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
TRUE_C, FALSE_C = "#1baf7a", "#6250d6"  # truth yes / no in the agreement scatter

plt.rcParams.update(
    {
        "font.family": ["Inter", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "legend.frameon": False,
        "legend.fontsize": 9,
    }
)

LABEL = {"subq": "SubQ snap-preview", "laya": "Laya english", "laya-typed": "Laya typed-decisions"}


def color_of(backend: str, order: list[str]) -> str:
    if backend in FIXED:
        return SLOTS[FIXED[backend]]
    extra = [b for b in order if b not in FIXED]
    return SLOTS[(2 + extra.index(backend)) % len(SLOTS)]


def label_of(b: str) -> str:
    return LABEL.get(b, b)


def load_rows(root: Path) -> dict[str, list[dict]]:
    out = {}
    if not root.exists():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        f = d / "rows.jsonl"
        if f.exists():
            rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
            if rows:
                out[d.name] = rows
    return out


# ------------------------------------------------------------------ metrics
def outcome_pairs(rows):
    pairs = []
    for r in rows:
        d, y = _dist(r)
        pairs += [(p, int(k == y)) for k, p in enumerate(d)]
    return pairs


def ece(pairs, bins=10):
    if not pairs:
        return float("nan")
    tot = 0.0
    for b in range(bins):
        sel = [(p, y) for p, y in pairs if min(int(p * bins), bins - 1) == b]
        if sel:
            tot += len(sel) / len(pairs) * abs(np.mean([p for p, _ in sel]) - np.mean([y for _, y in sel]))
    return tot


def brier_rows(rows):
    vals = []
    for r in rows:
        d, y = _dist(r)
        vals.append(sum((p - int(k == y)) ** 2 for k, p in enumerate(d)))
    return float(np.mean(vals)) if vals else float("nan")


def nll_rows(rows):
    vals = []
    for r in rows:
        d, y = _dist(r)
        vals.append(-math.log(max(d[y], 1e-6)))
    return float(np.mean(vals)) if vals else float("nan")


def correct_after(r):
    d, y = _dist(r)
    return int(max(range(len(d)), key=d.__getitem__) == y)


def temp_scale(rows):
    """Fit T per question type on even items, return (scaled odd rows, raw odd rows, temps)."""
    temps = {}
    cal = defaultdict(list)
    for r in rows:
        if r["i"] % 2 == 0:
            cal[r["q"]["type"]].append(r)
    for t, rs in cal.items():
        temps[t] = fit_temperature(rs)
    test = [r for r in rows if r["i"] % 2 == 1]
    scaled = [apply(r, temps.get(r["q"]["type"], 1.0)) for r in test]
    return scaled, test, temps


def reliability_curve(pairs, bins=10):
    xs, ys, ns = [], [], []
    for b in range(bins):
        sel = [(p, y) for p, y in pairs if min(int(p * bins), bins - 1) == b]
        if sel:
            xs.append(np.mean([p for p, _ in sel]))
            ys.append(np.mean([y for _, y in sel]))
            ns.append(len(sel))
    return xs, ys, ns


def save(fig, path: Path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ------------------------------------------------------------------ figures
def fig_accuracy(e1, order, out):
    keys = sorted({f"{r['task']}/{r['question']}" for rows in e1.values() for r in rows})
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(keys) + 1.2))
    y = np.arange(len(keys))
    acc = {}
    for b in order:
        by = defaultdict(list)
        for r in e1[b]:
            by[f"{r['task']}/{r['question']}"].append(int(r["correct"]))
        acc[b] = [np.mean(by[k]) if by[k] else np.nan for k in keys]
    # connector between backends per row
    if len(order) >= 2:
        for i in range(len(keys)):
            v = [acc[b][i] for b in order if not np.isnan(acc[b][i])]
            if len(v) >= 2:
                ax.plot([min(v), max(v)], [y[i], y[i]], color=GRID, lw=3, zorder=1, solid_capstyle="round")
    for b in order:
        ax.scatter(acc[b], y, s=64, color=color_of(b, order), label=label_of(b), zorder=3,
                   edgecolor=SURFACE, linewidth=2)
    # chance level per question (uniform guess over options)
    chance = {}
    for rows in e1.values():
        for r in rows:
            k = f"{r['task']}/{r['question']}"
            q = r["q"]
            n_opt = 2 if q["type"] == "noul" else len(q["criteria"])
            chance[k] = 1 / n_opt
    ax.scatter([chance.get(k, np.nan) for k in keys], y, marker="|", s=120, color=MUTED, zorder=2,
               label="chance (uniform guess)")
    ax.set_yticks(y, keys)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("top-1 accuracy")
    ax.grid(axis="y", visible=False)
    ax.set_title("Accuracy by question — same items for every model")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, borderaxespad=1.6)
    save(fig, out / "fig1_accuracy.png")


def fig_reliability(e1, order, out):
    fig, axes = plt.subplots(2, len(order), figsize=(4.0 * len(order), 6.2), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1.1]}, sharex=True)
    for j, b in enumerate(order):
        c = color_of(b, order)
        scaled, test, temps = temp_scale(e1[b])
        raw_p = outcome_pairs(test)
        sc_p = outcome_pairs(scaled)
        ax = axes[0, j]
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":", zorder=1)
        for pairs, ls, lab, alpha in [(raw_p, "-", "raw", 1.0), (sc_p, "--", "temperature-scaled", 0.75)]:
            xs, ys, ns = reliability_curve(pairs)
            ax.plot(xs, ys, ls=ls, color=c, lw=2, alpha=alpha, zorder=2, label=f"{lab}  ECE {ece(pairs):.3f}")
            ax.scatter(xs, ys, s=[12 + 140 * n / max(ns) for n in ns], color=c, alpha=alpha,
                       edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(label_of(b))
        if j == 0:
            ax.set_ylabel("observed frequency")
        ax.legend(loc="upper left", fontsize=8.5)
        tt = ", ".join(f"{k} T={v:.2f}" for k, v in sorted(temps.items()))
        ax.text(0.98, 0.03, tt, transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK2)
        # histogram of predicted probabilities (where the model puts its mass)
        hx = axes[1, j]
        hx.hist([p for p, _ in raw_p], bins=np.linspace(0, 1, 21), color=c, alpha=0.9,
                edgecolor=SURFACE, linewidth=2)
        hx.set_xlabel("predicted probability")
        if j == 0:
            hx.set_ylabel("count")
        hx.grid(axis="x", visible=False)
    fig.suptitle("Reliability on held-out items (dot size = items in bin)", x=0.02, ha="left",
                 fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout()
    save(fig, out / "fig2_reliability.png")


def calib_table(e1, order):
    tab = {}
    for b in order:
        scaled, test, temps = temp_scale(e1[b])
        tab[b] = {
            "raw": {"ECE": ece(outcome_pairs(test)), "Brier": brier_rows(test), "LogLoss": nll_rows(test),
                    "Acc": float(np.mean([correct_after(r) for r in test])) if test else float("nan")},
            "scaled": {"ECE": ece(outcome_pairs(scaled)), "Brier": brier_rows(scaled), "LogLoss": nll_rows(scaled),
                       "Acc": float(np.mean([correct_after(r) for r in scaled])) if scaled else float("nan")},
            "temps": temps,
            "n_test": len(test),
        }
    return tab


def fig_calib_bars(tab, order, out):
    metrics = ["ECE", "Brier", "LogLoss"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    w = 0.36
    for ax, m in zip(axes, metrics):
        x = np.arange(len(order))
        raw = [tab[b]["raw"][m] for b in order]
        sc = [tab[b]["scaled"][m] for b in order]
        for i, b in enumerate(order):
            c = color_of(b, order)
            ax.bar(x[i] - w / 2 - 0.01, raw[i], w, color=c, edgecolor=SURFACE, linewidth=2)
            ax.bar(x[i] + w / 2 + 0.01, sc[i], w, color=c, alpha=0.4, edgecolor=c, linewidth=1.2, hatch="////")
            ax.margins(y=0.12)
            ax.text(x[i] - w / 2, raw[i], f"{raw[i]:.3f}", ha="center", va="bottom", fontsize=8, color=INK2)
            ax.text(x[i] + w / 2, sc[i], f"{sc[i]:.3f}", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(x, [label_of(b) for b in order], fontsize=8.5)
        ax.set_title(f"{m}  (lower is better)")
        ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=MUTED, label="raw"),
                        Patch(facecolor="none", edgecolor=MUTED, hatch="////", label="temperature-scaled")],
               loc="upper right", ncol=2, fontsize=8.5)
    fig.suptitle("Calibration before / after per-type temperature scaling (held-out half)", x=0.02, ha="left",
                 fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout()
    save(fig, out / "fig3_calibration_bars.png")


def fig_confidence(e1, order, out):
    fig, axes = plt.subplots(1, len(order), figsize=(3.8 * len(order), 3.6), squeeze=False, sharey=True)
    rng = np.random.default_rng(0)
    for j, b in enumerate(order):
        ax = axes[0, j]
        c = color_of(b, order)
        groups = [[r["top_prob"] for r in e1[b] if r["correct"]], [r["top_prob"] for r in e1[b] if not r["correct"]]]
        for k, g in enumerate(groups):
            if not g:
                continue
            xs = k + rng.uniform(-0.18, 0.18, len(g))
            ax.scatter(xs, g, s=10, color=c if k == 0 else MUTED, alpha=0.45, linewidth=0)
            q1, med, q3 = np.percentile(g, [25, 50, 75])
            ax.plot([k - 0.28, k + 0.28], [med, med], color=INK, lw=2)
            ax.add_patch(plt.Rectangle((k - 0.28, q1), 0.56, q3 - q1, fill=False, edgecolor=INK2, lw=1))
            ax.text(k, 1.04, f"n={len(g)}  median {med:.2f}", ha="center", fontsize=8, color=INK2)
        ax.set_xticks([0, 1], ["correct", "wrong"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="x", visible=False)
        ax.set_title(label_of(b))
        if j == 0:
            ax.set_ylabel("probability on its own answer")
    fig.suptitle("Is the model less sure when it is wrong?", x=0.02, ha="left", fontsize=12,
                 fontweight="semibold", color=INK)
    fig.tight_layout()
    save(fig, out / "fig4_confidence_right_vs_wrong.png")


def fig_agreement(e1, order, out):
    if len(order) < 2:
        return
    a, b = order[0], order[1]
    ia = {(r["i"], r["question"]): r for r in e1[a] if r["q"]["type"] == "noul"}
    ib = {(r["i"], r["question"]): r for r in e1[b] if r["q"]["type"] == "noul"}
    common = sorted(set(ia) & set(ib))
    if not common:
        return
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for truth, c, lab in [(True, TRUE_C, "truth: yes"), (False, FALSE_C, "truth: no")]:
        pts = [(ia[k]["answer"]["noul"], ib[k]["answer"]["noul"]) for k in common if bool(ia[k]["truth"]) == truth]
        if pts:
            x, y = zip(*pts)
            ax.scatter(x, y, s=34, color=c, alpha=0.7, edgecolor=SURFACE, linewidth=1.5, label=f"{lab} (n={len(pts)})",
                       marker="o" if truth else "s")
    ax.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax.axvline(0.5, color=MUTED, lw=1, ls=":")
    agree = np.mean([(ia[k]["answer"]["noul"] >= 0.5) == (ib[k]["answer"]["noul"] >= 0.5) for k in common])
    r = np.corrcoef([ia[k]["answer"]["noul"] for k in common], [ib[k]["answer"]["noul"] for k in common])[0, 1]
    ax.set_xlabel(f"P(yes) — {label_of(a)}\n\nsame decision on {agree:.0%} of {len(common)} yes/no items · Pearson r = {r:.2f}")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_ylabel(f"P(yes) — {label_of(b)}")
    ax.set_title("Where the two models disagree (yes/no questions)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=2, borderaxespad=1.4)
    save(fig, out / "fig5_agreement.png")


def fig_wording(e2, order, out):
    if not e2:
        return
    tasks = sorted({r["task"] for rows in e2.values() for r in rows})
    phr = []
    for rows in e2.values():
        for r in rows:
            for k in r["p"]:
                if k not in phr:
                    phr.append(k)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
    # left: per-instance spread of P(yes) across phrasings
    ax = axes[0]
    rng = np.random.default_rng(1)
    pos, ticks = 0, []
    for t in tasks:
        for b in order:
            vals = [r["range"] for r in e2.get(b, []) if r["task"] == t]
            if not vals:
                continue
            c = color_of(b, order)
            ax.scatter(pos + rng.uniform(-0.2, 0.2, len(vals)), vals, s=12, color=c, alpha=0.5, linewidth=0)
            med = np.median(vals)
            ax.plot([pos - 0.3, pos + 0.3], [med, med], color=INK, lw=2)
            flips = np.mean([r["flips"] for r in e2[b] if r["task"] == t])
            ax.text(pos, 1.03, f"flip {flips:.0%}", ha="center", fontsize=8, color=INK2)
            ticks.append((pos, f"{t}\n{label_of(b)}"))
            pos += 1
        pos += 0.6
    ax.set_xticks([p for p, _ in ticks], [l for _, l in ticks], fontsize=7.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("max − min P(yes) across 4 phrasings")
    ax.grid(axis="x", visible=False)
    ax.set_title("Same question, different wording: how far does P(yes) move?")
    # right: accuracy heatmap phrasing x (task, backend)
    ax = axes[1]
    cols = [(t, b) for t in tasks for b in order if any(r["task"] == t for r in e2.get(b, []))]
    M = np.full((len(phr), len(cols)), np.nan)
    for j, (t, b) in enumerate(cols):
        sub = [r for r in e2[b] if r["task"] == t]
        for i, ph in enumerate(phr):
            got = [r for r in sub if ph in r["p"]]
            if got:
                M[i, j] = np.mean([(r["p"][ph] >= 0.5) == bool(r["truth"]) for r in got])
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if M[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(cols)), [f"{t}\n{label_of(b)}" for t, b in cols], fontsize=7.5)
    ax.set_yticks(range(len(phr)), phr)
    ax.grid(False)
    ax.set_title("Accuracy by phrasing")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).outline.set_visible(False)
    fig.tight_layout()
    save(fig, out / "fig6_wording.png")


def fig_latency(e1, e2, order, out):
    lat = {}
    for b in order:
        v = [r["latency_ms"] for r in e1.get(b, []) if r.get("latency_ms")]
        # one latency per request: dedupe rows that share a request
        seen, vals = set(), []
        for r in e1.get(b, []):
            if r["i"] not in seen and r.get("latency_ms"):
                seen.add(r["i"])
                vals.append(r["latency_ms"])
        if vals:
            lat[b] = vals
    if not lat:
        return
    fig, ax = plt.subplots(figsize=(6.4, 0.9 * len(lat) + 1.4))
    rng = np.random.default_rng(2)
    for i, (b, vals) in enumerate(lat.items()):
        c = color_of(b, order)
        ax.scatter(vals, i + rng.uniform(-0.18, 0.18, len(vals)), s=12, color=c, alpha=0.5, linewidth=0)
        med = np.median(vals)
        ax.plot([med, med], [i - 0.3, i + 0.3], color=INK, lw=2)
        ax.text(med, i + 0.36, f"median {med:.0f} ms", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_yticks(range(len(lat)), [label_of(b) for b in lat])
    ax.set_xscale("log")
    ax.set_xlabel("wall-clock latency per request (ms, log) — includes network for hosted models")
    ax.set_ylim(-0.6, len(lat) - 0.2)
    ax.grid(axis="y", visible=False)
    ax.set_title("Speed (one request = all questions for one item)")
    save(fig, out / "fig7_latency.png")


# ------------------------------------------------------------------ report
def write_report(e1, e2, order, tab, out_md: Path, fig_dir: Path):
    rel = lambda p: Path("..") / fig_dir.name / p  # noqa: E731
    L = ["# Results", "", f"Backends: {', '.join(label_of(b) for b in order)}", ""]
    L += ["## Calibration (held-out half, temperature fitted on the other half)", "",
          "| model | n items | acc | ECE raw → scaled | Brier raw → scaled | log-loss raw → scaled | fitted T |",
          "|---|---|---|---|---|---|---|"]
    for b in order:
        t = tab[b]
        temps = ", ".join(f"{k} {v:.2f}" for k, v in sorted(t["temps"].items()))
        L.append(f"| {label_of(b)} | {t['n_test']} | {t['raw']['Acc']:.2f} | {t['raw']['ECE']:.3f} → {t['scaled']['ECE']:.3f} "
                 f"| {t['raw']['Brier']:.3f} → {t['scaled']['Brier']:.3f} | {t['raw']['LogLoss']:.3f} → {t['scaled']['LogLoss']:.3f} | {temps} |")
    L += ["", "T > 1 means the raw probabilities were over-confident; T < 1, under-confident.", ""]
    keys = sorted({f"{r['task']}/{r['question']}" for rows in e1.values() for r in rows})
    L += ["## Accuracy by question (all items)", "", "| question | " + " | ".join(label_of(b) for b in order) + " |",
          "|---|" + "---|" * len(order)]
    for k in keys:
        cells = []
        for b in order:
            v = [int(r["correct"]) for r in e1[b] if f"{r['task']}/{r['question']}" == k]
            cells.append(f"{np.mean(v):.2f}" if v else "–")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    if e2:
        L += ["", "## Wording sensitivity", "", "| model | task | n | median range of P(yes) | decision flips |", "|---|---|---|---|---|"]
        for b in order:
            for t in sorted({r["task"] for r in e2.get(b, [])}):
                sub = [r for r in e2[b] if r["task"] == t]
                L.append(f"| {label_of(b)} | {t} | {len(sub)} | {statistics.median(r['range'] for r in sub):.3f} | "
                         f"{np.mean([r['flips'] for r in sub]):.0%} |")
    L += ["", "## Figures", ""]
    for f in sorted(fig_dir.glob("fig*.png")):
        L += [f"![{f.stem}]({rel(f.name).as_posix()})", ""]
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", out_md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.figures)
    out.mkdir(parents=True, exist_ok=True)
    e1 = load_rows(res / "exp01")
    e2 = load_rows(res / "exp02")
    if not e1:
        sys.exit("no results/exp01/<backend>/rows.jsonl yet — run the experiments first")
    order = sorted(e1, key=lambda b: (FIXED.get(b, 99), b))
    fig_accuracy(e1, order, out)
    fig_reliability(e1, order, out)
    tab = calib_table(e1, order)
    fig_calib_bars(tab, order, out)
    fig_confidence(e1, order, out)
    fig_agreement(e1, order, out)
    fig_wording(e2, [b for b in order if b in e2] + [b for b in e2 if b not in order], out)
    fig_latency(e1, e2, order, out)
    write_report(e1, e2, order, tab, res / "REPORT.md", out)


if __name__ == "__main__":
    main()
