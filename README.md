# subq-decision-lab

Small experiments comparing **decision models**: models that return typed
probabilities (yes/no, choice, score) instead of generated text.

- **SubQ** `subq-snap-preview`: hosted, public preview
- **Laya** (Convai Innovations, Apache-2.0): open weights, run locally with `laya-serve`, `english` and `typed-decisions` checkpoints

Both expose the same `POST /v1/systemone` wire format, so one client
(`subq_lab/client.py`) talks to either and every model gets the same items.

The question: **when the right answer can be computed exactly, how accurate
and how well calibrated are the probabilities, and how much does the wording
move them?** Ground truth comes from mechanism-design micro-tasks where the
answer takes a few lines of code.

## Experiments

| Script | What it measures |
|---|---|
| `exp00_smoke.py` | One SubQ call: key, credits, response shape |
| `exp01_calibration.py` | Accuracy, Brier, log-loss, ECE on 3 task families |
| `exp02_wording.py` | Spread of P(yes) across 4 phrasings of the same question |
| `analysis/make_figures.py` | Figures 1–7 and `results/REPORT.md` from whatever backends have results |

Task families (`subq_lab/tasks.py`):

- **Auctions**: first- vs second-price sealed bid. Questions: winner, price paid, whether truthful bidding is dominant.
- **Stable matching**: 3×3 preferences. Questions: is a shown matching stable, which pair blocks it, m1's Gale–Shapley partner, number of blocking pairs.
- **Drone landing-point triage**: a fixed divert rule (queue/pads, wind, battery). Questions: divert or not, congestion level.

## Figures

| | |
|---|---|
| `fig1_accuracy` | top-1 accuracy per question, models side by side, with the chance level |
| `fig2_reliability` | reliability diagrams (raw vs temperature-scaled) plus a histogram of predicted probabilities |
| `fig3_calibration_bars` | ECE / Brier / log-loss before and after temperature scaling |
| `fig4_confidence_right_vs_wrong` | whether a model is less sure when it is wrong |
| `fig5_agreement` | P(yes) from model A vs model B on the same yes/no items |
| `fig6_wording` | spread of P(yes) across phrasings, and accuracy per phrasing |
| `fig7_latency` | wall-clock latency per request |

Temperature scaling fits one temperature per question type on even-numbered
items and reports odd-numbered items, so the scaled numbers are out-of-sample.
Laya's docs say its checkpoints ship over-confident and recommend this step.

## Usage

Windows, all in one: `powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1` (finds or starts a local `laya-serve`, runs everything, writes `figures/` and `results/REPORT.md`).

Manual:

```bash
pip install -r requirements.txt
export SUBQ_API_KEY=sk-...                 # PowerShell: $env:SUBQ_API_KEY="sk-..."
python experiments/exp00_smoke.py

# Laya: pip install "laya[serve]"; LAYA_DEVICE=cuda LAYA_PORT=8000 laya-serve
export LAYA_BASE_URL=http://127.0.0.1:8000/v1

python experiments/exp01_calibration.py --backend subq --backend laya --backend laya-typed --n 40
python experiments/exp02_wording.py     --backend subq --backend laya --backend laya-typed --n 30
python analysis/make_figures.py
```

`--backend mock-sharp --backend mock-soft` runs the whole pipeline offline with random answers, for testing only.

## Cost control

- SubQ pricing as documented (Oct 2026): $0.05 per 1M input tokens, output free. Both experiments together use well under $0.01.
- Every request is cached in `results/cache.jsonl`, keyed by a hash of backend + body, so re-runs cost nothing.
- `SUBQ_BUDGET_USD` (default `0.50`) stops the client once that much has been spent in a run.

## Caveats

- Both models are previews or early releases; results describe one version at one point in time.
- Hosted latency includes the network round trip; local latency depends on your GPU.
- Laya's `english` checkpoint has a 512-token context; the states here are well under that.
- Never commit API keys. `.env` is git-ignored.
