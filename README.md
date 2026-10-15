# subq-decision-lab

Small experiments with the SubQ Decisions API (`subq-snap-preview`), a
"System One" model that returns typed probabilities (yes/no, choice, score)
instead of generated text.

The question here: **when the right answer can be computed exactly, how
accurate and how well calibrated are those probabilities?** Ground truth comes
from mechanism-design micro-tasks where the answer is a few lines of code.

## Experiments

| Script | What it measures | Approx. cost |
|---|---|---|
| `exp00_smoke.py` | One call: key, credits, response shape | < $0.0001 |
| `exp01_calibration.py` | Accuracy, Brier, log-loss, ECE on 3 task families | ~$0.003 at `--n 40` |
| `exp02_wording.py` | Spread of P(yes) across 4 phrasings of the same question | ~$0.006 at `--n 30` |

Task families (`subq_lab/tasks.py`):

- **Auctions** — first- vs second-price sealed bid: winner, price paid, whether truthful bidding is dominant.
- **Stable matching** — 3×3 preferences: is a shown matching stable, which pair blocks it, m1's Gale–Shapley partner, number of blocking pairs.
- **Drone landing-point triage** — a fixed divert rule (queue/pads, wind, battery) and a congestion level.

## Usage

```bash
pip install -r requirements.txt
export SUBQ_API_KEY=sk-...          # PowerShell: $env:SUBQ_API_KEY="sk-..."
python experiments/exp00_smoke.py
python experiments/exp01_calibration.py --n 40
python experiments/exp02_wording.py --n 30
```

Add `--mock` to any experiment to run it offline for free.

## Cost control

- Pricing as documented (Oct 2026): $0.05 per 1M input tokens, output free.
- Every request is cached in `results/cache.jsonl` by a hash of its body; re-runs cost nothing.
- `SUBQ_BUDGET_USD` (default `0.50`) stops the client once that much has been spent in a run.

## Notes

- The API is in public preview; results describe one model version at one point in time.
- Never commit your API key. `.env` is git-ignored.
