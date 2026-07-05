# ats_optuna_optimizer.py

Bayesian (Optuna/TPE) threshold optimization for `AtsPriceQuickReversal` /
`AtsPriceBrkout` entry-filter parameters.

This is a different tool than `ats_param_optimizer.py`. The grid sweep in that
script tests one or two parameters at a time. This script searches **all**
candidate parameters jointly in a single search, letting the optimizer decide
for itself which parameters belong in the filter (via a "none / ≥ / ≤" choice
per parameter) instead of requiring you to hand-pick which pairs to test — and
it validates whatever it finds on data it never optimized against.

## Requirements

```bash
pip install pandas numpy optuna
```

## How overfitting is controlled

Read this before trusting any output from this script.

1. **Chronological train/test split.** Trades are sorted by `EntryDate`.
   Optuna only ever sees the training window while searching. The filter it
   finds is then evaluated **once** on the held-out test window that comes
   after it in time. **The test-window numbers are the only numbers that
   mean anything** — the training-window numbers are what Optuna was allowed
   to fit to, and will always look good by construction, the same way any
   search over enough thresholds eventually finds something that looks good
   on the data it searched.
2. **Min-n guard.** Any candidate filter that drops the training subset below
   `--min-n` trades is rejected (heavily penalized) during the search.
3. **Optional k-fold cross-validation within training data** (`--cv-folds`).
   Instead of optimizing raw training expectancy, the objective can average
   expectancy across K folds of the training data, which penalizes filters
   that only work on one lucky subset of the training window.

## Basic usage

```bash
python ats_optuna_optimizer.py trades.csv
```

Loads the CSV, splits long/short, holds out the most recent 25% of trades per
direction as a test set, runs 1000 Optuna trials per direction on the
remaining 75%, and reports both training and (the only ones that matter)
test-window results.

## Options

| Flag | Default | Description |
|---|---|---|
| `--min-n N` | 30 | Minimum trades required in any filtered training subset (overfitting guard) |
| `--n-trials N` | 1000 | Optuna trials to run per direction |
| `--test-fraction F` | 0.25 | Fraction of trades (chronologically last) held out as the test set |
| `--cv-folds N` | 1 (off) | If > 1, cross-validate the objective across this many folds of the training data instead of raw training expectancy |
| `--params "a,b,c"` | auto-detect | Comma-separated list of parameter columns to search over, instead of every numeric `ind_*` column |
| `--seed N` | 42 | Random seed for reproducibility |
| `--output PATH` | none | Write the full structured report as JSON |

## Examples

Run more trials with a stricter overfitting guard:

```bash
python ats_optuna_optimizer.py trades.csv --n-trials 2000 --min-n 30
```

Give the search more history to test against, with in-training cross-validation:

```bash
python ats_optuna_optimizer.py trades.csv --test-fraction 0.3 --cv-folds 5
```

Restrict the search to a short list of parameters you already trust — e.g.
ones that came out on top of `ats_feature_importance.py`'s ranking — instead
of searching every column:

```bash
python ats_optuna_optimizer.py trades.csv \
  --params ind_FullDeltaATRs,ind_FullAngle,ind_ATRsFromHma
```

Save the full structured results for later comparison across dataset runs:

```bash
python ats_optuna_optimizer.py trades.csv --output optuna_report.json
```

## What the output looks like

For each direction (LONG / SHORT):

```
Training window: 41 trades total, baseline_expectancy=$-8.06
Test window (held-out, never seen by optimizer): 14 trades total, baseline_expectancy=$31.94

Optuna-found filter (800 trials searched):
    ind_ATRsFromHma<=0.555848
    ind_RevATRsPerSec<=0.54032
    ind_HMAGapCV<=2.17527

  TRAIN performance (subset matching the filter, within the 41-trade training window):
    n=16  hit_rate=37.5%  expectancy=$8.30/trade

  >>> TEST performance (subset matching the filter, within the 14-trade
      held-out window -- the only trustworthy number) <<<
    n=5  hit_rate=40.0%  expectancy=$1.16/trade  total_pl=$5.81  [LOW CONFIDENCE (n=5 < min-n=15)]
    NOTE: expectancy dropped substantially from train to test ($8.30 -> $1.16).
    This is a sign of overfitting to the training window -- treat this filter with skepticism.
```

Plus a closing summary block reminding you which numbers to trust.

## How to interpret the results

- **Only look at the TEST performance block.** The training-window numbers
  exist so you can see the gap between them and the test numbers, not so you
  can act on them directly.
- **A big drop from train expectancy to test expectancy is a red flag, not
  bad luck.** The script calls this out explicitly (the "NOTE" line above).
  It means the filter fit noise in the training window rather than a real
  pattern.
- **`[LOW CONFIDENCE (n=X < min-n=Y)]`** on the test line means the held-out
  window was too small to say anything reliable — the filter might be
  perfectly fine, but this run can't tell you that. Don't promote it to the
  live strategy on this evidence alone.
- **`n=0` in the test window** means the filter never fired on held-out data
  at all — you cannot confirm anything about it from this run.
- **A filter that survives with test expectancy close to (or better than)
  training expectancy, at an adequate test-window `n`, is the strongest
  signal this script can give you** — it means the pattern held up on data
  the optimizer never got to search against.

## A note on small datasets

With a dataset in the low hundreds of trades (or fewer per direction), a 25%
test split can leave single-digit test windows — too few to confirm or reject
anything. In that situation, the honest reading of the output is usually
"not enough data to test this yet," not "this filter works" or "this filter
fails." Increasing `--test-fraction` gives a more reliable test at the cost of
a smaller (noisier) training search, and there's no way around that trade-off
except collecting more trades over more time. Adding more symbols to a
backtest without extending the date range does not fix this — it adds
breadth, not depth, and the number of trades over time is what this script's
train/test split actually needs.
