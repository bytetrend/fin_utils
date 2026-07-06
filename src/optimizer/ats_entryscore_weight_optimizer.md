# ats_entryscore_weight_optimizer.py

Optimizes the per-component **weights** (and the score cutoff) of the
`AtsPriceQuickReversal` / `AtsPriceBrkout` `EntryScore` formula:

```
EntryScore = W1 * IFF(C5, 1, 0)                                  // Speed flip
           + W2 * IFF(RevATRsPerSec > RevATRsPerSecLim, 1, 0)    // Fast reversal   (== C13)
           + W3 * IFF(CVDAvg >= CVDAvgLim, 1, 0)                 // Volume directional (== C6)
           + W4 * IFF(CountIf(CVDAcel>=CVDAcelLim,2)>0, 1, 0)    // Accel last 2 bars  (== C10)
           + W5 * IFF(CVDDelta>0, 1, 0)                          // Delta confirms
           + W6 * IFF(C7, 1, 0)                                  // PipSpeedPct >= HMinPipSpeedPct
           + W7 * IFF(C9, 1, 0)                                  // HMAGapCV <= HMinHMAGapCV

Take the trade if EntryScore >= Threshold
```

Currently `W1..W7 = 1` for every component (an unweighted vote). This script
searches for better weights **and** a matching score threshold using Optuna.

## Why this is a different tool than the other two optimizers

- `ats_param_optimizer.py` and `ats_optuna_optimizer.py` search independent
  `>=`/`<=` thresholds on **continuous columns**.
- This script instead searches **one integer weight per binary component**,
  plus a single score cutoff — the actual decision variables in your
  `EntryScore` formula. It's a fundamentally different (and larger) search
  space: 7 weights × 1 threshold, vs. one threshold per parameter.

## Requirements

```bash
pip install pandas numpy optuna
```

## How component detection works

By default the script pulls the 7 components straight from your CSV's own
columns, matching your formula exactly:

| Label | Column | Condition |
|---|---|---|
| Speed flip (C5) | `ind_C5` | flag != 0 |
| Fast reversal (C13) | `ind_C13` | flag != 0 |
| Volume directional (C6) | `ind_C6` | flag != 0 |
| CVD accel last 2 bars (C10) | `ind_C10` | flag != 0 |
| CVDDelta confirms | `ind_CVDDelta` | value > 0 |
| PipSpeedPct sustained (C7) | `ind_C7` | flag != 0 |
| HMAGapCV consistent (C9) | `ind_C9` | flag != 0 |

`C13`, `C6`, and `C10` are used directly because they already encode
`RevATRsPerSec > RevATRsPerSecLim`, `CVDAvg >= CVDAvgLim`, and the
`CVDAcel` count condition respectively, per the strategy's own flag
definitions — no need to reconstruct them from raw thresholds.

**Data-quality note:** reconstructing `EntryScore` as the plain sum of these
7 raw components does not always exactly match a logged `ind_EntryScore`
column if one exists in your CSV (in one dataset it matched ~90% of rows,
off by ±1 elsewhere). This script optimizes from the **raw components**
directly, not from a logged `EntryScore` column, so that mismatch doesn't
affect its results — but if you see it, it's worth checking whether your
`CVDDelta > 0` condition should be direction-normalized (i.e. different for
long vs. short) in the live engine.

## Basic usage

```bash
python ats_entryscore_weight_optimizer.py trades.csv
```

Loads the CSV, splits long/short, holds out the most recent 25% of trades per
direction as a test set, searches weights + threshold on the remaining 75%,
and reports the optimized formula's performance against a fair baseline (the
current equal-weight formula, using its own best threshold).

## Options (inputs) explained

| Flag | Default | Meaning |
|---|---|---|
| `--min-n N` | 30 | Minimum trades required in any filtered training subset. Any weight/threshold combination that would drop below this is rejected during the search. |
| `--n-trials N` | 2000 | Optuna trials per direction. This search space (7 weights + threshold) is bigger than the single-parameter sweep, so it benefits from more trials than `ats_optuna_optimizer.py`'s default. |
| `--test-fraction F` | 0.25 | Fraction of trades (chronologically last) held out as the test set. The optimizer never sees these trades while searching. |
| `--cv-folds N` | 1 (off) | If > 1, cross-validate the objective across this many folds of the training data instead of raw training expectancy — penalizes weight/threshold combos that only work on one lucky slice of training data. |
| `--max-weight N` | 5 | Maximum integer weight searched per component (each component gets an integer from 0 to N). Lower this to shrink the search space on small datasets (see Pitfalls). |
| `--min-threshold-frac F` | 0.3 | Minimum score threshold searched, as a fraction of that trial's own max possible score. **Read the "Degenerate threshold" pitfall below before changing this.** Set to `0` to restore the old unconstrained search. |
| `--components "..."` | auto-detect | Override the default 7 components. Format: `"Label1:col1:flag,Label2:col2:gt0,..."` where the comparison is `flag` (column already 0/1, tested as != 0) or `gt0` (column tested as > 0). |
| `--seed N` | 42 | Random seed for reproducibility. |
| `--output PATH` | none | Write the full structured report as JSON. |

### Example: shrink the search space for a small dataset

```bash
python ats_entryscore_weight_optimizer.py trades.csv --max-weight 2 --n-trials 1000
```

### Example: cross-validate within training data

```bash
python ats_entryscore_weight_optimizer.py trades.csv --cv-folds 3
```

### Example: use a custom component list

```bash
python ats_entryscore_weight_optimizer.py trades.csv \
  --components "SpeedFlip:ind_C5:flag,FastRev:ind_C13:flag,VolDir:ind_C6:flag"
```

### Recommended starting command for a few-hundred-trade dataset

```bash
python ats_entryscore_weight_optimizer.py trades.csv \
  --n-trials 2000 --min-n 25 --max-weight 3 --cv-folds 3 --min-threshold-frac 0.3 \
  --output entryscore_report.json
```

## What the output looks like

```
LONG
Training window: 114 trades   Test window (held-out): 38 trades

Current formula (all weights=1), best threshold found on training data: score >= 1
  TRAIN expectancy: $-2.55/trade
  TEST expectancy (same filter, held-out):  $12.31/trade

Optuna-optimized weights (2000 trials searched):
    Speed flip (C5)                  weight = 1   #
    Fast reversal (C13)              weight = 3   ###
    Volume directional (C6)          weight = 0   -
    CVD accel last 2 bars (C10)      weight = 3   ###
    CVDDelta confirms                weight = 0   -
    PipSpeedPct sustained (C7)       weight = 3   ###
    HMAGapCV consistent (C9)         weight = 0   -
  Score threshold: EntryScore >= 3  (max possible score = 10)

  TRAIN performance (subset matching the optimized weights, within the 114-trade training window):
    n=113  hit_rate=25.7%  expectancy=$-2.55/trade

  >>> TEST performance (held-out, the only trustworthy number) <<<
    n=38  hit_rate=55.3%  expectancy=$12.31/trade  total_pl=$467.86  [OK]
    vs. equal-weight formula on same test window: +$0.00/trade
```

## How to interpret the results

1. **Ignore the TRAIN numbers except as a comparison point.** They're what
   Optuna was allowed to fit to and will look good by construction.
2. **The "TEST performance" block is the only one that matters.** It shows
   how the optimized weights perform on trades the search never touched.
3. **"vs. equal-weight formula on same test window"** is the single most
   important line. This compares the optimized weights against your
   **current formula's own best threshold**, on the same held-out data —
   not against "no filter." A positive number means real improvement over
   what you're running today; negative or near-zero means the reweighting
   didn't help (or hurt).
4. **`[LOW CONFIDENCE (n=X < min-n=Y)]`** means the held-out window was too
   small to trust the number either way — treat it as "not yet tested,"
   not as a pass or fail.
5. **The "sign of overfitting" note** (large train-to-test expectancy drop)
   is the script actively warning you that the weights it found are
   probably noise-fitting, not a real pattern.
6. **Weight bars (`###`)** are a quick visual for which components the
   search leaned on. A component landing at weight 0 means the search found
   it added nothing (or hurt) once the others were accounted for — but
   don't over-read this on a small, overfit run; a component's "true" value
   can still be nonzero even if a noisy search zeroed it out this time.
7. **`+$0.00/trade` vs. baseline is not automatically a "no effect" result —
   check whether it's the degenerate-threshold pattern below first.** A
   filter can match the exact same 113-of-114 trades as baseline for
   structural reasons, not because reweighting was tested and found neutral.

## Pitfalls to watch

- **Degenerate threshold ("take almost everything"), the most important
  pitfall to check for.** With several components carrying positive
  weight, a low-enough threshold can be satisfied by *any single component*
  firing on its own — turning the "optimized" filter into something that
  matches nearly every trade in the dataset, indistinguishable from no
  filter at all, regardless of what the specific weights are. This showed
  up in an earlier run as `threshold=1` out of a possible score of 17,
  producing a filter that matched 113 of 114 training trades and reported
  `vs. equal-weight formula: +$0.00/trade` — not because reweighting was
  neutral, but because the search never actually tested a meaningfully
  different filter. **`--min-threshold-frac` (default 0.3) exists
  specifically to close this loophole** by forcing the threshold to be at
  least that fraction of the trial's own max achievable score. It doesn't
  fully eliminate the risk on its own, though: a search can still assign
  several components the same maximum weight so that any one of them alone
  clears the (now higher) threshold. If you see a result where several
  components share the top weight value and `train_n` is barely below the
  full training window size, treat it as the same degenerate pattern in a
  different shape, not a real combination effect — check `train_n` against
  `total_train_n` for that direction to catch this.
- **This search space is much bigger than the single-threshold sweeps.**
  7 free integer weights plus a threshold is roughly 8 free parameters. With
  only a few dozen trades per direction in your training window, this is
  very easy to overfit — expect the training numbers to look great and the
  test numbers to disappoint until your trade count grows substantially.
  This is normal and the script is designed to surface it, not hide it.
- **Small test windows make "vs. equal-weight" noisy.** With `test_fraction`
  producing single-digit test trades, a "beats baseline" or "loses to
  baseline" result can flip from one additional trade. Don't commit to new
  live weights on a test window under your `--min-n`.
- **`--max-weight` trades search flexibility for overfitting risk.** A wide
  range (5+) lets the optimizer express more nuanced weightings but needs
  much more data to do so reliably. On a dataset in the low hundreds of
  trades, start with `--max-weight 2` or `3` and only widen it once you have
  enough trades that the test window stays above `--min-n`.
- **`--min-threshold-frac` too high can make the search infeasible.** If you
  raise it a lot (e.g. above 0.6) on a small `--max-weight`, few or no
  training subsets may clear both the threshold floor and `--min-n`
  simultaneously, and the direction will report no usable result. If a
  direction comes back empty, try lowering `--min-threshold-frac` or raising
  `--max-weight` before assuming there's nothing to find.
- **A legitimate "no benefit from reweighting" result looks similar to the
  degenerate pattern, so check `train_n` to tell them apart.** If, even
  after applying `--min-threshold-frac`, the best filter still matches
  nearly all of `total_train_n`, that can mean the search is genuinely
  concluding that no component combination beats taking almost every trade
  in that direction — which is a real (if underwhelming) finding, not a
  bug, provided several components didn't just gang up at the same weight
  to recreate the loophole above. When in doubt, look at the actual
  weights: if 2-3 components share the maximum weight value and the
  threshold is close to that shared value, it's worth re-running with a
  higher `--min-threshold-frac` to see if the conclusion holds.
- **Weights found here are a starting hypothesis, not a formula to deploy
  immediately.** Even a run that "beats baseline" on a small test window
  should be treated as a candidate to re-check on the next batch of trades
  (the same way the earlier `FullDeltaATRs` threshold finding was confirmed
  by testing it again on a separate dataset) before changing the live
  strategy.
- **The CVDDelta direction-normalization question.** If your CSV's logged
  `EntryScore` doesn't match the plain sum of these 7 components, check
  whether `CVDDelta > 0` should flip sign for short trades in your engine.
  Using the wrong convention here would make this script optimize weights
  against a component that isn't quite the one your engine actually uses.
- **Long and short almost certainly need different weights.** The script
  already treats them separately (as it should) — don't average or share a
  single weight set across directions.

## Recommended workflow

1. Run `ats_feature_importance.py` first to see which of these 7 components
   (if any) show real signal at all for each direction.
2. Run this script with a conservative `--max-weight` (2-3) and the default
   `--min-threshold-frac` (0.3) given your current trade count.
3. Before trusting any result, check `train_n` against `total_train_n` for
   that direction — if the "optimized" filter matches nearly every training
   trade, treat it as the degenerate pattern (see Pitfalls) rather than a
   real finding, even if `--min-threshold-frac` is enabled.
4. Only trust a result where TEST performance beats the equal-weight
   baseline **and** the test window meets your `--min-n`.
5. Re-run on the next batch of trades before changing the live formula, the
   same way you'd forward-test any other parameter finding.
