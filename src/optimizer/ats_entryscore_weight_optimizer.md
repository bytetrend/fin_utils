# ats_entryscore_weight_optimizer.py

Optimizes the per-component **weights** (and the score cutoff) of the
`AtsFastReversal` / `AtsSlowReversal` `EntryScore` formula:

```
EntryScore = W1 * IFF(C5, 1, 0)                                  // Speed flip
           + W2 * IFF(RevATRsPerSec > RevATRsPerSecLim, 1, 0)    // Fast reversal   (== C13)
           + W3 * IFF(CVDAvg >= CVDAvgLim, 1, 0)                 // Volume directional (== C6)
           + W4 * IFF(CountIf(CVDAcel>=CVDAcelLim,2)>0, 1, 0)    // Accel last 2 bars  (== C10)
           + W5 * IFF(CVDDeltaPct>0, 1, 0)                          // Delta confirms
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
| CVDDeltaPctPct confirms | `ind_CVDDeltaPct` | value > 0 |
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
`CVDDeltaPct > 0` condition should be direction-normalized (i.e. different for
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
    CVDDeltaPct confirms                weight = 0   -
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

## JSON Output File Parameters

When using the `--output` flag, the script produces a JSON file with the following
structure and parameters. Understanding these is critical to applying the optimization
results in your trading system.

### Top-Level Parameters

| Parameter | Type | Description |
|---|---|---|
| `csv_path` | string | Path to the input trades CSV file analyzed |
| `min_n` | integer | Minimum number of trades required for any test subset to be considered valid |
| `n_trials` | integer | Number of Optuna optimization trials performed |
| `test_fraction` | float | Fraction of trades held out as the test set (e.g., 0.3 = 30%) |
| `cv_folds` | integer | Number of cross-validation folds used (1 = no cross-validation) |
| `max_weight` | integer | Maximum weight value searched for any component (0 to max_weight) |
| `min_threshold_frac` | float | Minimum threshold as a fraction of max_possible_score (prevents degenerate filters) |

### Long/Short Direction Results

The JSON contains both `"long"` and `"short"` objects with identical parameter structures
but separate optimization results. Each contains:

#### Direction-Level Parameters

| Parameter | Type | Description |
|---|---|---|
| `direction_label` | string | "long" or "short" — the trade direction optimized |
| `total_train_n` | integer | Total number of trades available for training in this direction before filtering |
| `total_test_n` | integer | Total number of trades in the test set before filtering |

#### Weights Dictionary

| Parameter | Type | Example | Description |
|---|---|---|---|
| `weights` | object | `{"Fast reversal (C13)": 1, "HMAGapCV consistent (C9)": 1}` | The optimized weight for each component. Weight 0 means the component was deemphasized or eliminated by the search as unhelpful. Higher weights (e.g., 3) mean the component is more important. Only components with weight > 0 fire the signal. |

#### Performance Metrics — Training Data

These represent how the optimized filter performed on the training window
(which Optuna was allowed to overfit to). **Do not trust these numbers for
live trading decisions—use TEST metrics instead.**

| Parameter | Type | Example | Description |
|---|---|---|---|
| `train_n` | integer | 94 | Number of training trades that passed the optimized filter (EntryScore >= threshold) |
| `train_expectancy` | float | 1.859893617021277 | Average profit/loss per trade in the training set matching the filter ($/trade) |
| `train_hit_rate` | float | 0.2765957446808511 | Fraction of trades in the training set that were profitable (0.0 to 1.0 or 0-100%) |

#### Performance Metrics — Test Data (Held-Out, Primary Metric)

**These are the most important numbers.** The test set was never seen by the
optimizer and represents expected live performance.

| Parameter | Type | Example | Description |
|---|---|---|---|
| `test_n` | integer | 41 | Number of test trades that passed the optimized filter. If this is very small (< min_n), the result has low confidence. Compare to total_test_n to detect degenerate filters. |
| `test_expectancy` | float | -1.175853658536585 | Average profit/loss per trade in the test set ($/trade). **This is the key performance metric for live deployment.** Negative values mean the filter loses money on average per trade. |
| `test_hit_rate` | float | 0.36585365853658536 | Fraction of test trades that were profitable (0.0 to 1.0). A high hit rate with negative expectancy means winners are smaller than losers; conversely, a lower hit rate with positive expectancy means the winners are larger on average. |
| `test_total_pl` | float | -48.20999999999999 | Total profit/loss across all test trades (not per-trade; sum of all P&L). Useful for sanity-checking against test_expectancy × test_n. |

#### Baseline Metrics (for Comparison)

The baseline is the **current formula with all weights = 1**, optimized to its
own best threshold on the training data. This is your **control group** for
assessing whether the new weights are actually better.

| Parameter | Type | Example | Description |
|---|---|---|---|
| `baseline_train_expectancy` | float | 1.6571578947368428 | Training-set expectancy of the equal-weight (all 1s) formula with its own best threshold. Shows what the current live formula would have produced on training data. |
| `baseline_test_expectancy` | float | 0.8235714285714286 | Test-set expectancy of the equal-weight formula with its best threshold on the same held-out data. **Compare this to test_expectancy to judge improvement: if test_expectancy > baseline_test_expectancy, the new weights beat the current formula.** |
| `equal_weight_best_threshold` | integer | 2 | The threshold (EntryScore cutoff) that was optimal for the equal-weight baseline on training data. Combined with baseline_train_expectancy and baseline_test_expectancy, this tells you what your current live formula would achieve. |

#### Optimization Metadata

| Parameter | Type | Example | Description |
|---|---|---|---|
| `threshold` | integer | 1 | The optimized EntryScore threshold (cutoff). Only trades with EntryScore >= threshold enter. Combined with the weights dict, this defines the new filter to deploy. |
| `max_possible_score` | integer | 2 | Maximum possible score achievable with the optimized weights (sum of all weights). Used to validate whether the threshold is degenerate (see Pitfalls). If threshold is very low relative to max_possible_score, beware. |
| `n_trials_run` | integer | 2000 | Number of Optuna trials actually performed for this direction. Confirms the search scope. |
| `warning` | string or null | null | If not null, a warning message (e.g., "Training set too small"). Check this before trusting results. |

### Example Interpretation

Given this output:
```json
{
  "threshold": 1,
  "max_possible_score": 2,
  "train_n": 94,
  "train_expectancy": 1.86,
  "test_n": 41,
  "test_expectancy": -1.18,
  "test_total_pl": -48.21,
  "baseline_test_expectancy": 0.82,
  "equal_weight_best_threshold": 2,
  "weights": {
    "Fast reversal (C13)": 1,
    "HMAGapCV consistent (C9)": 1
  }
}
```

**What this means:**
- The optimizer found that just 2 out of 7 components matter: Fast reversal and HMAGapCV
- With these weights, a threshold of ≥1 (meaning either component firing) is optimal
- **On test data (the honest metric):** This filter loses $1.18/trade on average — **worse than the current formula's +$0.82/trade**
- The result is **not recommended for live deployment** because test_expectancy is negative

**Red flag:** The test_n (41) is close to total_test_n, and max_possible_score is only 2
while threshold is 1, which means the filter triggers whenever even ONE component fires.
This is a degenerate filter (see Pitfalls section).

## Recommended Workflow

1. Run `ats_feature_importance.py` first to see which of these 7 components
   (if any) show real signal at all for each direction. This informs whether
   the optimization will have enough raw material to work with.
   
2. Run this script with a conservative `--max-weight` (2-3) and the default
   `--min-threshold-frac` (0.3) given your current trade count. Small datasets
   (< 200 total trades) should use `--max-weight 2` to reduce overfitting risk.

3. Review the JSON output carefully. Check:
   - Are test_expectancy results positive?
   - Does test_expectancy beat baseline_test_expectancy?
   - Is test_n sufficiently large (ideally > 2× min_n)?
   - Is the filter selective (test_n << total_test_n)?
   - Does train_expectancy drop more than 2× relative to test_expectancy?
   
4. Before trusting any result, check `train_n` against `total_train_n` for
   that direction — if the "optimized" filter matches nearly every training
   trade, treat it as the degenerate pattern (see Pitfalls) rather than a
   real finding, even if `--min-threshold-frac` is enabled.

5. Only deploy weights if TEST performance beats the equal-weight baseline
   **and** the test window meets your `--min-n` AND none of the red flags
   (degenerate filter, overfitting, negative expectancy) apply.

6. Forward-test on the next batch of trades before committing the new weights
   to your live trading formula. Same as you'd validate any other parameter finding.
   
7. Re-run monthly or quarterly as you accumulate more data to track whether
   component importance is stable or shifting with market conditions.

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
- **The CVDDeltaPct direction-normalization question.** If your CSV's logged
  `EntryScore` doesn't match the plain sum of these 7 components, check
  whether `CVDDeltaPct > 0` should flip sign for short trades in your engine.
  Using the wrong convention here would make this script optimize weights
  against a component that isn't quite the one your engine actually uses.
- **Long and short almost certainly need different weights.** The script
  already treats them separately (as it should) — don't average or share a
  single weight set across directions.

## How to Use Weights and Parameters to Improve Profitability

The optimization output reveals which signal components are valuable and how to combine
them for better results. Here's the practical workflow:

### 1. Identify High-Conviction Components (Weight > 0)

The optimized weights dictionary shows which components survived the search:
- **Weight 0 = noise or redundant signal**: The component added no value or conflicted
  with stronger components. You can deprioritize or ignore it.
- **Weight 1 = core signal**: The component reliably contributed; keep it active.
- **Weight 2+ = high conviction**: The search needed this component at higher weight
  to capture signal other components missed. Prioritize these.

**Action:** Update your live trading engine to enforce the new weights. In the
`EntryScore` formula, multiply each component by its optimized weight instead of
the current all-1s weighting.

### 2. Adjust the Threshold (Score Cutoff)

The `threshold` parameter specifies the EntryScore cutoff (≥ condition):

- **threshold=1** with max_possible_score=5: Very loose filter, catches 80%+ of trades.
  Best for high-conviction strategies or when you want to trade frequently but need
  reliability via the weights themselves.
- **threshold=4** with max_possible_score=5: Tight filter, only trades when 4+ components
  align. Lower trade frequency but potentially higher quality and selectivity.

**Action:** Update your engine's score cutoff. If test_expectancy is positive, use the
exact threshold from the JSON. If you want to be conservative, you can raise it by 1
(reducing trades traded but hopefully improving quality).

### 3. Compare Against Baseline to Judge Real Improvement

The baseline metrics reveal whether the new weights actually help:

```
Improvement = test_expectancy - baseline_test_expectancy
```

- **Improvement > 0 (e.g., +$0.50/trade)**: The new weights beat your current formula.
  Confidence is highest if:
  - test_expectancy > $0.50/trade (positive edge)
  - test_n is large (≥ min_n, ideally 2× min_n or more)
  - test_n << total_test_n (not degenerate; filter is selective)
  
- **Improvement ≈ 0 or slightly negative**: No clear benefit over the current formula.
  Before deploying:
  - Re-run on a fresh batch of trades to confirm (forward-testing)
  - Check whether the threshold is degenerate (see red flags below)
  
- **Improvement << 0 (e.g., -$2.00/trade)**: The optimized weights underperform.
  Do **not** deploy. The search likely overfit to training data or discovered a
  component combination that doesn't generalize.

### 4. Red Flags: When NOT to Deploy

Before applying new weights, check for these disqualifying patterns:

**a) Degenerate Filter**
```
Rule: if threshold <= max_possible_score * 0.3 AND test_n ≈ total_test_n:
    -> Filter is too loose; treats like no filter at all
```
Example: `threshold=1, max_possible_score=6, test_n=38, total_test_n=40`
means almost every trade passes (only 2 were filtered out). This isn't a real
optimization; it's just "take almost every trade." Don't deploy.

**b) Large Train-to-Test Drop**
```
Rule: if (train_expectancy - test_expectancy) > 2 * baseline_test_expectancy:
    -> Sign of severe overfitting
```
Example: `train_expectancy=+$5.00, test_expectancy=-$1.00` suggests the search
fit to noise in the training window. Very unlikely to replicate on future trades.

**c) Low Confidence (Small Test Set)**
```
Rule: if test_n < min_n:
    -> Result is unreliable due to small sample
```
Single-digit test trades can flip from one lucky/unlucky trade. Increase --test-fraction
or gather more data before trusting.

**d) Negative Test Expectancy**
```
Rule: if test_expectancy < -$0.25/trade AND improvement < 0:
    -> The filter loses money
```
Unless you have a strong reason to believe it's a one-time anomaly, don't deploy.

### 5. Forward-Test Before Committing

Even a result that beats the baseline on historical data should be forward-tested:

1. Deploy the new weights on **fresh, unseen trades** only (dates after the test window)
2. Collect at least `min_n` or more forward trades
3. Calculate actual expectancy on these trades
4. If forward test confirms (same direction, similar magnitude as test_expectancy),
   keep the weights; otherwise, revert or re-optimize on the new data

This is the gold standard for validating any parameter change.

### 6. Directional Asymmetry: Long ≠ Short

The JSON always provides separate results for long and short directions. **Never use
the same weights for both.** Example:
```json
"long": { "weights": { "Fast reversal (C13)": 0, "HMAGapCV consistent (C9)": 1 }, ... }
"short": { "weights": { "Fast reversal (C13)": 1, "HMAGapCV consistent (C9)": 0 }, ... }
```
Here, Fast reversal is useless for longs but crucial for shorts (and vice versa).
Applying the long weights to shorts would destroy performance.

**Action:** Update your trading engine to branch on direction and apply the correct
weights + threshold for each.

### 7. Monitor and Refresh Regularly

Market conditions change; components that were high-conviction last quarter may decay:

- Run this script **monthly or quarterly** as you accumulate more trade data
- If new runs show the same weights + better test performance, confidence increases
- If new runs show different weights or negative test results, the old pattern may be gone
- If test_n grows but test_expectancy shrinks, the edge is degrading — investigate why

### 8. Incremental Deployment Strategy

If you're risk-averse, deploy in stages:

1. **Stage 1:** Use optimized weights but keep the current threshold (conservative)
   - Trades more than the new filter but with the new component weighting
   - Monitors whether the weight changes alone help
   
2. **Stage 2:** Apply the new threshold (full deployment)
   - Restricts trades to the optimized score cutoff
   - Should hit the test_expectancy if the optimization was sound

3. **Stage 3 (if needed):** Revert quickly
   - If forward tests show degradation, revert to Stage 1 or the original formula
   - Re-optimize on the new data to understand what changed

### Example Deployment Checklist

Before updating your live trading engine, verify:

- [ ] `test_expectancy > baseline_test_expectancy`? (Beats current formula)
- [ ] `test_expectancy > 0`? (Profitable, not just better-than-current)
- [ ] `test_n >= min_n`? (Sufficient sample, not a fluke)
- [ ] `test_n` << `total_test_n`? (Filter is selective, not degenerate)
- [ ] `(train_expectancy - test_expectancy) < 2 * baseline_test_expectancy`? (Not overfit)
- [ ] `warning == null`? (No script warnings)
- [ ] Components with weight 0 removed or deprioritized in code? (Cleaner logic)
- [ ] Separate weights applied for long vs. short? (Direction-aware)

Only when all boxes are checked should you commit the new weights to live trading.
