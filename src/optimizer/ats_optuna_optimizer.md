# ats_optuna_optimizer.py

Bayesian (Optuna/TPE) threshold optimization for `AtsFastReversal` /
`AtsSlowReversal` entry-filter parameters.

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

## JSON Output File Structure (optuna_optimized.json)

When using the `--output` flag, the script produces a JSON file with the Optuna
optimization results directly. This is the definitive output of the multi-parameter
search and the file you should use for live deployment.

### Top-Level Parameters

| Parameter | Type | Example | Description |
|---|---|---|---|
| `csv_path` | string | "C:\\logs\\AtsFastReversal-merged.csv" | Path to the trades CSV analyzed |
| `min_n` | integer | 30 | Minimum trades threshold used (overfitting guard) |
| `n_trials` | integer | 1000 | Number of Optuna trials performed per direction |
| `test_fraction` | float | 0.25 | Fraction of trades held out as test set (0.25 = 25%) |
| `cv_folds` | integer | 1 | K-fold cross-validation folds (1 = no cross-validation) |
| `long` | object | {...} | Optimization results for long trades |
| `short` | object | {...} | Optimization results for short trades |

### Direction Results (long / short)

Each direction contains the same structure with independent optimization results.

#### Clauses Array (The Filter)

| Field | Type | Example | Description |
|---|---|---|---|
| `clauses` | array | ["ind_Interval<=43.7966", "ind_CVDDeltaPct<=22.6168", ...] | **The multi-parameter filter to deploy.** Each element is a condition (param + direction + threshold). All conditions must be satisfied (AND logic). This is the exact filter to implement in your trading engine. |

**Understanding clause format:**
- Format: `"parameter_name OPERATOR threshold"`
- Operators: `>=` (greater-than-or-equal), `<=` (less-than-or-equal)
- Examples:
  - `"ind_BarATR>=0.177"` means keep trades where ind_BarATR ≥ 0.177
  - `"ind_Angle<=27.8"` means keep trades where ind_Angle ≤ 27.8
- **All clauses must match** (AND conditions): filter matches only trades passing every single clause

**How to interpret a multi-clause filter:**
```json
"clauses": [
  "ind_Interval<=43.7966",
  "ind_CVDDeltaPct<=22.6168",
  "ind_Angle<=27.856",
  "ind_PipSpeedNorm<=17.0965"
]
```
**Means:** Keep trades only if:
- ind_Interval ≤ 43.7966 **AND**
- ind_CVDDeltaPct ≤ 22.6168 **AND**
- ind_Angle ≤ 27.856 **AND**
- ind_PipSpeedNorm ≤ 17.0965

If ANY condition fails, the trade is filtered out.

#### Training Window Metrics

These metrics show performance on the training window (data Optuna saw during optimization).
**Do not use these for deployment decisions—they're optimistic by design.**

| Field | Type | Example | Description |
|---|---|---|---|
| `total_train_n` | integer | 107 | Total trades available for training (before filter) |
| `train_n` | integer | 30 | Trades matching the filter within the training window |
| `train_expectancy` | float | 27.45 | Average $/trade for filtered training trades |
| `train_hit_rate` | float | 0.433 | Win rate (%) for filtered training trades |

**Training metrics interpretation:**
- High train_expectancy is EXPECTED because Optuna fit to this data
- Compare train_expectancy to baseline_train_expectancy to see filter improvement
- Comparison example:
  - baseline_train_expectancy: $0.56/trade (all trades, no filter)
  - train_expectancy: $27.45/trade (with filter)
  - Improvement: $26.89/trade (very attractive but may not persist)

#### Test Window Metrics (PRIMARY FOR DEPLOYMENT)

**These metrics are from the held-out test window that Optuna NEVER saw.**
**Only these numbers matter for assessing real-world performance.**

| Field | Type | Example | Description |
|---|---|---|---|
| `total_test_n` | integer | 36 | Total trades available in test window (before filter) |
| `test_n` | integer | 5 | Trades matching the filter in the test window |
| `test_expectancy` | float | 23.02 | Average $/trade for filtered test trades ⭐ PRIMARY METRIC |
| `test_hit_rate` | float | 0.60 | Win rate (%) for filtered test trades |
| `test_total_pl` | float | 115.11 | Total P&L from all test trades matching filter |

**Test metrics interpretation:**
- **test_expectancy is your projected live P&L per trade** (if market conditions remain similar)
- **test_n < min_n triggers a warning** — small sample sizes are unreliable
- **test_expectancy vs baseline_test_expectancy** shows real improvement:
  - baseline_test_expectancy: $5.35/trade (all test trades, no filter)
  - test_expectancy: $23.02/trade (with filter)
  - Improvement: $17.67/trade (strong edge, likely to persist)
- **test_total_pl = test_n × test_expectancy** (sanity check)

#### Baseline Metrics (For Comparison)

These represent "all trades with no filter"—your control group.

| Field | Type | Example | Description |
|---|---|---|---|
| `baseline_train_expectancy` | float | 0.56 | Average $/trade across ALL training trades (no filter) |
| `baseline_test_expectancy` | float | 5.35 | Average $/trade across ALL test trades (no filter) |

**Baseline interpretation:**
- These are your starting point (no filter applied)
- **improvement = test_expectancy - baseline_test_expectancy**
  - Improvement > $5/trade: strong
  - Improvement > $0: candidate
  - Improvement ≤ 0: filter makes things worse
- If baseline_test_expectancy is already strong (> $5/trade), you already have a good strategy; look for small improvements

#### Optimization Metadata

| Field | Type | Example | Description |
|---|---|---|---|
| `n_trials_run` | integer | 1000 | Number of Optuna trials actually executed |
| `warning` | string or null | "Held-out test subset only has 5 trades (<--min-n 30) -- low confidence." | Alert if result is unreliable (null if no warning) |

**Warnings to understand:**
- `null` (no warning): Result is reliable
- "Held-out test subset only has X trades (< --min-n Y) -- low confidence.": Test set too small
- "Filter matched ZERO trades in the held-out test window": Filter never fired on test data
- "No held-out test data available": Test fraction too small or dataset too small

## Understanding Multi-Parameter Filters

### Why Multiple Clauses Matter

Instead of searching one parameter at a time (like `ats_param_optimizer.py`), Optuna searches
**all parameters jointly** and discovers multi-parameter combinations:

**Example comparison:**

Single-parameter approach:
```
Best: ind_BarATR >= 0.177  →  expectancy $27/trade
```

Multi-parameter Optuna approach:
```
Best: ind_BarATR >= 0.177  AND  ind_AvgATR >= 0.12  AND  ind_RevATRsPerSec <= 0.427
 →  expectancy $32/trade (better because parameters interact)
```

Optuna can capture **interaction effects**—where two parameters together are more selective
than either alone.

### Clause Count & Filter Tightness

| Clause Count | Tightness | Selectivity | Use Case |
|---|---|---|---|
| 1–2 | Loose | 70–90% of trades pass | High-volume strategies |
| 3–5 | Moderate | 40–70% of trades pass | Balanced |
| 6+ | Tight | 20–40% of trades pass | Quality-focused, low volume |

**Interpretation example (AtsFastReversal long):**
- 13 clauses, test_n=5 out of total_test_n=36
- Filter selectivity: 5/36 ≈ 14% of trades pass
- This is very tight; high quality but few opportunities

## Interpreting Results for Deployment

### Red Flags: When NOT to Deploy

**1. Test Expectancy Below Baseline**
```
baseline_test_expectancy: $5.35/trade
test_expectancy: $3.20/trade
```
Filter makes things WORSE. Do not deploy.

**2. Small Test Window (test_n < min_n)**
```
test_n: 5
min_n: 30
warning: "Held-out test subset only has 5 trades (< --min-n 30) -- low confidence."
```
Result is anecdotal. Need more data to confirm.

**3. Large Train-to-Test Gap**
```
train_expectancy: $27.45/trade
test_expectancy: $23.02/trade
Gap: $4.43 (16% drop)
```
Minor overfitting, acceptable. But if gap > 50%, likely overfit.

**4. Filter Never Fired on Test Data (test_n = 0)**
```
test_n: 0
warning: "Filter matched ZERO trades in the held-out test window -- cannot confirm."
```
Filter is too tight; no validation possible.

### Green Flags: When to Deploy

**✓ Strong Test Expectancy with Large Sample**
```
test_expectancy: $23.02/trade
test_n: 5 (but baseline has many trades, so filter is highly selective)
improvement: $17.67/trade vs baseline $5.35/trade
```
Even with small n, improvement is strong.

**✓ Stable Train-to-Test Performance**
```
train_expectancy: $27.45/trade
test_expectancy: $23.02/trade
Gap: ~16% (normal overfitting)
```
Filter holds up on unseen data; likely real pattern.

**✓ Meaningful Improvement Over Baseline**
```
improvement = test_expectancy - baseline_test_expectancy
improvement > $3-5/trade: Deploy with confidence
improvement > $0: Deploy cautiously
```

## How to Use Clauses to Improve Profitability

### Step 1: Extract the Clauses

From the JSON output, find the `"clauses"` array for your direction:

```json
"long": {
  "clauses": [
    "ind_Interval<=43.7966",
    "ind_CVDDeltaPct<=22.6168",
    "ind_Angle<=27.856",
    ...
  ]
}
```

### Step 2: Verify Performance Metrics

Before implementing, check:
- **test_expectancy > baseline_test_expectancy?** (profitable improvement)
- **test_n >= 5?** (at least some validation data)
- **test_expectancy > 0?** (profitable on test set)
- **warning == null?** (no disqualifying flags)

**Decision logic:**
```
IF test_expectancy > baseline_test_expectancy AND test_n >= 10:
    Deploy filter (high confidence)
ELSE IF test_expectancy > baseline_test_expectancy AND test_n >= 5:
    Deploy filter (moderate confidence)
ELSE IF test_n < min_n:
    Deploy cautiously; forward-test on new data
ELSE:
    Reject; re-optimize with more data
```

### Step 3: Implement in Trading Engine

Update your strategy's entry logic with AND conditions:

**Before (no filter):**
```
IF entry_signal THEN
    take_trade()
```

**After (with Optuna clauses):**
```
IF entry_signal AND
   ind_Interval <= 43.7966 AND
   ind_CVDDeltaPct <= 22.6168 AND
   ind_Angle <= 27.856 AND
   ... (all clauses)
THEN
    take_trade()
```

### Step 4: Separate Long vs Short

The JSON provides separate filters for each direction. **Apply the correct filter based on direction:**

```
IF direction == "long" AND
   (all long clauses pass) THEN
    take_long_trade()
    
IF direction == "short" AND
   (all short clauses pass) THEN
    take_short_trade()
```

### Step 5: Monitor and Validate Live

Track actual performance on **NEW trades only** (after the test window date):

1. Collect at least 20+ new trades with the filter
2. Calculate actual expectancy: `actual_pl / count`
3. Compare to test_expectancy:
   - **Within ±20%?** Filter is performing as expected; keep it
   - **Worse than projected by >20%?** Market conditions changed; re-optimize
   - **Better than projected?** Bonus; confidence in filter increases

### Step 6: Decision Matrix

Use this matrix to decide deployment confidence:

| Condition | Confidence | Action |
|---|---|---|
| test_exp > baseline × 1.5 AND test_n ≥ 20 | Very High | Deploy immediately |
| test_exp > baseline × 1.5 AND 5 ≤ test_n < 20 | High | Deploy + forward-test |
| test_exp > baseline × 1.2 AND test_n ≥ 10 | Moderate | Deploy cautiously |
| test_exp > baseline AND test_n < 10 | Low | Deploy + intensive monitoring |
| test_exp ≤ baseline | None | Reject; re-optimize |

### Step 7: Clause Optimization Strategy

If results aren't satisfactory, consider re-running with adjustments:

| Issue | Solution |
|---|---|
| Filter too loose (test_n close to total_test_n) | Lower `--min-n` or increase `--n-trials` to find tighter filter |
| Filter too tight (test_n = 0) | Increase `--test-fraction` to get more test data; raise `--min-n` threshold |
| Poor test performance (test_exp ≈ baseline) | Increase `--n-trials` (2000+), enable `--cv-folds 5`, check data quality |
| Overfitting (large train-test gap) | Enable `--cv-folds 3-5` to penalize filters that only work on lucky training subsets |

## Real-World Example: AtsFastReversal Long Direction

Given this JSON output:
```json
{
  "direction_label": "long",
  "clauses": [
    "ind_Interval<=43.7966",
    "ind_CVDDeltaPct<=22.6168",
    ...
  ],
  "total_train_n": 107,
  "total_test_n": 36,
  "train_n": 30,
  "train_expectancy": 27.45,
  "train_hit_rate": 0.433,
  "test_n": 5,
  "test_expectancy": 23.02,
  "test_hit_rate": 0.60,
  "test_total_pl": 115.11,
  "baseline_train_expectancy": 0.56,
  "baseline_test_expectancy": 5.35,
  "n_trials_run": 1000,
  "warning": "Held-out test subset only has 5 trades (< --min-n 30) -- low confidence."
}
```

**Analysis:**

**✓ Strengths:**
- test_expectancy ($23.02/trade) >> baseline_test_expectancy ($5.35/trade)
- improvement = $17.67/trade (331% better!)
- test_hit_rate = 60% (majority winners)
- test_total_pl = $115.11 (real money, not anecdotal)

**⚠️ Weaknesses:**
- test_n = 5 is small (< min_n = 30)
- Warning: "low confidence"
- train_expectancy ($27.45) is higher than test_expectancy ($23.02), suggesting 15% overfitting

**Decision: Deploy with Forward-Testing**

The improvement is so strong ($17.67/trade) that even with only 5 test trades, deployment is justified **provided you monitor heavily** on new data:
1. Deploy the 13-clause filter
2. Collect next 20+ long trades matching the filter
3. Calculate actual expectancy
4. If actual ≥ $15/trade, confidence increases; keep filter
5. If actual < $10/trade, pause; re-optimize on expanded data

## Troubleshooting

**Q: I have 13 clauses but only a few trades pass the filter (test_n is very small)**
A: This is normal for tight filters. If test_n = 0, the filter is too strict for your test window size. Either:
   - Increase --test-fraction (0.3 or 0.4) to get more test data
   - Accept the small sample and forward-test more intensively
   - Try re-running with higher --min-n to force the search toward less stringent filters

**Q: My clauses are all ">= " (lower bounds), no "<=" (upper bounds). Is that a problem?**
A: No. Optuna discovered that only lower-bound conditions help for your data. This is valid.
Each parameter is independent; some will have >= and some <=.

**Q: Should I manually adjust the clause thresholds to be "nicer" numbers (e.g., 43.7966 → 44)?**
A: No. Use the exact thresholds from the JSON. They're optimized for your data.
Rounding will slightly hurt performance. If you must round, test on recent data first.

**Q: Can I combine clauses from long and short filters?**
A: No. Keep them separate. Long and short trades respond to different parameters; mixing them will destroy performance.

**Q: What if I want to mix Optuna clauses with my own custom parameters?**
A: Possible but risky. The clauses are already optimized as a package. Adding external conditions will:
   - Reduce trade frequency (fewer clauses will pass)
   - Likely reduce profitability (clauses interact; breaking the set hurts performance)
Better approach: Re-run Optuna with your custom parameters included.

**Q: How often should I re-run Optuna optimization?**
A: Monthly or after every 100+ new trades. If the new run produces similar clauses with similar thresholds, confidence in the edge increases. If thresholds shift significantly, market conditions may have changed; adopt the new filter.
