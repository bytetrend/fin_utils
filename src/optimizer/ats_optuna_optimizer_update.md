
## Overview
`ats_optuna_optimizer.py` is a tool designed for Bayesian optimization using Optuna and Tree-structured Parzen Estimator (TPE) to find optimal threshold parameters for AtsFastReversal and AtsSlowReversal entry filters. Unlike `ats_param_optimizer.py`, which tests one or two parameters at a time, this script searches all candidate parameters jointly in a single search. It allows the optimizer to decide which parameters belong in the filter, rather than requiring manual selection of pairs to test.

## Requirements
```bash
pip install pandas numpy optuna
```

## How Overfitting is Controlled
### Read This Before Trusting Any Output
1. **Chronological Train/Test Split**: Trades are sorted by `EntryDate`. Optuna only sees the training window during optimization. The filter is evaluated on a held-out test window that comes after the training window in time. The test-window numbers are the only ones that matter.
2. **Min-n Guard**: Any candidate filter that drops the training subset below `--min-n` trades is rejected during the search.
3. **Optional k-Fold Cross-Validation**: Cross-validation within the training data can be enabled with `--cv-folds`.

## What is a Warm Start?
The `--warm-start` flag allows Optuna to leverage historical trial history rather than initializing its search completely blind. When enabled, the script pre-seeds the optimization memory with existing trade distributions and realized P/L before the active optimization begins.

## Why and When to Use It
- **Why Use It**: Standard Bayesian optimization spends its first 20-30% of trials randomly guessing parameter boundaries. Warming up the study seeds the optimizer with existing performance trends, anchoring the TPE algorithm to focus on refining profitable zones immediately.
- **When to Use It**: 
  - When working with large historical datasets (e.g., 500+ rows).
  - When running the script iteratively inside an automated agent framework to save local computation resources.

## Basic Usage
```bash
python ats_optuna_optimizer.py trades.csv --warm-start
```

## Options
| Flag | Default | Description |
|------|---------|-------------|
| `--warm-start` | `False` | Boolean flag to pre-seed the optimization memory with historical data. |
| `--min-n` | `30` | Minimum trades required in any filtered training subset (overfitting guard). |
| `--n-trials` | `1000` | Number of Optuna trials to run per direction. |
| `--test-fraction` | `0.25` | Fraction of trades held out as the test set. |
| `--cv-folds` | `1` | Number of cross-validation folds (off by default). |
| `--params` | `auto-detect` | Comma-separated list of parameter columns to search over. |
| `--direction` | `none` | Restrict optimization to only long or short trades. |
| `--seed` | `42` | Random seed for reproducibility. |
| `--output` | `none` | Write the full structured report as JSON. |
| `--min-pattern-score` | `none` | Enables entry-path stratification. |
| `--min-cvd-score` | `none` | Companion to `--min-pattern-score`. |
| `--pattern-score-col` | `auto-detect` | Explicit PatternEntryScore column name. |
| `--cvd-score-col` | `auto-detect` | Explicit CVDEntryScore column name. |

## Examples
### Run a Warm-Started Optimization
```bash
python ats_optuna_optimizer.py trades.csv --warm-start --n-trials 2000 --min-n 30
```

### Use Cross-Validation
```bash
python ats_optuna_optimizer.py trades.csv --warm-start --test-fraction 0.3 --cv-folds 5
```

### Restrict Parameters
```bash
python ats_optuna_optimizer.py trades.csv --warm-start --params ind_FullDeltaATRs,ind_FullAngle,ind_ATRsFromHma
```

### Optimize for Long Trades Only
```bash
python ats_optuna_optimizer.py trades.csv --warm-start --direction long
```

### Save Results as JSON
```bash
python ats_optuna_optimizer.py trades.csv --warm-start --output optuna_report.json
```

## What the Output Looks Like
For each direction (LONG / SHORT):
- **Training window**: Metrics on the training data (data Optuna saw during optimization).
- **Test window**: Metrics on the held-out data (data Optuna never saw).

Example output:
```
Loaded 620 trades from trades.csv
Long: 315
Short: 305
Searching over 4 parameters: ['ind_ATRsFromHma', 'ind_RevATRsPerSec', 'ind_HMAGapCV', 'ind_ATRsFromHma']
Train/test split: 75%/25% chronological, 1000 trials, cv_folds=1

LONG
Training window: 236 trades total, baseline_expectancy=$-8.06
Test window: 79 trades total, baseline_expectancy=$31.94

Optuna-found filter:
ind_ATRsFromHma<=0.555848
ind_RevATRsPerSec<=0.54032
ind_HMAGapCV<=2.17527

TRAIN performance:
n=86
hit_rate=44.2%
expectancy=$18.30/trade

TEST performance:
n=24
hit_rate=41.5%
expectancy=$14.16/trade
total_pl=$339.84

[Important: Only the TEST performance block is trustworthy.]
```

## How to Interpret the Results
- **Only trust the TEST performance block**.
- A large drop from training expectancy to test expectancy indicates overfitting.
- `[LOW CONFIDENCE (n=X < min-n=Y)]` in the test line means the filter may not be reliable.
- A filter with test expectancy close to or better than training expectancy, with adequate test-window n, is a strong signal.

## Note on Small Datasets
With small datasets, the test window may be too small to draw reliable conclusions. Increasing `--test-fraction` can improve reliability at the cost of a noisier training search.

## JSON Output File Structure (`optuna_report.json`)
### Top-Level Parameters
- `csv_path`: Path to the trades CSV analyzed.
- `min_n`: Minimum trades threshold used.
- `n_trials`: Number of Optuna trials performed per direction.
- `test_fraction`: Fraction of trades held out as test set.
- `cv_folds`: Number of cross-validation folds.

### Direction Results (long / short)
Each direction contains independent optimization results.

### Clauses Array
- Format: `"parameter_name OPERATOR threshold"`
- Example: `"ind_Interval<=43.7966"`
- All conditions must be satisfied (AND logic).

### Training Window Metrics
- `total_train_n`: Total trades available for training.
- `train_n`: Trades matching the filter in the training window.
- `train_expectancy`: Average $/trade for filtered training trades.
- `train_hit_rate`: Win rate (%) for filtered training trades.

### Test Window Metrics (Primary for Deployment)
- `total_test_n`: Total trades available in test window.
- `test_n`: Trades matching the filter in the test window.
- `test_expectancy`: Average $/trade for filtered test trades.
- `test_hit_rate`: Win rate (%) for filtered test trades.
- `test_total_pl`: Total P&L from all test trades matching the filter.

### Baseline Metrics
- `baseline_train_expectancy`: Average $/trade across all training trades.
- `baseline_test_expectancy`: Average $/trade across all test trades.

### Optimization Metadata
- `n_trials_run`: Number of Optuna trials actually executed.
- `warning`: String or null indicating potential issues.

## Understanding Multi-Parameter Filters
- Optuna searches all parameters jointly, capturing interaction effects.
- Example comparison:
  - Single-parameter approach: Best filter may be too loose.
  - Multi-parameter approach: Best filter may be tighter and more selective.

## Clause Count & Filter Tightness
| Clause Count | Tightness | Selectivity | Use Case                    |
|--------------|-----------|-------------|-----------------------------|
| 1-2          | Loose     | 70-90%      | High-volume strategies      |
| 3-5          | Moderate  | 40-70%      | Balanced                    |
| 6+           | Tight     | 20-40%      | Quality-focused, low volume |

## Interpreting Results for Deployment
### Red Flags
1. **Test Expectancy Below Baseline**: Filter makes things worse.
2. **Small Test Window**: Need more data.
3. **Large Train-to-Test Gap**: Likely overfitting.
4. **Filter Never Fired**: Filter is too tight.

### Green Flags
1. **Strong Test Expectancy with Large Sample**: Trustworthy results.
2. **Stable Train-to-Test Performance**: Filter holds up on unseen data.
3. **Meaningful Improvement Over Baseline**: Positive improvement.

## How to Use Clauses to Improve Profitability
### Step 1: Extract the Clauses
From the JSON output:
```json
"long": {
  "clauses": [
    "ind_Interval<=43.7966",
    "ind_CVDDeltaPct<=22.6168",
    "ind_Angle<=27.856"
  ]
}
```

### Step 2: Verify Performance Metrics
- Check if `test_expectancy` is greater than `baseline_test_expectancy`.
- Ensure `test_n` is adequate.
- Confirm `test_expectancy` is positive.
- Check for warnings.

### Step 3: Implement in Trading Engine
Before:
```python
IF entry_signal THEN
    take_trade()
```

After:
```python
IF entry_signal AND
    ind_Interval <= 43.7966 AND
    ind_CVDDeltaPct <= 22.6168 AND
    ind_Angle <= 27.856 THEN
    take_trade()
```

### Step 4: Separate Long vs Short
```python
IF direction == "long" AND (all long clauses pass) THEN
    take_long_trade()
ELSE IF direction == "short" AND (all short clauses pass) THEN
    take_short_trade()
```

### Step 5: Monitor and Validate Live
- Track new trades post-test window.
- Calculate actual expectancy.
- Compare to test_expectancy.

### Step 6: Decision Matrix
| Condition                                     | Confidence | Action                        |
|-----------------------------------------------|------------|-------------------------------|
| test_exp > baseline × 1.5 AND test_n ≥ 20     | Very High  | Deploy immediately            |
| test_exp > baseline × 1.5 AND 5 ≤ test_n < 20 | High       | Deploy + forward-test         |
| test_exp > baseline × 1.2 AND test_n ≥ 10     | Moderate   | Deploy cautiously             |
| test_exp > baseline AND test_n < 10           | Low        | Deploy + intensive monitoring |
| test_exp ≤ baseline                           | None       | Reject; re-optimize           |

## Troubleshooting
### Q: I have 13 clauses but only a few trades pass the filter (test_n is very small)
**A**: Increase `--test-fraction` or `--min-n`.

### Q: My clauses are all ">= " (lower bounds), no "<=" (upper bounds). Is that a problem?
**A**: No. It is valid if the data supports it.

### Q: Should I manually adjust the clause thresholds?
**A**: No. Use the exact thresholds from the JSON.

### Q: Can I combine clauses from long and short filters?
**A**: No. Long and short trades respond to different market mechanics.

### Q: How often should I re-run Optuna optimization?
**A**: Monthly or after every 100+ new trades.

## Additional Resources
- **Optuna Documentation**: [https://optuna.readthedocs.io](https://optuna.readthedocs.io)
- **TPE Algorithm**: [https://arxiv.org/abs/1111.4246](https://arxiv.org/abs/1111.4246)
- **GitHub Repository**: [https://github.com/your-repository](https://github.com/your-repository)

## Conclusion
`ats_optuna_optimizer.py` is a powerful tool for optimizing trading strategies by jointly searching all candidate parameters. By following the steps outlined in this document, you can effectively deploy and monitor optimized filters to improve trading performance.