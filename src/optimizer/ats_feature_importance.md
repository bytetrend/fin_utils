# ats_feature_importance.py

Ranks which entry-filter parameters matter most for `AtsFastReversal` /
`AtsSlowReversal` trade outcomes, using cross-validated classifier feature
importance (Random Forest or Gradient Boosting + permutation importance) and,
if `shap` is installed, an independent SHAP cross-check.

This is a different tool than the threshold-sweep scripts
(`ats_param_optimizer.py`, `ats_optuna_optimizer.py`). Those search for the
specific threshold or combination that maximizes expectancy. This script asks
a more basic question first: **does this parameter carry any real information
about win/loss at all, and how much, relative to the others?** Run this one
first — if a parameter shows near-zero importance here, a threshold found for
it by the other scripts is very likely noise, and you can narrow their search
to only the parameters that actually matter.

## Requirements

```bash
pip install pandas numpy scikit-learn
pip install shap   # optional, for the independent interaction-aware cross-check
```

## Honesty guardrails baked in

- **Reports cross-validated ROC-AUC first**, before any importance ranking.
  If AUC is close to 0.5, the model found no real predictive edge over
  guessing, and the script says explicitly that the ranking below it is not
  meaningful.
- **AUC below 0.5 is flagged as noise, not an inverse signal.** With small
  per-fold sample sizes, an AUC under 0.5 is what sampling noise looks like —
  it is not evidence the parameters predict the opposite of what you'd
  expect, and the script's output says so directly rather than implying an
  "inverted" filter would work.
- **Uses permutation importance**, measured out-of-fold, not raw
  impurity-based importance — impurity importance is biased toward
  high-cardinality continuous columns regardless of whether they're actually
  predictive.
- **Long and short are always modeled and reported separately.**
- **Small-sample guard**: a direction with fewer than 60 trades is skipped
  entirely with an explicit message, rather than fitting an unstable model
  and reporting numbers that don't mean anything.

## Basic usage

```bash
python ats_feature_importance.py trades.csv
```

Loads the CSV, splits long/short, fits a cross-validated Random Forest
classifier per direction to predict win/loss from the numeric `ind_*`
columns, and reports AUC plus a ranked importance table.

## Options

| Flag | Default | Description |
|---|---|---|
| `--model {random_forest,gradient_boosting}` | random_forest | Classifier used for importance ranking |
| `--cv-folds N` | 5 | Number of cross-validation folds |
| `--params "a,b,c"` | auto-detect | Comma-separated list of parameter columns to rank, instead of every numeric `ind_*` column |
| `--seed N` | 42 | Random seed for reproducibility |
| `--no-shap` | off | Skip SHAP computation even if the package is installed |
| `--output PATH` | none | Write the full structured report as JSON |

## Examples

Use gradient boosting instead of random forest, with more folds:

```bash
python ats_feature_importance.py trades.csv --model gradient_boosting --cv-folds 5
```

Rank only a specific set of parameters (e.g. ones you're considering adding
as new gates):

```bash
python ats_feature_importance.py trades.csv \
  --params ind_FullDeltaATRs,ind_FullAngle,ind_ATRsFromHma
```

Save the full structured results:

```bash
python ats_feature_importance.py trades.csv --output importance_report.json
```

## What the output looks like

```
SHORT  (n=60, baseline hit rate=28.3%)
Model: random_forest   Cross-validated AUC: 0.612 (+/- 0.151)
  (0.50 = no better than guessing, 1.00 = perfect separation)

Parameter importance ranking (cross-validated permutation importance,
drop in ROC-AUC when the column is shuffled -- higher = more important):
Parameter                   Importance  Direction of effect
ind_FullDeltaATRs               0.0640  higher = more likely to win
ind_RevATRsPerSec               0.0296  higher = more likely to win
...
ind_ATRsFromHma                -0.0406  lower = more likely to win

SHAP mean |impact| (independent cross-check, also captures interactions):
  ind_FullDeltaATRs             0.0444
  ind_FullAngle                 0.0314
  ...
```

If a direction doesn't have enough trades:

```
LONG  (n=55, baseline hit rate=23.6%)
  SKIPPED: Only 55 trades (< 60) -- too few to fit a stable model. Skipping.
  Gather more trades before modeling this direction.
```

## How to interpret the results

1. **Check the AUC line first, always.**
   - **~0.50:** No real signal. The importance ranking below is noise —
     don't act on the top-ranked parameter. The script will say this
     explicitly.
   - **Notably below 0.50:** Also not a signal to trust — this is what
     small-sample noise looks like, not an inverse relationship. The
     script flags this case with its own explicit warning.
   - **Meaningfully above 0.50** (roughly 0.55+, more convincing the higher
     it goes): The model found real structure. The ranking below is worth
     reading.

2. **Read the importance ranking as a relative ordering, not an absolute
   score.** The numbers are the drop in AUC when that column's values are
   shuffled — bigger drop means the model relied on it more. A parameter
   near 0.0000 contributed essentially nothing.

3. **"Direction of effect" tells you which way to move a threshold** (raise a
   `≥` gate or lower a `≤` gate), based on comparing the parameter's median
   value in winning vs. losing trades. **It does not tell you the right
   threshold value** — that's what `ats_optuna_optimizer.py` or
   `ats_param_optimizer.py` are for. Feed the top few parameters from this
   ranking into their `--params` list instead of searching every column.

4. **Cross-check with SHAP when available.** If the SHAP ranking roughly
   agrees with the permutation importance ranking, that's a second
   independent method reaching the same conclusion — meaningfully more
   trustworthy than either alone. If they disagree sharply, treat the
   result with more caution; it may indicate an interaction effect or an
   unstable fit.

5. **A `SKIPPED` direction means "no verdict," not "no edge."** It's a
   sample-size limitation, not a finding that the parameters don't matter for
   that direction. Re-run once more trades are available.

## Recommended workflow with the other scripts

1. Run `ats_feature_importance.py` first on your full trade log.
2. Only for directions where AUC shows real lift, take the top few ranked
   parameters.
3. Feed just those into `ats_optuna_optimizer.py --params ...` instead of
   letting it search all columns — a narrower, better-justified search space
   makes the train/test result more trustworthy, especially on small
   datasets.
4. Only change a live parameter once a filter has survived the held-out test
   window in `ats_optuna_optimizer.py` with an adequate sample size — not
   from the importance ranking alone.
