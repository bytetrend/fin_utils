# ats_param_optimizer.py

Optimizes entry-filter parameters for `AtsPriceQuickReversal` / `AtsPriceBrkout` trade
logs and recommends parameter changes, using the same methodology as the strategy's
own documented analysis rules.

## Requirements

```bash
pip install pandas numpy scipy
```

## Methodology

- **Outcome definition:** `ProfitHit = Profit/Loss > 0`. All exit types are included
  (e.g. a `StpLoss` exit with P/L > 0 counts as a win, not a loss).
- **Direction split:** Long and short trades (`ind_SignalSent` = 1 / -1) are always
  analyzed separately — they respond to different parameters.
- **Significance test:** Mann-Whitney U, comparing the parameter's values in the
  ProfitHit group vs. the PureLoss group.
- **Optimization target:** Expectancy (`mean(Profit/Loss)`) over the filtered trade
  set — not win rate alone, since a filter can raise win rate while still losing
  money, or vice versa.
- **Overfitting guard:** No filtered subset is reported unless it has at least
  `--min-n` trades (default 30).
- **Single-parameter AND combination sweeps:** Both are run, since real edges (e.g.
  `FullDeltaATRs` + `FullAngle` for longs) are often combinations, not single gates.
- **Confidence labeling:** Every recommendation is tagged with its sample size and
  significance. The script does not silently promote an unproven small-sample combo
  to "recommended" — that judgment is left visible in the report as HIGH vs. LOW
  confidence.

## Basic usage

```bash
python ats_param_optimizer.py trades.csv
```

Loads the CSV, splits long/short, and prints:
1. Statistical significance ranking per parameter (Mann-Whitney U)
2. Best single-parameter threshold (ranked by expectancy gain over baseline)
3. Boolean condition flags (`C1`–`C13` style) tested as "require flag == 1"
4. Best 2-parameter combinations
5. A summary recommendation block per direction, with a confidence label

## Options

| Flag | Default | Description |
|---|---|---|
| `--min-n N` | 30 | Minimum trades required in any filtered subset (overfitting guard) |
| `--top-n N` | 8 | How many top single-parameters feed into the combo search / how many rows print per section |
| `--output PATH` | none | Write the full structured report as JSON to `PATH` |
| `--compare-filter "..."` | none | Evaluate a specific filter string instead of (or alongside) the full sweep |
| `--direction {long,short,both}` | both | Restrict `--compare-filter` evaluation to one direction |

## Examples

Run the full sweep with a stricter sample-size guard and more candidates reported:

```bash
python ats_param_optimizer.py trades.csv --min-n 30 --top-n 8
```

Save the full structured results for later comparison (e.g. checking whether a
result replicates on a new batch of trades):

```bash
python ats_param_optimizer.py trades.csv --output report.json
```

Forward-test a specific, already-hypothesized filter against a dataset (useful for
checking whether a parameter finding from one dataset replicates on another,
out-of-sample dataset):

```bash
python ats_param_optimizer.py trades.csv \
  --compare-filter "ind_FullDeltaATRs>=9,ind_FullAngle>=26" \
  --direction long
```

`--compare-filter` accepts comma-separated clauses using `>=`, `<=`, `==`, `>`, or `<`,
e.g.:

```bash
--compare-filter "ind_ATRsFromHma>=0.8"
--compare-filter "ind_FullDeltaATRs>=9,ind_FullAngle>=26,ind_HMAGapCV<=1.5"
```

## Interpreting the output

- **"no signal" / p >= 0.10:** The parameter shows no reliable difference between
  winning and losing trades in this dataset. Treat any threshold built on it as
  exploratory only.
- **"weak signal" (p < 0.10) / "secondary signal" (p < 0.05) / "hard gate candidate"
  (p < 0.01):** Increasing confidence that the parameter genuinely separates
  winners from losers.
- **Confidence: LOW (small-sample, unconfirmed)** in the recommendations section
  means the best combo/threshold found relies on a subset that passed the `--min-n`
  guard but isn't backed by a significant single-parameter signal. These are worth
  tracking on future data, not wiring into a live strategy yet.
- **Confidence: HIGH** means the best result includes a parameter that also showed
  up as statistically significant on its own.

## A note on small samples

With a dataset in the low hundreds of trades (which is typical for a single batch
of live/paper trades), most single parameters will not reach significance, and any
"best" 2-parameter combo found by an exhaustive sweep is likely to be a small-sample
artifact — by construction, searching many possible thresholds will always turn up
something that looks good on the data it was searched on. The right use of this
script on a small file is less "find new parameters" and more "check whether a
parameter finding from a bigger dataset still holds up" via `--compare-filter`.
Save each run's `--output` JSON so results can be compared across datasets over time
before anything gets promoted into the live strategy's parameter set.
