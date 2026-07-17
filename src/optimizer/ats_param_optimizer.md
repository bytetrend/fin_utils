# ats_param_optimizer.py

Optimizes entry-filter parameters for `AtsFastReversal` / `AtsSlowReversal` trade
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

---

## The `--output` JSON: field-by-field reference

Every example below is real output from a 288-trade run (137 long / 151 short) to
make the numbers concrete instead of abstract.

### Top level

```json
{
  "csv_path": "C:\\Invest\\logs\\merged\\AtsFastReversal-merged.csv",
  "min_n": 30,
  "long": { ... },
  "short": { ... }
}
```

| Field | Meaning |
|---|---|
| `csv_path` | The trade log this report was generated from. Check this before comparing two JSON reports — a common mistake is comparing two runs that actually point to the same underlying CSV (see "Verifying an improvement" below for how to avoid this). |
| `min_n` | The `--min-n` value used for this run. **Any comparison between two JSON reports is only fair if `min_n` matches.** A looser guard (e.g. 15) will surface more aggressive, less reliable filters than a stricter one (e.g. 30). |
| `long` / `short` | Identical structure, analyzed independently. Never mix conclusions across them — a filter that helps longs can hurt shorts. |

### `long` / `short` object

```json
{
  "direction": "long",
  "baseline": { ... },
  "significance": [ ... ],
  "single_param_best": [ ... ],
  "boolean_flags": [ ... ],
  "top_combos": [ ... ]
}
```

#### `baseline`

```json
{
  "n": 137,
  "hit_rate": 0.3138,
  "expectancy": 1.1607,
  "total_pl": 159.02
}
```

| Field | Meaning |
|---|---|
| `n` | Total trades in this direction, unfiltered. This is your denominator — every `improvement` figure elsewhere in the report is relative to this baseline. |
| `hit_rate` | Fraction of trades with `Profit/Loss > 0`, unfiltered. |
| `expectancy` | Mean `Profit/Loss` per trade, unfiltered. **This is the number every filter is trying to beat.** |
| `total_pl` | Sum of `Profit/Loss` across all trades in this direction, unfiltered — equals `n * expectancy`. |

Read this first every time. In the example above, longs are already marginally
profitable ($1.16/trade) and shorts are doing better ($3.27/trade) — any filter's
job is to raise `expectancy` above these numbers without shrinking `n` so much that
the strategy rarely fires.

#### `significance` (list, sorted by ascending p-value)

```json
{
  "param": "ind_PipSpeedPct",
  "hit_median": 63.0,
  "loss_median": 72.0,
  "p_value": 0.0351,
  "n_hit": 43,
  "n_loss": 94,
  "verdict": "secondary signal (p<0.05)"
}
```

| Field | Meaning |
|---|---|
| `param` | The indicator column tested. |
| `hit_median` | Median value of this parameter across winning trades only. |
| `loss_median` | Median value of this parameter across losing trades only. |
| `p_value` | Mann-Whitney U test p-value comparing the hit and loss distributions. **This is the single most important field in the whole report** — it tells you whether a parameter genuinely separates winners from losers, independent of any specific threshold. |
| `n_hit` / `n_loss` | How many winning / losing trades had a non-missing value for this parameter — sanity-check that these aren't tiny before trusting the p-value. |
| `verdict` | A plain-English label derived from `p_value`: `"hard gate candidate (p<0.01)"`, `"secondary signal (p<0.05)"`, `"weak signal (p<0.10)"`, or `"no signal"`. |

**How to read the direction of the effect:** compare `hit_median` to `loss_median`.
In the example, hit trades have a *lower* median `PipSpeedPct` (63) than loss trades
(72) — so the parameter separates outcomes, and the useful filter direction is
"require `PipSpeedPct` to be low," not high. Always check this comparison yourself;
the `verdict` only tells you *that* there's a signal, not *which way* to set the gate.

#### `single_param_best` (list, sorted by descending `improvement`)

```json
{
  "param": "ind_PipSpeedPct",
  "direction": "le",
  "threshold": 54.5,
  "n": 31,
  "hit_rate": 0.4839,
  "expectancy": 18.58,
  "total_pl": 575.89,
  "baseline_expectancy": 1.1607,
  "baseline_n": 137,
  "improvement": 17.42
}
```

| Field | Meaning |
|---|---|
| `param` | The parameter this specific filter is built on. |
| `direction` | `"ge"` (require `param >= threshold`) or `"le"` (require `param <= threshold`). |
| `threshold` | The cutoff value found by sweeping the parameter's range to maximize expectancy. |
| `n` | How many trades pass this filter — i.e., how many of your `baseline_n` trades survive. |
| `hit_rate` | Win rate among the `n` trades that pass the filter. |
| `expectancy` | Mean `Profit/Loss` among the `n` filtered trades — the number to compare against `baseline_expectancy`. |
| `total_pl` | Sum of `Profit/Loss` among the filtered trades. Watch this alongside `expectancy` — a filter can raise per-trade expectancy while cutting `total_pl` if it discards too many trades (see worked example below). |
| `baseline_expectancy` / `baseline_n` | Copied from the direction's `baseline` block, included here for convenience so you don't have to cross-reference. |
| `improvement` | `expectancy - baseline_expectancy`. This is what the list is sorted by, and what "best" means in "single_param_best." |

**This list is a menu, not a single recommendation.** Every entry that clears
`--min-n` is included, regardless of whether its underlying parameter showed up in
`significance` with a real p-value. Cross-reference the two lists yourself — see
"Turning a result into a parameter change" below.

#### `boolean_flags` (list, sorted by descending `improvement`)

Same fields as `single_param_best`, but `direction` is always `"==1"` and
`threshold` is always `1.0` — these are your `C1`–`C13`-style hard-gate flags,
tested as "what if I additionally required this flag to be true." A flag with
`improvement <= 0` here isn't listed (the report only shows flags that beat
baseline).

#### `top_combos` (list, sorted by descending `improvement`)

```json
{
  "filters": "ind_FullDeltaATRs>=9.1 AND ind_FullAngle<=27.7273",
  "n": 30,
  "hit_rate": 0.4667,
  "expectancy": 25.45,
  "total_pl": 763.59,
  "baseline_expectancy": 1.1607,
  "improvement": 24.29
}
```

| Field | Meaning |
|---|---|
| `filters` | A human-readable string describing both conditions of a 2-parameter combination — read it directly, it's not encoded. |
| `n` / `hit_rate` / `expectancy` / `total_pl` / `baseline_expectancy` / `improvement` | Same meaning as in `single_param_best`, but for the joint (AND) filter. |

Note there's **no `baseline_n` field here** (unlike `single_param_best`) — look at
the parent direction's `baseline.n` if you need it.

**Combos are the highest-improvement, highest-overfitting-risk numbers in the
whole report.** A 2-parameter combo has two independent thresholds to tune, which
gives an exhaustive sweep many more chances to land on a subset that looks good by
chance. Always check `n` here relative to `--min-n` — a combo sitting exactly at or
just above the floor (e.g. `n=30` when `min_n=30`) is a specific warning sign, not
a coincidence to ignore; it usually means the sweep kept relaxing the thresholds
until the subset barely stopped shrinking below the guard, which is exactly the
overfitting behavior the guard exists to catch, not prevent outright.

---

## Turning a result into a parameter change

Don't take the top row of `single_param_best` or `top_combos` and wire it
straight into the live strategy. Instead, cross-reference across sections:

1. **Start from `significance`.** Only parameters with `verdict` of `"secondary
   signal"` or `"hard gate candidate"` (p < 0.05) are showing a real, general
   relationship with outcome — independent of any one threshold choice. Treat
   these as your candidate list.
2. **Find that same parameter in `single_param_best`** to get a concrete
   threshold and direction. Confirm the direction matches what you'd expect from
   comparing `hit_median` vs. `loss_median` in the significance entry — if a
   parameter is significant but the threshold direction contradicts the
   median comparison, something is off (e.g. a nonlinear relationship, or too
   few trades) and shouldn't be trusted as-is.
3. **Check `top_combos` for a pairing that includes an already-significant
   parameter.** A combo built from two non-significant parameters (as in the
   `FullDeltaATRs`/`FullAngle` example above, if neither showed p<0.05 that run)
   is the classic small-sample artifact — impressive `improvement`, no
   statistical backing. Still worth tracking, not worth deploying.
4. **Check `boolean_flags` the same way** — a `C` flag with real improvement and
   a plausible mechanism (e.g. tightening an existing hard gate) is lower-risk
   than a brand-new combo, since it's not introducing a new threshold to overfit.
5. **Look at `total_pl`, not just `expectancy`, before deciding a filter is
   worth it.** A filter that improves `expectancy` from $1/trade to $18/trade but
   cuts `n` from 137 to 31 is concentrating profit into fewer trades — check
   whether `total_pl` for that subset is still an attractive fraction of the
   baseline's `total_pl`, since a tighter filter means fewer opportunities taken
   even if each one is better on average.
6. **Translate the winning threshold into the actual strategy parameter** it
   maps to (e.g. `ind_FullDeltaATRs >= 9` → raise the live `HMinATRs` limit from
   its current value to 9; `ind_PipSpeedPct <= 54.5` → tighten `HMinPipSpeedPct`
   or add a new upper-bound gate, depending on how your engine implements it).
   The script tells you the value and direction; you still have to map the
   column name back to the live config parameter it drives.

---

## Verifying an improvement after applying a parameter change

Finding a promising number in one JSON report is not confirmation — it's a
hypothesis. Confirming it takes a second, independent step:

1. **Apply the parameter change in the live/paper strategy**, not just in this
   analysis script. The point is to test the actual trading engine's behavior,
   not just re-filter old trades (that only tells you what already happened).
2. **Let it run and accumulate a genuinely new batch of trades** — ideally
   covering a comparable amount of calendar time to the batch the finding came
   from, not just a handful of trades. A handful of new trades passing the
   filter is not a confirmation, for the same reason a `single_param_best` entry
   at `n=30` isn't reliable — you need enough trades for `expectancy` to mean
   something.
3. **Run `ats_param_optimizer.py --output` again on the new batch alone** (not
   merged with the old data) and compare directly:
   - Does `baseline.expectancy` for that direction come in higher than the old
     `baseline.expectancy`? If the parameter change worked, the *new baseline*
     (which now reflects trades taken under the new limits) should already look
     like the *old filtered* result, not the old *unfiltered* baseline.
   - Does the same parameter still show up with a `"secondary signal"` or
     better `verdict` in the new `significance` list? If it now shows `"no
     signal"`, the earlier finding likely didn't replicate.
4. **Better: use `--compare-filter` for an explicit, apples-to-apples check.**
   Run the *old* threshold against the *new* data directly:
   ```bash
   python ats_param_optimizer.py new_trades.csv \
     --compare-filter "ind_FullDeltaATRs>=9,ind_FullAngle<=27.7" \
     --direction long
   ```
   This prints the new data's baseline expectancy alongside the filtered
   expectancy for that exact rule, so you're comparing the same threshold
   across two different, independently-collected batches — the strongest
   confirmation this script family can give you.
5. **Best: use `ats_optuna_optimizer.py` with a chronological train/test
   split** instead of relying on two separate full-batch runs. It automates
   exactly this train-then-confirm-on-unseen-data pattern within a single run,
   and will explicitly warn you if training-window performance collapses on
   the held-out window — the definitive sign a finding was fit to noise
   rather than a real pattern (see `ats_optuna_optimizer.md`).
6. **Only promote a parameter change to "confirmed" once it has survived at
   least one out-of-sample check** (new data, or a held-out split) with a
   sample size at or above your `--min-n`. Until then, treat it as "candidate,"
   log it, and keep watching — don't let one good-looking JSON report alone
   change how the live strategy trades.

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
