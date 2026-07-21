# AtsFastReversal & AtsSlowReversal — Strategy Reference

This document describes how these two strategies work, defines every
parameter discussed while tuning them, explains how to use the Python
optimization toolchain built alongside them, and catalogs the nuances and
outright gotchas discovered along the way — most importantly with
`HMAGapCV`.

---

## 1. How the strategies work

Both strategies are reversal-family systems built on the same skeleton: a
fast/slow Hull Moving Average (HMA) pair, a set of boolean "C" condition
flags, two weighted composite scores, and a final entry gate that combines
them. They differ in *what kind* of reversal they're built to catch.

### AtsFastReversal
Requires a **strong, already-extended** reversal — the strategy wants proof
that the preceding move was genuinely fast and forceful before betting
against it. Concretely, this shows up as: `ATRsFromHma` (distance from the
HMA, in ATR units) is a significant, confirmed predictor for both long and
short; and `PipSpeedTrendPct` needs to be **high** (the preceding trend leg
was uniformly fast, not petering out) — the opposite requirement from Slow.

### AtsSlowReversal
Catches a move that has **already begun turning** — less extension is
required, because the strategy is looking for early signs of exhaustion
rather than a violent snap-back. `ATRsFromHma` is *not* a significant
predictor here (by design — that's what "already turning" means). Instead,
`TrendBarCount` (shorter preceding trend = better) and `HMAGapCV`
(consistency of the HMA gap) carry the real signal, and `PipSpeedTrendPct`
needs to be **low** (the preceding move was already decelerating).

Same underlying indicators, opposite calibration — this pattern (one
indicator, two strategies, two different "good" directions) shows up more
than once below, and is one of the most important things to keep in mind
when tuning either strategy.

### The entry mechanism

Both strategies enter on the same structural logic:

```
PatternEntryScore = IFF(C5, 2, 0)                                    // Speed flip
                   + IFF(PipSpeedTrendPct >= HighPipSpeedTrendPct, 1, 0)
                   + IFF(C14, 3, 0)                                  // Trend bar pattern
                   + IFF(C15, 3, 0)                                  // Trend bar pattern

CVDEntryScore = IFF(C6, 2, 0)                                        // CVDSpeedPct
               + IFF(C10, 4, 0)                                      // CVD accel last 2 bars
               + IFF(C13, 1, 0)                                      // CVDDeltaPct confirms
               + IFF(C11, 2, 0)                                      // ATRsFromHma bar expansion

C3  = PatternEntryScore >= MinPatternEntryScore
C12 = CVDEntryScore     >= MinCVDEntryScore

Entry: If (C3 Or C12) And C7 And C8 Then Begin ...
```

That's an **OR-gate between two independent evidence paths** — a trade can
qualify on strong price-action/pattern evidence alone, or on strong
volume/CVD evidence alone, without needing both. `C7` and `C8` are separate,
mandatory hard gates layered on top regardless of which path fired.

This OR-gate structure is the reason the toolchain has an **entry-path
stratification** feature (see §4): a `PatternEntryScore`-triggered trade and
a `CVDEntryScore`-triggered trade are, in principle, two different trade
populations that may respond to different parameters. Pooling them together
in one analysis can dilute or hide a real effect that's specific to one
path — confirmed visually from chart review (a Pattern-only trade and a
CVD-only trade in the sample set showed different failure modes).

---

## 2. Parameter glossary

### Speed / momentum family

| Parameter | What it measures |
|---|---|
| `PipSpeed` | Raw bar-to-bar price speed, in pips. The base building block for several other metrics and for `C5`/`C8`. |
| `PipSpeedLimit` | Threshold constant defining what counts as a "fast" bar. |
| `PipSpeedTrendPct` | **% of bars in the preceding trend leg that were "fast."** Measures how *uniform* the prior move's speed was — high = a persistently strong move; low = a move that already had stalling bars mixed in. **Needs opposite calibration per strategy**: Fast wants this high (`>=71` or higher — confirmed via direct testing), Slow wants it low (`<=71` — confirmed via significance test at p=0.0087, the strongest single finding in this project). |
| `HMinPipSpeedTrendPct` / `HighPipSpeedTrendPct` | Two related constants: `HMinPipSpeedTrendPct` is the hard-gate floor used in `C7`; `HighPipSpeedTrendPct` is a separate, higher bar used inside `PatternEntryScore` for a bonus point. |
| `PipSpeedNorm` | `PipSpeed` divided by `ShortATR` (not `AvgATR`) — makes raw pip speed comparable across symbols at very different price levels/volatility regimes. |
| `PipSpeedAcel` | Bar-over-bar change in `PipSpeed` (acceleration). |
| `PipSpeedAcelNorm` | `PipSpeedAcel` normalized by `ShortATR`, same rationale as `PipSpeedNorm`. |
| `CVDSpeedPct`, `CVDAcelPct`, `CVDDeltaPct` | The same speed/acceleration/delta concepts applied to cumulative volume delta (CVD) instead of price. |

### ATR / volatility family

| Parameter | What it measures |
|---|---|
| `BarATR` | `TrueRange` of the **current** bar — a single-bar, un-smoothed value. **Do not use as a decision threshold.** A low value just means "this bar hasn't expanded yet" — a failure-to-qualify state, not a quality signal. |
| `AvgATR` | `Average(TrueRange, 36)[1]` — a long, smoothed baseline. Good for normalizing *structural, whole-leg* measures (like `FullDeltaATRs`). **Also should not be used as a standalone decision threshold** — same reasoning as `BarATR`: it doesn't distinguish "genuinely bad setup" from "there was a volatility spike recently for unrelated reasons." |
| `ShortATR` | `Average(TrueRange, 4)[1]` — short window, and offset by one bar (`[1]`) so it doesn't include the current bar. Purpose-built for normalizing *instantaneous* metrics (`PipSpeedNorm`, `PipSpeedAcelNorm`) without the current bar's own range diluting the ratio it's meant to isolate. |
| `ATRsFromHma` | Distance between price and the HMA fast line, in ATR units — "how extended is price." Significant for **Fast** (both directions); not significant for **Slow** (matches its design — it catches earlier-stage turns). |
| `DeltaATRs` / `FullDeltaATRs` | Total displacement over a trend leg, in ATR units (normalized). **Preferred over raw `DeltaPips`** — see Gotchas. |
| `DeltaPips` | Raw, un-normalized version of the above. Confounded by instrument price level (a $500 stock naturally produces bigger raw pip moves than a $20 stock, for reasons unrelated to setup quality) — avoid as a decision gate. |
| `RevATRsPerSec` | Speed of the reversal move itself: distance from the turn point, in ATR units, per second elapsed. |
| `Angle` / `FullAngle` | Steepness of the HMA slope over a leg. |

### Trend-shape family

| Parameter | What it measures |
|---|---|
| `TrendBarCount` | Number of bars in the preceding trend leg. Significant for Slow long (shorter = better, `<=14`). |
| `HMAGapMean` | Average distance between HMA fast and HMA slow over the preceding leg, normalized by `AvgATR`. |
| `HMAGapStdDev` | Standard deviation of that same gap, normalized by `AvgATR` — how much the gap wobbled. |
| `HMAGapCV` | `HMAGapStdDev / HMAGapMean` (or a sentinel value if the mean is too small). Intended to measure "was the trend's gap consistent." **Has a real mathematical flaw — see §5, this is the most important gotcha in this document.** |

### CVD (volume) family

| Parameter | What it measures |
|---|---|
| `CVDAvg` | Average cumulative volume delta over a window — feeds `C6` ("volume directional"). |
| `CVDAcel` | 2-bar-lookback acceleration of CVD — feeds `C10` ("accel last 2 bars"). |
| `CVDDelta` | Single-bar CVD delta — feeds `C13`. **Sign must flip per direction**: `> 0` confirms long, `< 0` confirms short. |

### Composite scores

| Parameter | What it measures |
|---|---|
| `PatternEntryScore` | Weighted sum of price-action/pattern evidence: `C5`, the `PipSpeedTrendPct`-vs-`HighPipSpeedTrendPct` bonus, `C14`, `C15`. |
| `CVDEntryScore` | Weighted sum of volume evidence: `C6`, `C10`, `C13`, `C11`. |
| `EntryPath` | Not a strategy parameter — a **derived classification** (`pattern_only` / `cvd_only` / `both` / `neither`) added to the analysis toolchain, based on which side of the `(C3 Or C12)` gate actually fired for a given trade. Lets you analyze the two entry paths as separate populations. |

---

## 3. The C1–C15 condition reference

**Important caveat before reading this table**: flag numbers are **not**
consistent across strategy versions. The same number (`C5`, `C9`, `C12`,
`C13` in particular) has meant different conditions in earlier strategy
variants (`AtsPriceQuickReversal`, `AtsPriceBrkout`) discussed earlier in
this project than it means here. Always check the specific strategy's own
source before assuming a flag's meaning carries over. The table below
reflects the current AtsFastReversal/AtsSlowReversal definitions.

| Flag | Current meaning (Fast/Slow) |
|---|---|
| `C1`–`C4` | Additional hard gates present in the strategy; specific formulas weren't detailed in this project's discussions. |
| `C3` | `PatternEntryScore >= MinPatternEntryScore` — the "pattern path" half of the entry OR-gate. |
| `C5` | Speed flip — prior bar moved fast in one direction, current bar moves fast in the other. Component of `PatternEntryScore` (weight 2). |
| `C6` | CVDSpeedPct-based "volume directional" condition. Component of `CVDEntryScore` (weight 2). |
| `C7` | `PipSpeedTrendPct >= HMinPipSpeedTrendPct` — a **mandatory hard gate**, separate from either composite score. |
| `C8` | `PipSpeed >= PipSpeedLimit` (long) / `AbsValue(PipSpeed) >= PipSpeedLimit` (short) — current-bar absolute speed, **mandatory hard gate**. |
| `C9` | `HMAGapCV <= HMinHMAGapCV` — consistency-of-trend-gap condition. (An earlier reported "fix" changing this to `>=` was tested and rejected — see §5; the `<=` direction is the empirically correct one.) |
| `C10` | CVDAcel-based "accel last 2 bars" condition. Component of `CVDEntryScore` (weight 4, most recent version). |
| `C11` | `ATRsFromHma` bar-expansion condition. Component of `CVDEntryScore` (weight 2, most recent version). |
| `C12` | `CVDEntryScore >= MinCVDEntryScore` — the "CVD path" half of the entry OR-gate. **Note the naming collision**: in `AtsPriceBrkout`, `C12` meant something entirely different (an earlier HMA cross check) and was found to be a constant, uninformative flag there. Different strategy, different meaning. |
| `C13` | CVDDeltaPct confirms direction. Component of `CVDEntryScore` (weight 1). Sign-dependent per direction (see CVDDelta above). |
| `C14`, `C15` | Specific bullish/bearish bar-formation (candlestick) patterns, each contributing 3 points to `PatternEntryScore`. Long and short versions are mirror images of each other. |

### Using C1–C15 with the optimization tools

- **`ats_param_optimizer.py`**: boolean flags are tested directly via its
  "boolean flags" section (`require flag == 1`), showing expectancy
  improvement over baseline for each — a quick way to see which individual
  `C` conditions are already pulling their weight.
- **`ats_feature_importance.py`** (Random Forest + permutation importance —
  see below): include the `C` flags alongside continuous parameters in
  `--params` to rank them by real predictive contribution, not just raw
  improvement.
- **`ats_optuna_optimizer.py`**: can search `C`-derived continuous
  thresholds jointly, but treats `0/1` flags like any other column — useful
  mainly for the underlying continuous values that produce a flag (e.g.
  `PipSpeedTrendPct` itself, rather than the boolean it feeds into).
- **`ats_entryscore_weight_optimizer.py`**: purpose-built for exactly this —
  searches integer weights *for a fixed set of these flags* plus a score
  cutoff, directly mirroring the `PatternEntryScore`/`CVDEntryScore`
  structure. This is the right tool when the question is "how should these
  specific conditions be weighted relative to each other," as opposed to
  "does this threshold value matter at all."

---

## 4. The Python optimization toolchain

Four scripts, each answering a different question, all sharing the same
core conventions: `ProfitHit = Profit/Loss > 0` (every exit type counts),
long and short always analyzed separately, and every result labeled with a
confidence level rather than presented as uniformly trustworthy.

### `ats_param_optimizer.py` — grid sweep & forward-testing
Mann-Whitney significance test per parameter, single-parameter and
2-parameter combo expectancy sweeps, boolean flag tests, all gated by
`--min-n` (default 30) so small, lucky subsets don't get reported as
findings. Its standout feature is `--compare-filter`, which tests one
specific, already-hypothesized rule against a dataset directly — e.g.
`--compare-filter "ind_PipSpeedTrendPct<=71" --direction short` — the
fastest way to forward-test a finding on a fresh batch of trades.

### `ats_optuna_optimizer.py` — joint Bayesian threshold search
Where the grid sweep tests one or two parameters at a time, this searches
**all** candidate parameters jointly, letting the optimizer decide which to
include. Overfitting control is real, not cosmetic: a genuine chronological
train/test split (`--test-fraction`), with the reported filter evaluated
*once* on data the optimizer never saw. This has directly caught overfitting
in this project more than once — a filter looking great in training and
then collapsing (even flipping sign) on the held-out window. Optional
`--cv-folds` cross-validates the objective within the training window
itself for an extra layer of regularization.

### `ats_feature_importance.py` — which parameters matter at all
Trains a Random Forest (or Gradient Boosting) classifier to predict
win/loss from the candidate parameters, cross-validated, and reports
**AUC first** — if it's near 0.5 (or even below it), the script says
explicitly that the importance ranking beneath it is noise, not a finding.
Uses permutation importance (measured out-of-fold) rather than raw impurity
importance, since the latter is biased toward high-cardinality continuous
columns regardless of whether they're predictive. Run this *before* a
threshold search to know which parameters are worth searching over — this
is the tool that first confirmed `TrendBarCount`/`HMAGapCV`/`PipSpeedNorm`
for Slow long and `PipSpeedTrendPct`/`PipSpeed` for Slow short, each
independently agreeing with the significance test.

### `ats_entryscore_weight_optimizer.py` — optimizing the composite scores
A fundamentally different search problem from the other three: instead of
independent thresholds, it searches **one integer weight per boolean
component plus a shared score cutoff** — the actual decision variables in
`PatternEntryScore`/`CVDEntryScore`. Supports `--components-long` /
`--components-short` (needed because `CVDDelta`'s sign flips per direction)
and a `--min-threshold-frac` guard against a specific failure mode described
below.

### Entry-path stratification (all three analysis scripts)
Given `(C3 Or C12)`, pooling all trades together mixes two potentially
different populations. Passing `--min-pattern-score` and `--min-cvd-score`
to any of `ats_param_optimizer.py` / `ats_feature_importance.py` /
`ats_optuna_optimizer.py` / `ats_entryscore_weight_optimizer.py` additionally
splits each direction into `pattern_only` / `cvd_only` / `both` subsets and
reruns the full analysis on each — the classification column itself is
automatically excluded from that bucket's own candidate list (it would be
tautological). Expect smaller per-bucket sample sizes and more
`SKIPPED`/`LOW CONFIDENCE` results until enough trades accumulate per
bucket — that's the guard doing its job, not a bug.

---

## 5. Nuances and gotchas

### `HMAGapCV` — the big one
`HMAGapCV = HMAGapStdDev / HMAGapMean` is meant to detect "was the trend's
gap consistent," but dividing by a small mean is inherently unstable: a
genuinely small-but-real gap (mean near zero, as commonly happens right
around a turn, when the HMAs are naturally converging) produces an
exploding ratio that looks identical to genuine chop, even though the
underlying situation — "the averages nearly touched" — is a completely
different market state. A real chart example demonstrated this directly:
one setup had `StdDev=0.21, CV=0.39` (mean gap ≈0.54 ATR — large and stable,
correctly flagged low), while another had `StdDev=0.36, CV=9.49` (mean gap
≈0.038 ATR — nearly zero, misleadingly flagged as "wildly inconsistent"
when the real story was "the gap collapsed").

**Fix**: gate the mean directly, not just the CV — `IFF(HMAGapMean >= 0.3,
HMAGapStdDev / HMAGapMean, 99)` (folding the floor into the sentinel
condition, rather than a separate `AND` clause, is simpler and equivalent).
Low `HMAGapCV` remains trustworthy on its own (there's no way to get a low
ratio without a real, sizable gap) — it's specifically *high* `HMAGapCV`
that's ambiguous between "genuinely erratic" and "division-by-near-zero."

### Raw ATR values should never be decision gates
`BarATR` and `AvgATR` are both meant for **normalization only**. Using
either as a standalone threshold risks tripping on incidental recent
volatility rather than real setup quality — a low `BarATR` just means "this
bar hasn't expanded," not "this is a bad setup." There's partial empirical
support for a related, narrower effect on Slow short: hit-rate-based checks
(significance test, feature-importance direction) mildly favor *higher*
`AvgATR`, while the raw-dollar expectancy sweep favors capping it *low* —
consistent with high-`AvgATR` trades winning about as often but **losing
by more when wrong** (a fat-tailed stop-loss effect), which is a stop-sizing
question, not an entry-filter one.

### Raw, price-scale-dependent quantities are confounded
`DeltaPips` looked significant in isolation, but its ATR-normalized
counterpart `DeltaATRs` is both conceptually cleaner and empirically
stronger — prefer the normalized version as the actual decision gate.

### The same indicator can need opposite calibration by strategy design
`PipSpeedTrendPct` is the clearest example: Fast wants it high (confirms a
strong preceding move worth fading), Slow wants it low (confirms genuine
deceleration/exhaustion already showing). Don't assume a finding from one
strategy transfers to its sibling without checking direction explicitly.

### A significant p-value with identical hit/loss medians is a red flag
Seen repeatedly (`CVDAvg`, `CVDEntryScore`, `ind_Interval`,
`ind_PatternEntryScore`) — usually means a lumpy/discrete distribution
skewing the rank test, not real separation. Always eyeball
`hit_median` vs. `loss_median`, don't trust the p-value alone.

### Quantized indicators produce tie clusters that distort simple thresholds
`PipSpeedTrendPct` is derived from a `CountIf(...)/TotalBars` ratio, so it
takes on a limited set of rounded values — a large fraction of trades can
sit at exactly the same number (e.g., 30 of 145 FastReversal long trades sat
at exactly `71`). A naive `<=`/`>=` split both include this tied cluster,
which can mask the true shape of the relationship. When investigating a
threshold near a common value, split the exact-tie group out separately
(`<71` / `==71` / `>71`) to see the real pattern.

### The weight optimizer can only reweight what you give it
If the strongest real signal for a direction (e.g. `TrendBarCount`,
`ATRsFromHma`, `PipSpeedTrendPct`) isn't in the component list handed to
`ats_entryscore_weight_optimizer.py`, no amount of weight searching will
find it — check `ats_param_optimizer.py`/`ats_feature_importance.py` first
to know what actually matters before restricting the weight search.

### Degenerate-threshold pitfall in the weight optimizer
Even with the score-threshold floor (`--min-threshold-frac`) enabled, the
search can let several components share the same high weight so that *any
one* of them alone still clears the (now-higher) threshold — functionally
identical to no filter at all. Always check the "optimized" filter's
`train_n` against the full training window size; if it matches nearly every
trade, treat it as this pattern rather than a real combination effect.

### A component with 0% or 100% firing rate carries zero information
Confirmed for `AtsPriceBrkout`'s `C5`/`C12` (100% — structural preconditions
already enforced upstream, not real gates) and `C9` (0% — dead at that
strategy's typical value range). Always check firing rate before including
a flag in any weight or threshold search.

### Small-sample overfitting: train looks great, test collapses
The single most common failure mode across every tool in this project.
`ats_optuna_optimizer.py`'s train/test split exists specifically to catch
this, and has — a filter reaching `$8/trade` in training and then flipping
to `-$3/trade` out-of-sample is a repeat occurrence, not a one-off. Never
act on a training-window number alone.

### Always verify a JSON report against the actual current CSV
Several times in this project an uploaded report turned out to be generated
from an older, smaller version of the trade log even though it was just
uploaded. Check `csv_path` and the baseline `n` against what you'd expect
before trusting a report; regenerate fresh when in doubt.

---

## 6. Future enhancements and testing

A few concrete next steps worth prioritizing. **Log `HMAGapMean` directly**
rather than continuing to reconstruct it algebraically from `HMAGapStdDev`
and `HMAGapCV` — cleaner, and avoids edge cases at the `99` sentinel.
**Confirm the FastReversal long `PipSpeedTrendPct` floor on a genuinely new
batch of trades** before wiring it into the live formula; the current
finding, while clean and not outlier-driven, is still in-sample.
**Investigate the exit-type breakdown directly** (stop-loss vs. profit-target
magnitude, conditioned on `AvgATR`) to properly confirm or refute the
tail-risk hypothesis rather than inferring it indirectly from expectancy
sweeps — this likely requires adding an exit-type-aware analysis mode to the
toolchain. **Differentiate the `C11`/`ATRsFromHma` weight between Fast and
Slow's `CVDEntryScore`** explicitly, rather than sharing one formula across
two strategies that need it to matter by very different amounts. **Build
Fast/Slow-specific default component lists** for
`ats_entryscore_weight_optimizer.py` reflecting the current
`PatternEntryScore`/`CVDEntryScore` structure (`C5`/`C14`/`C15`/
`PipSpeedTrendPct`-bonus for pattern; `C6`/`C10`/`C13`/`C11` for CVD), since
the script's built-in defaults still reflect an earlier, simpler 7-component
formula. Finally, **let the entry-path stratification features mature with
more data** — the pattern-only/cvd-only split is a promising, only recently
added lens, and needs a few more trade batches accumulated per bucket before
its own findings can be trusted at the same level as the pooled analysis.
