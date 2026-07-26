# Context: Trading Strategy Parameter Optimization

This document summarizes an ongoing project analyzing and tuning entry-filter
parameters for several related trading strategies. Use this to pick up the
work in a new conversation without re-deriving everything from scratch.

## The strategies

Four related strategies, all built on the same underlying pattern: a
multi-condition entry system using boolean "C" flags (`C1`-`C15`), continuous
indicator thresholds, and (in newer versions) grouped weighted scores.

- **AtsPriceQuickReversal** — enters on a sharp, sudden reversal (a "speed
  flip"). The original/most-analyzed strategy in this project.
- **AtsPriceBrkout** — similar structure, but enters on breakout continuation
  rather than reversal. Its `C5`/`C12` definitions differ from
  QuickReversal's (see "Per-strategy notes" below) and turned out to be
  **constant (always true) in the trade log**, not real gates — a
  strategy-specific finding, not a general pattern.
- **AtsFastReversal** — a newer strategy requiring a *strong, already-extended*
  reversal. `ATRsFromHma` matters a lot here (confirmed significant).
- **AtsSlowReversal** — sibling to FastReversal, but catches a move that has
  *already started* turning (less extension needed). `ATRsFromHma` does
  **not** show significance here — this is a designed difference between the
  two, not a gap in analysis.

Trades are logged per-symbol, merged into one CSV per strategy
(`Ats<Name>-merged.csv`), split by `ind_SignalSent` (`1`=long, `-1`=short) —
**always analyze long and short separately**, they respond to different
parameters.

## The toolchain (all delivered, all in `/mnt/user-data/outputs/` with matching `.md` docs)

1. **`ats_param_optimizer.py`** — grid sweep. Mann-Whitney significance test
   per parameter, single-parameter and 2-parameter combo expectancy sweeps,
   boolean flag tests. `--min-n` overfitting guard (default 30).
   `--compare-filter` forward-tests a specific rule against new data.
2. **`ats_optuna_optimizer.py`** — Bayesian (TPE) joint threshold search
   across all parameters at once, with a real chronological train/test split
   (`--test-fraction`) and optional in-training CV (`--cv-folds`). Fixed bug:
   `test_fraction=0` used to silently force 1 test trade; now correctly means
   "no held-out split."
3. **`ats_feature_importance.py`** — Random Forest + permutation importance +
   optional SHAP, cross-validated AUC reported first. Explicitly flags
   AUC≈0.5 (no signal) and AUC<0.5 (noise, not an "inverse signal") rather
   than presenting a ranking as meaningful when it isn't. Skips modeling
   below 60 trades per direction.
4. **`ats_entryscore_weight_optimizer.py`** — optimizes per-component
   integer weights + a score threshold for the EntryScore formula (as
   opposed to independent thresholds). Supports `--components-long` /
   `--components-short` (needed because e.g. `CVDDelta` needs opposite sign
   per direction: `gt0` for long, `lt0` for short) and `--min-threshold-frac`
   (prevents a degenerate "any single component fires" solution — see
   Pitfalls below).

All four share conventions: `ProfitHit = Profit/Loss > 0` (all exit types
count), long/short always separate, `--output` writes full JSON, and every
finding is labeled with a confidence level (HIGH = backed by real
significance / survives a held-out test; LOW = small-sample or unconfirmed).

## Hard-won methodology rules (apply these before trusting any new result)

- **Always check `csv_path` and baseline `n` in an uploaded JSON against
  what you'd expect.** Several times in this project a JSON was stale
  (generated from an older, smaller CSV) even though it was just uploaded.
  When both a CSV and a JSON are provided, regenerate fresh rather than trust
  the JSON blindly.
- **Train-window numbers are not evidence.** Only test-window (held-out)
  results, at adequate sample size, should ever inform a live parameter
  change. A filter that looks great in training and collapses on test is the
  single most common failure mode encountered in this project.
- **A significant p-value with identical hit/loss medians is a red flag, not
  a finding.** Seen repeatedly (`CVDAvg`, `CVDEntryScore`, `CVDDeltaPct`,
  `ind_Interval`, `ind_PatternEntryScore`) — usually indicates a
  lumpy/discrete distribution skewing the test, not real separation. Always
  eyeball hit_median vs. loss_median, don't trust p-value alone.
- **"Beats baseline" is not the same as "profitable."** A reweighted filter
  can beat a losing baseline while still losing money itself (seen in
  BrkOut short). Check the absolute expectancy, not just the delta.
- **Raw, price-scale-dependent quantities (e.g. `DeltaPips`) are confounded
  by instrument price level.** Prefer their ATR-normalized counterparts
  (`DeltaATRs`) as actual decision gates; the raw version can look
  significant purely because higher-priced stocks produce bigger raw moves.
- **Single-bar or short-window raw ATR values (`BarATR`, and by the same
  logic `AvgATR`) should not be used as standalone decision thresholds** —
  they're meant for normalization only. A low value often just means "this
  bar/window hasn't expanded yet" (a failure-to-qualify state), and using
  them as gates risks tripping on incidental recent volatility rather than
  real setup quality. Prefer ratio-form indicators that already encode
  "relative to volatility" (`ATRsFromHma`, `DeltaATRs`, `RevATRsPerSec`,
  `PipSpeedNorm`) over raw ATR thresholds.
- **A component with 0% or 100% firing rate in the trade log carries zero
  information** and should be excluded from any weight/threshold search —
  confirmed for BrkOut's `C5`/`C12` (100%) and `C9` (0%).
- **Degenerate-threshold pitfall in `ats_entryscore_weight_optimizer.py`:**
  even with `--min-threshold-frac` set, the search can still let several
  components share the same (high) weight so that any one of them alone
  clears the threshold — functionally identical to no filter. Always check
  `train_n` against `total_train_n`; if the "optimized" filter matches
  nearly every trade, treat it as this pattern, not a real combination
  effect.
- **The weight optimizer can only reweight components you give it.** If the
  strongest real signal (e.g. `BarATR`, `AvgATR`, `DeltaPips`, `Angle`) isn't
  one of the candidate components, reweighting will never find it — check
  `ats_param_optimizer.py`/`ats_feature_importance.py` results first to know
  what actually matters before restricting the weight search to a component
  list.
- **Direction-specific component signs matter.** `CVDDelta > 0` for long vs.
  `CVDDelta < 0` for short is the same underlying idea, opposite polarity —
  use `--components-long`/`--components-short` rather than one shared list.

## Per-strategy state (as of last analysis)

### AtsPriceQuickReversal
- Long: `FullDeltaATRs >= 9 AND FullAngle >= 26` — confirmed, replicated
  across two independently-collected batches (rare, strong confirmation).
- `PipSpeedPct` needs a **ceiling** (~60-65 long, ~50-60 short), not a higher
  floor — the existing `HMinPipSpeedPct` floor is directionally fine but
  uninformative (fires ~86% of trades); a max-cap parameter (`HMaxPipSpeedPct`)
  is the actual missing piece. Confirmed mechanistically: `PipSpeedPct`
  measures what fraction of the *preceding trend leg* was moving fast — high
  values mean a uniformly strong, still-accelerating prior move (risky to
  fade); low values mean the prior move already had stalling bars mixed in
  (real exhaustion, matches this strategy's premise).
- `HMAGapCV`: keep the `<=` direction (already correct), but recalibrate the
  threshold — real useful range is ~1.0-1.6, not the current `0.40` (which
  made the gate too rare to matter). ~1.4 works for both directions.
- Short: `ATRsFromHma` finding has NOT reliably replicated across datasets —
  treat with caution, re-test before trusting.
- EntryScore reweighting (7-component, later split into `SpeedEntryScore`/
  `CVDEntryScore` groups): repeated runs across growing datasets consistently
  find **no benefit from reweighting on the long side** — equal weights
  already captures nearly all available long-side value. Short side has
  shown one promising-but-unconfirmed candidate (`C5 OR C13`, i.e. weight 1
  each, others 0) that beat baseline on one held-out run — needs
  confirmation on a fresh batch before trusting.

### AtsPriceBrkout
- Long: significant on `ATRsFromHma` (p=0.037) and `FullDeltaATRs` (p=0.038),
  both higher=better. Best combo: `FullDeltaATRs>=10.04 AND
  RevATRsPerSec<=0.63`.
- Short: significant cluster around `PipSpeed`/`PipSpeedAcel`/
  `PipSpeedAcelNorm` (all p<0.05) plus weak `ATRsFromHma`. Best combo:
  `PipSpeed>=-0.68 AND ATRsFromHma>=0.49`.
- `HMAGapCV`: `<=0.40` is dead (0% firing both directions — BrkOut's typical
  values sit around median 1.8-1.9). Short benefits from `<=1.0` ($1.63→
  $11.82/trade, promising, needs forward-test); **long shows no benefit from
  any HMAGapCV filter tested** — don't add this gate to BrkOut long.
- `C5`/`C12` are structurally constant here (100% firing) — exclude from any
  weight search; this differs from QuickReversal where `C5` is a real,
  sometimes-true signal.

### AtsFastReversal
- Requires a strong/extended reversal — `ATRsFromHma` matters here
  (confirmed significant for long, p=0.031, threshold `>=0.96`).
- Short side already profitable at baseline (+$5.57/trade on n=103) even
  before tuning; best combo (after excluding `BarATR` per the
  normalization-only rule): `PipSpeed>=-3.28 AND ATRsFromHma>=1.04` — not yet
  independently significance-tested after the BarATR exclusion, but
  theoretically coherent (fast-down moves are a real, expected asymmetry).
- Long baseline ~breakeven (-$0.30/trade on n=117).
- A large recent code change (new `PatternEntryScore`/`CVDEntryScore` split,
  `HighPipSpeedTrendPct` added, several limit changes, entry condition
  changed from `(C3 And C12)` to `(C3 Or C12) And C7 And C8`) was described
  but **the most recently uploaded CSV still shows the same n and stats as
  the pre-change analysis (117 long / 103 short, same expectancy)** — the new
  trade data reflecting these changes had not yet arrived as of the last
  message. **Open item: re-run the full toolchain once fresh post-change
  trades are available.**

### AtsSlowReversal
- Catches an already-turning move — `ATRsFromHma` does NOT show significance
  here (expected, by design).
- Short: strongest, best-confirmed finding in the whole project —
  `AvgATR` was p=0.003 but is being excluded per the
  normalization-only rule (per the Fast/BarATR logic, agreed by the user).
  Fallback: `DeltaATRs >= 6.39` (ATR-normalized version of `DeltaPips`,
  p=0.012 on the raw version, `DeltaATRs` outperforms it: $4.00/trade vs
  $2.55/trade improvement) — this is the current best recommendation for
  SlowReversal short. Two independent methods (significance test +
  feature-importance permutation ranking) agreed on `AvgATR`/`DeltaPips` as
  the top-2 parameters before the AvgATR exclusion, which is a meaningfully
  strong double-confirmation.
- Long: **persistent, unresolved, worsening problem** — baseline expectancy
  got worse as more data arrived (-$2.53 → -$5.88), ruling out "just noise."
  EntryScore reweighting does not fix it (test expectancy still ~-$8.92
  vs. baseline -$10.25, both bad). No confirmed fix yet — needs
  investigation beyond parameter tuning, not just another threshold sweep.

## Open items / where to pick up next

1. Re-run the full toolchain (`ats_param_optimizer.py`,
   `ats_feature_importance.py`, `ats_entryscore_weight_optimizer.py`) on
   AtsFastReversal once trade data reflecting the newest code changes
   (`PatternEntryScore`/`CVDEntryScore` split, `HighPipSpeedTrendPct`,
   updated limits) is actually available.
2. Confirm whether `DeltaATRs` (vs. raw `DeltaPips`) is also the better
   choice for AtsFastReversal specifically — only checked for SlowReversal
   short so far.
3. Investigate AtsSlowReversal long's worsening baseline directly (e.g., by
   date/symbol segmentation) rather than continuing to search for a
   parameter fix.
4. Consider differentiating the weight given to `C11`/`ATRsFromHma` between
   Fast (should matter more) and Slow (should matter little/not at all) in
   the `CVDEntryScore` formula, rather than sharing one weight across both
   strategies.
5. Any newly-confirmed threshold (QuickReversal `PipSpeedPct` cap,
   `HMAGapCV` recalibration, BrkOut combos, etc.) still needs a genuine
   forward-test on a fresh batch of trades before being called "confirmed"
   rather than "promising."
