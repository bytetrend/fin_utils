# Trading Strategy Optimization — Addendum (Updates Since Last Summary)

This covers everything from the most recent analysis session, which is not
reflected in the earlier full project summary. Read alongside that document
rather than as a replacement for it.

---

## 1. Architecture change: both strategies moved from OR-gate to AND-gate entry logic

Both `AtsFastReversal` and `AtsSlowReversal` were restructured so that
`PatternEntryScore` (`C3`) and `CVDEntryScore` (`C12`) **no longer decide
trade entry at all**. The new entry logic is a strict AND of four
conditions:

```
C7 And C8 And C11 And C13
```

with these redefinitions:

```
C7  = PipSpeedTrendPct <= HMaxPipSpeedTrendPct        (now a ceiling for both strategies)
C9  = HMAGapCV <= HMaxHMAGapCV and HMAGapStdDev <= HMAGapStdDevMax   (no longer part of entry decision)
C11 = ATRsFromHma > ATRsFromHmaMinLim and ATRsFromHma < ATRsFromHmaMaxLim
C13 = CVDDeltaPct > CVDDeltaPctLim                    (unchanged)
```

**New methodology lesson: AND-gate compounding is a distinct risk from any
single miscalibrated threshold.** Even individually reasonable thresholds
can crush trade volume when four must all be true simultaneously —
probabilities multiply. This showed up dramatically on `AtsFastReversal`
short (see below) and is worth checking any time an OR-based (scored)
system is replaced with a strict AND.

**Current limit values, both strategies (for reference):**

`AtsFastReversal`: `ATRsFromHmaMaxLim(2.8)`, `ATRsFromHmaMinLongLim(0.6)`,
`ATRsFromHmaMinShortLim(1.0)`, `HMAGapStdDevLongMax(0.36)`,
`HMAGapStdDevShortMax(0.40)`, `HMaxPipSpeedTrendPct(90)`. Changed from
previous values: `CVDSpeedPctLim` 50→40, `CVDAcelPctLim` 70→60,
`HMinDeltaPips` 20→30, `PipSpeedLimit` 0.90→0.60.

`AtsSlowReversal`: `CVDDeltaPctLim(10)`, `CVDSpeedPctLim(30)`,
`CVDAcelPctLim(50)`, `HMinDeltaPips(30)`, `PipSpeedLimit(0.50)`,
`PipSpeedTrendPctLimit(0.2)`, `PipSpeedFlipLimit(1.0)`,
`ATRsFromHmaMaxLim(2.8)`, `ATRsFromHmaMinLongLim(0.6)`,
`ATRsFromHmaMinShortLim(1.0)`, `RevATRsPerSecLim(0.4)`,
`HMinAngleLim(19)`, `HMinATRs(6)`, `HMaxHMAGapCV(0.5)`,
`HMinGapMean(0.05)`, `HMAGapStdDevLongMax(0.36)`,
`HMAGapStdDevShortMax(0.40)`, `HMaxPipSpeedTrendPct(80)`,
`MinPatternEntryScore(4)`, `MinCVDEntryScore(5)`.

**Context on overall impact of the changes:** despite the short-volume
problem diagnosed below, the changes as a whole improved the strategy —
reported Total Net Profit went from **-$615.31 → +$171.32** and Sharpe from
**-2.011 → +4.311**. The diagnosis below is about fine-tuning an
already-net-positive change, not fixing something that made things worse
overall.

---

## 2. AtsFastReversal short: diagnosed and fixed the near-zero-trades problem

**Symptom:** a 127-trade file (07/01–08/04) showed 124 long vs. only **3**
short trades — short entries had essentially stopped firing.

**New diagnostic technique used, worth reusing:** the trade log only
contains *entered* trades, not rejected signals, so individual condition
pass/fail rates can't be directly computed for a starved direction. Instead:
(a) examine how close the few survivors sit to each threshold's edge, and
(b) use the *other* direction's distribution (assuming a roughly similar
market regime) as a rough calibration reference for the affected direction's
thresholds.

**Root cause, once a fuller 700-trade historical file (287 shorts) was
obtained:** it was **not** primarily `ATRsFromHma`, the original suspect.

```
C7 alone:  88.9% pass, exp=-$1.45
C8 alone: 100.0% pass, exp=-$0.95   (complete no-op, same pattern as many gates found before)
C11 alone: 68.6% pass, exp=-$0.01   (best of the four)
C13 alone:  4.5% pass, exp=-$2.95   ← the actual bottleneck, and poor quality even when it does fire
```

`C13` (`CVDDeltaPct > CVDDeltaPctLim=20`) passes only 13 of 287 short
setups — far tighter than any other condition, and negative expectancy on
its own (n=13, low confidence, but consistent with it not being a good
filter as currently calibrated).

**Fix found and recommended:** replace `C13` with `C6` in the AND:

```
C7 And C8 And C11 And C6   →  n=39,  $3.06/trade, total $119   (best quality)
C7 And C8 And C11 And C10  →  n=122, $1.14/trade                (more volume, lower edge)
```

This recovers 13-40x the trade volume of the current setup while matching
or beating every other combination tested in this analysis, including the
original `ATRsFromHma`-only approach.

**Secondary fix:** `ATRsFromHmaMinShortLim` should move from **1.0 → ~1.2**
— the current floor sits close to the median of the short-side distribution
rather than comfortably in the tail (long's equivalent floor, 0.6, sits
well below its own 25th percentile by contrast — a real asymmetry in how
loosely the two directions were calibrated).

**Important correction to the August 6 recommendation, found while
re-verifying on the fuller 287-trade dataset:** the previously reported
`ATRsFromHma>=1.1: $7.62/trade` and the `DeltaATRs>=7` combo reaching
`$11.66/trade` did **not** hold up at anything close to that magnitude:

```
                    Aug 6 (n=168, baseline $5.29)   Now (n=287, baseline -$0.95)
ATRsFromHma>=1.1:   $7.62/trade                     $0.11/trade
DeltaATRs>=7 combo: $11.66/trade                    negative at every ATRsFromHma level tested
```

The *direction* of the `ATRsFromHma` effect is still real and actually
*more* statistically significant on the larger sample (p=0.0061, stronger
than before) — but the dollar magnitude was overstated on the smaller
slice. **General lesson: a confirmed direction does not guarantee the
dollar magnitude will hold as more data arrives** — always re-verify
magnitude, not just direction, once more data is available.

---

## 3. AtsFastReversal: `C14`/`C15` finding reconfirmed on a larger, short-only sample

Previously found on a mixed long/short dataset; now reconfirmed
independently on 287 short trades alone:

```
C14: fires 48.8%, n=140, exp=-$2.09
C15: fires 51.2%, n=147, exp=-$2.21
Both C14 & C15 = 1: n=140  ← still exactly equals C14's own count
```

`C14` remains a strict logical subset of `C15` — `(C14 Or C15)` is
mathematically identical to `C15` alone, contributing nothing extra. Both
are still net-negative when they fire (worse than baseline). This is not
part of the current entry logic (neither `C14` nor `C15` is in the
`C7 And C8 And C11 And C13/C6` gate), so it doesn't affect the
recommendation above, but remains a priority item to check in source code —
now confirmed twice, on two different datasets.

---

## 4. AtsSlowReversal long: real, held-out-validated improvement found

**Starting point:** `C7`, `C8`, `C11`, `C13` are **all complete no-ops** for
Slow long (100% firing on all 61 trades in the dataset checked) — none of
the current limits are doing any filtering work. Baseline: n=61,
$11.95/trade (already a strong baseline).

**Significance test:** nothing individually reached significance (n=61 is
thin), but every parameter checked leaned the same direction —
`PipSpeedTrendPct`, `PipSpeed`, `ATRsFromHma`, and `CVDAcelPct` all showed
"lower = better," a consistent pattern worth taking seriously despite the
lack of formal significance.

**Threshold sweep result — best combo:**

```
ATRsFromHma<=1.1 AND PipSpeedTrendPct<=75
  → n=31, hit=64.5%, $25.78/trade, total $799   (vs. baseline $11.95/trade, $729 total)
```

Beats baseline on every dimension: higher edge, higher total dollars
(despite using half the trades), and much better hit rate.

**Held-out validated via chronological train/test split:**

```
TRAIN (n=43): baseline $3.88  → filtered n=21, $13.85/trade
TEST  (n=18): baseline $31.23 → filtered n=10, $50.82/trade
```

The filter beat its *own test-window's baseline* by a wide margin — it did
not collapse on held-out data. Caveat: `n=10` on test is well below the
usual 30-trade comfort threshold, so this is "survived a real check," not
"fully confirmed" — and the test window's unusually strong own baseline
means some of this could reflect a generally favorable recent period, not
only the filter.

**`CVDSpeedPct`/`CVDAcelPct` (behind `C6`/`C10`) checked separately:**
`CVDAcelPct<=30` gave a modest independent improvement
(`ATRsFromHma<=1.1 AND CVDAcelPct<=30` → n=29, $21.38/trade) but weaker
than the primary combo. `CVDSpeedPct` alone showed **no real improvement at
any threshold** — recommend leaving `CVDSpeedPctLim` unchanged.

**Adding a 3rd condition on top of the best 2-parameter combo actively hurt
results** (`ATRsFromHma<=1.1 AND PipSpeedTrendPct<=75 AND CVDSpeedPct<=20`
→ n=25, $25.04/trade, total $626 — lower total than the 2-parameter
version) — a visible instance of over-stretching an already-small sample
with too many simultaneous free thresholds.

**Recommendation:** `ATRsFromHmaMaxLim: 2.8 → ~1.1`,
`HMaxPipSpeedTrendPct: 80 → ~75`. Leave `CVDSpeedPctLim`/`CVDAcelPctLim` as
currently set.

---

## 5. AtsSlowReversal short: still unresolved

Every SlowReversal file checked in this session had almost no short trades
(6 total with 0 shorts; 62 total with 1 short). The single available short
trade cleared `ATRsFromHma` comfortably (1.14 vs. floor 0.7 — not sitting
at an edge the way Fast's survivors did), which suggests the short-scarcity
root cause here may be **different** from Fast's `C13` bottleneck, though
this can't be confirmed with only one data point.

**Checked directly and ruled out as an obvious lead:** unlike Fast,
`C7`/`C8`/`C11`/`C13` are *also* complete no-ops on Slow's long side
(100% firing, n=61) — so there's no clear signal from the long side about
which condition might be disproportionately blocking short. The likely
explanations are either a genuinely long-biased market period in the
dataset checked, or something upstream of these four gates entirely (not
visible in a trades-only log).

**Blocked on:** a broader historical `AtsSlowReversal` merge with a
meaningful number of short trades — the same kind of file (287 shorts) that
made the Fast diagnosis possible. Without it, this can't be diagnosed
further.

---

## 6. Open items (supersedes the equivalent section in the last summary)

1. **AtsFastReversal**: deploy `C7 And C8 And C11 And C6` (or `C10` for more
   volume) in place of the current `C13`-based AND; raise
   `ATRsFromHmaMinShortLim` to ~1.2. Forward-test before finalizing.
2. **AtsFastReversal**: `C14`/`C15` — confirmed twice now to be a
   subset/redundancy issue with negative-expectancy firing. Needs a
   source-code check, not more data analysis.
3. **AtsSlowReversal**: deploy `ATRsFromHmaMaxLim: 2.8→~1.1` and
   `HMaxPipSpeedTrendPct: 80→~75` for long. Forward-test given the thin
   held-out sample (n=10).
4. **AtsSlowReversal short**: unresolved — need a broader historical file
   with meaningful short volume before this can be diagnosed at all.
5. General: when re-verifying any earlier "confirmed" finding on a larger
   dataset, check magnitude explicitly, not just direction — this session
   found a real instance of a confirmed *direction* holding up while the
   *magnitude* dropped by roughly 10x once more data arrived.
