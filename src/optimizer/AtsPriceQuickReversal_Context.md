# AtsPriceQuickReversal — Strategy Optimization Context

## Overview

Two related reversal strategies built in MultiCharts PowerLanguage, sharing almost identical
logic. Both identify setups where price trended too far too fast and then reverses.

**AtsPriceQuickReversal** — C12 requires HMA Fast to be above/below HMA Slow AT the turn point.
The entry dot on the chart is **yellow**.

**AtsPriceBrkout** — C12 requires HMA Slow to have already crossed above/below HMA Fast BEFORE
the turn. The reversal transitions slower. The entry dot is **bright green**.

The indicator fires a signal; the strategy executes the entry. Charts are tick-based (10–50
ticks per bar). `SignalBar` = `ind_BarNumber`. `BarNumber` in the strategy can equal or exceed
`SignalBar` depending on fill timing.

---

## Chart Visual Guide

| Element | Description |
|---|---|
| RED line | HMA Fast — the primary trend indicator |
| CYAN solid line | HMA Slow — the baseline trend |
| Dotted line | MultiCharts trade path from entry to exit (not an indicator) |
| Yellow dot at HMA turn | AtsPriceQuickReversal signal |
| Bright green dot at HMA turn | AtsPriceBrkout signal |
| Orange dotted line | Prof_0 — first profit target level |
| Yellow horizontal line | Trailing stop level |
| Red horizontal line | Hard stop limit |
| CVD panel (middle) | CVDAvg = magenta line. CVDAcel histogram: CYAN = above limit, DARK GREEN = below limit. CVDDelta = yellow dots |
| PipSpeed panel (bottom) | CYAN bars = above limit (±1). MAGENTA bars = below limit. YELLOW bar = transition bar (crossing zero) |

---

## Data File Format

### Trade Columns
| Column | Description |
|---|---|
| Symbol | Stock symbol |
| EntryDate / EntryTime | Entry date (mm/dd/yyyy) and time (hhmm) |
| EntryName | `LE_PrcQuickRev` (long) or `SE_PrcQuickRev` (short) |
| EntryPrice / ExitPrice | Fill prices |
| ExitDate / ExitTime | Exit date and time |
| ExitName | Final exit trigger — see Exit Types below |
| Shares | Shares traded |
| Profit/Loss | Net P/L including ALL partial fills at profit targets |
| BarNumber | Bar number when trade entered |
| SignalBar | Bar number when signal was given |
| R/T | 1 = realtime, 0 = backtest |

### Critical Exit Interpretation Rule
**Only the LAST exit is recorded.** The strategy scales out across multiple profit targets.

| ExitName | Meaning |
|---|---|
| LX_StpLoss / SX_StpLoss | Final exit was trailing stop (~90% of all exits) |
| LX_CVD / SX_CVD | CVD momentum reversal exit |
| LX_Reversal / SX_Reversal | Opposite signal exit |
| LX_Prof_2 / SX_Prof_2 | Final profit target hit |
| LX_Prof_1 / SX_Prof_1 | Second profit target (partial) |
| LX_Prof_0 / SX_Prof_0 | First profit target (partial) |
| LX_1530 | End-of-session exit |

- **StpLoss with P/L > 0** = at least one profit target was hit before the stop. Partial win.
- **StpLoss with P/L < 0** = no profit target reached. True loser.
- **CVD / Reversal / Prof exits** = always winners by definition.

### Correct Outcome Definition for Analysis
- `ProfitHit = Profit/Loss > 0` (any target reached, regardless of exit type)
- `PureLoss  = Profit/Loss < 0` (no target reached)
- Do NOT treat ExitName=StpLoss as automatically a loss.
- Do NOT analyze stop-loss exits in isolation — include all exit types.

### Profit Target Formulas
```
Prof_0 Long:  MaxList(EntryLimitPrice, Close) + ATRValue * ProfitATR0
Prof_0 Short: MinList(EntryLimitPrice, Close) - ATRValue * ProfitATR0
```
ProfitATR0 = 2.0 ATRs. The anchor floats with the entry price via MaxList/MinList, so
ATRsFromHma does NOT reduce the distance to Prof_0 — the 2-ATR target is always measured
from the fill price. Prof_1 and Prof_2 are triggered by price hesitations (reversal signal
logic detecting a pause in the move), not fixed levels.

### Reward/Risk Structure (Dataset 4 observations)
- Avg P/L when profit target hit: ~$22–23
- Avg P/L on pure losses: ~-$13.30 to -$13.47
- Reward/risk ratio: ~1.7:1
- Break-even profit-hit rate required: ~37%
- Current overall: longs 29.1%, shorts 32.7%

---

## Indicator Columns (snapshot at signal time)

| Column | Description |
|---|---|
| ind_SignalSent | 1 = long, -1 = short |
| ind_CVDAvg | CVD average — directional volume in trade direction. Only counts ticks moving higher/lower than previous tick (not standard CVD). Magenta line in CVD panel |
| ind_CVDDelta | Per-bar difference between upticks and downticks. Yellow dots in CVD panel |
| ind_CVDAcel | CVD acceleration = CVDAvg / time. Cyan histogram when above limit, dark green when below |
| ind_RevATRsPerSec | Reversal speed in ATRs/sec from peak to current bar |
| ind_DeltaATRs | ATRs prior trend traveled — over MaxLength (CAPPED). Legacy metric |
| ind_FullDeltaATRs | ATRs prior trend traveled — over TrendBarCount (UNCAPPED). Preferred |
| ind_TrendBarCount | True bars the prior trend ran. NOT capped by HMaxTrendBars |
| ind_MaxLength | Trend bar count CAPPED by HMaxTrendBars. Causes truncation artifacts |
| ind_Angle | HMA Fast slope angle over MaxLength bars. Legacy metric |
| ind_FullAngle | HMA Fast slope angle over full TrendBarCount. Preferred |
| ind_PipSpeed | Pips/second of price over last 2 bars in trade direction. Bottom panel |
| ind_ATRsFromHma | ATRs price has reversed from HMA Fast at signal time |
| ind_DeltaPips | Pips in prior trend over MaxLength (CAPPED). Legacy |
| ind_FullDeltaPips | Pips in prior trend over TrendBarCount (UNCAPPED). Preferred |
| ind_BarPct | % bars moving in trend direction. Now over TrendBarCount (fixed) |
| ind_BarBreak | % bars where HMA alignment held. Now over TrendBarCount (fixed) |
| ind_C1 … ind_C13 | Boolean conditions at signal time (1=true, 0=false) |
| ind_Close | Stock price at signal |

### New Metrics (added for current run)
| Column | Description |
|---|---|
| ind_PipSpeedPct | % of prior trend bars where \|PipSpeed\| >= PipSpeedLimit |
| ind_PipSpeedAcel | PipSpeed[current] - PipSpeed[1] — one-bar acceleration at signal |
| ind_PipSpeedAcelNorm | PipSpeedAcel / ATR — normalized acceleration |
| ind_HMAGapMean | Mean of (HMAFast - HMASlow) over trend, in ATRs — average separation |
| ind_HMAGapStdDev | StdDev of HMA gap over trend, in ATRs |
| ind_HMAGapCV | HMAGapStdDev / HMAGapMean — coefficient of variation. LOW = consistent gap (quality trend). HIGH = choppy/inconsistent |

---

## Indicator Conditions

### Hard Gates — Always On (C1, C2, C3, C4, C12)
All must be true before any secondary conditions are evaluated.

| Condition | Metric | Current Parameter |
|---|---|---|
| C1 | TrendBarCount >= HMinTrendBars | Minimum bars in prior trend |
| C2 | FullDeltaATRs >= HMinATRs | **HMinATRs = 7** — full trend ATRs |
| C3 | FullDeltaPips >= HMinDeltaPips | Full trend pips |
| C4 | FullAngle >= HMinAngleLim | **HMinAngleLim = 25** |
| C12 | HMA alignment at turn | QuickReversal: Fast crossed Slow. Breakout: Slow already crossed |

### Secondary Conditions — Current Implementation

| Condition | Metric | Role |
|---|---|---|
| C5 | PipSpeed flip across limit | Transition: Short: `PipSpeed[1] >= limit AND PipSpeed <= -limit`. Long: `PipSpeed[1] <= -limit AND PipSpeed >= limit`. Confirms sharp speed reversal in one bar |
| C7 | PipSpeedPct >= HMinPipSpeedPct | **HMinPipSpeedPct = 40**. % of prior trend bars above speed limit |
| C8 | AbsValue(PipSpeed) >= PipSpeedLimit | Current bar speed is above limit in trade direction. Hard gate — always required |
| C9 | HMAGapCV <= HMinHMAGapCV | **HMinHMAGapCV = 0.40**. Consistent HMA gap during trend |
| C11 | ATRsFromHma < ATRsFromHmaLim | **ATRsFromHmaLim = 1.0**. Entry not too extended from HMA |
| C13 | RevATRsPerSec > RevATRsPerSecLim | **RevATRsPerSecLim = 0.1**. Reversal is fast enough |

### CVD Logic (C6 and C10 — three-way case)
| State | C6 | C10 | CVD Panel | Quality |
|---|---|---|---|---|
| Both above limit | 1 | 1 | Full cyan histogram + line above | Best |
| CVDAvg only | 1 | 0 | Line above limit, dark green histogram | Good |
| CVDAcel only | 0 | 1 | Cyan histogram spike, line not confirming | Weakest |
| Neither | 0 | 0 | Dark green, line below | No signal |

CVD parameters: `CVDAvgLimFactor = 6`, `CVDAcelLimFactor = 3`

### Proposed Weighted Scoring System (next implementation)
Replace the C5/C6/C10/C13/RevATRsPerSec AND-gate with an entry score.
Keep C8 and C11 as mandatory hard gates regardless of score.

```pascal
Score_C5     = IFF(C5, W_C5, 0)           // Speed flip — strongest visual signal
Score_RevSec = IFF(C13, W_RevSec, 0)      // Fast reversal
Score_CVDAvg = IFF(C6, W_CVDAvg, 0)       // Volume in direction
Score_CVDAcl = IFF(C10, W_CVDAcl, 0)      // Volume acceleration
Score_CVDDlt = IFF(CVDDelta confirms, W_CVDDlt, 0)
Score_GapCV  = IFF(C9, W_GapCV, 0)        // Consistent HMA gap
Score_PSPct  = IFF(C7, W_PSPct, 0)        // Sustained prior trend speed

EntryScore = sum of all scores
// Gate: EntryScore >= HMinEntryScore AND C8 AND C11
```
Start with equal weights (W_x = 1, HMinEntryScore = 4 of 7).
Promote to unequal weights after analysis of first scored dataset.

---

## Key Formula Details (Current Code)

### Short (TurnDown) Branch
```pascal
TrendUpBarCount = MinList(BarNumber - HMASlowTurnUpBar - HMAFastTurnDnBars,
                          MaxBarsBack - HMAFastTurnDnBars);
FullDeltaPips   = (HMAFastValue[HMAFastTurnDnBars]
                  - HMAFastValue[HMAFastTurnDnBars + TrendUpBarCount]) / OnePip;
FullAngle       = Round(AtsCalculateAngle(HMAFastValue, TrendUpBarCount, 1)
                        [HMAFastTurnDnBars], 0);
FullDeltaATRs   = Round((HMAFastValue[HMAFastTurnDnBars]
                  - HMAFastValue[TrendUpBarCount + HMAFastTurnDnBars]) / BarATR, DecP);
PipSpeedPct     = 100 * Round(CountIf(PipSpeed >= PipSpeedLimit, TrendUpBarCount)
                              [HMAFastTurnDnBars] / TrendUpBarCount, 2);
HMAGapMean      = Average(HMAFastValue - HMASlowValue,
                          TrendUpBarCount - HMAFastTurnDnBars)[HMAFastTurnDnBars] / BarATR;
HMAGapCV        = IFF(HMAGapMean > 0, HMAGapStdDev / HMAGapMean, 99);
C5              = PipSpeed[1] >= PipSpeedLimit AND PipSpeed <= -1 * PipSpeedLimit;
```

### Long (TurnUp) Branch
```pascal
TrendDnBarCount = MinList(BarNumber - HMASlowTurnDnBar - HMAFastTurnUpBars,
                          MaxBarsBack - HMAFastTurnUpBars);
FullDeltaPips   = (HMAFastValue[HMAFastTurnUpBars + TrendDnBarCount]
                  - HMAFastValue[HMAFastTurnUpBars]) / OnePip;
FullAngle       = -1 * Round(AtsCalculateAngle(HMAFastValue, TrendDnBarCount, 1)
                             [HMAFastTurnUpBars], 0);
FullDeltaATRs   = Round((HMAFastValue[TrendDnBarCount + HMAFastTurnUpBars]
                  - HMAFastValue[HMAFastTurnUpBars]) / BarATR, DecP);
PipSpeedPct     = 100 * Round(CountIf(PipSpeed <= -1 * PipSpeedLimit, TrendDnBarCount)
                              [HMAFastTurnUpBars] / TrendDnBarCount, 2);
HMAGapMean      = Average(HMASlowValue - HMAFastValue,
                          TrendDnBarCount - HMAFastTurnUpBars)[HMAFastTurnUpBars] / BarATR;
HMAGapCV        = IFF(HMAGapMean > 0, HMAGapStdDev / HMAGapMean, 99);
C5              = AbsValue(PipSpeed[1]) >= PipSpeedLimit AND PipSpeed >= PipSpeedLimit;
```

### Known Formula Issues to Watch
- `HMAGap` uses window `TrendBarCount - HMAFastTurnBars` but `PipSpeedPct` uses full
  `TrendBarCount`. Inconsistent window — pick one convention and apply to both.
- Guard `TrendBarCount` against zero: `MaxList(1, MinList(...))`.

---

## Analysis History and Statistical Findings

### Datasets Summary
| Dataset | Trades | ProfitHit Rate | Net P/L | Key Change |
|---|---|---|---|---|
| 1 | 316 | 31.3% | +$2,510 | Full C1-C13 active |
| 2 | 812 | 20.2% | -$11,479 | Conditions partially loosened |
| 3 | 1,566 | 28.2% | — | C5,C7,C8,C10,C11,C13 = always true |
| 4 | 1,499 | 30.4% | — | Full analysis with correct exit interpretation |

### Confirmed Statistical Results (Dataset 4 — most reliable)
Analysis method: Mann-Whitney U test, ProfitHit vs PureLoss across ALL exit types.

**LONGS (779 non-scratch trades, 29.1% profit-hit rate):**
| Parameter | ProfitHit med | PureLoss med | p-value | Verdict |
|---|---|---|---|---|
| ind_FullDeltaATRs | 7.20 | 6.70 | **0.007 ★★★** | Hard gate |
| ind_DeltaATRs | 6.90 | 6.50 | **0.002 ★★★** | Hard gate |
| ind_FullAngle | 26.0 | 25.0 | **0.004 ★★★** | Hard gate |
| ind_Angle | 27.0 | 26.0 | **0.007 ★★★** | Hard gate |
| All CVD metrics | ~flat | ~flat | 0.34–0.97 | Score only |
| ind_ATRsFromHma | 0.63 | 0.61 | 0.258 | No signal — longs |
| ind_RevATRsPerSec | 0.12 | 0.08 | 0.297 | Score only |

**SHORTS (700 non-scratch trades, 32.7% profit-hit rate):**
| Parameter | ProfitHit med | PureLoss med | p-value | Verdict |
|---|---|---|---|---|
| ind_ATRsFromHma | 0.62 | 0.53 | **0.009 ★★★** | Hard gate |
| ind_DeltaATRs | 6.70 | 6.40 | 0.081 ★ | Secondary |
| ind_FullAngle | 26 | 25 | 0.260 | No signal — shorts |
| All CVD metrics | ~flat | ~flat | 0.22–0.81 | Score only |

### PipSpeed Findings (current dataset — transition signal)
| Condition | Long WR | Short WR |
|---|---|---|
| C8=1 AND \|PipSpeed\| ≤ 0.5 at signal | **40.7%** | **40.0%** |
| C8=1 AND \|PipSpeed\| ≤ 1.5 at signal | 35.9% | 37.8% |
| C8=1 AND \|PipSpeed\| > 1.5 at signal | 26.8% | 26.9% |
| Baseline | 29.3% | 32.7% |

Longs with PipSpeed still negative at signal: **11.1% win rate** (trend not slowed).
Shorts with PipSpeed still strongly positive: **15% win rate**.
The transition (C5) is the most important PipSpeed signal.

### Best Filtered Combinations (Dataset 4)
**Longs:**
| Filter | n | ProfitHit rate | Expectancy |
|---|---|---|---|
| Baseline | 724 | 21.8% | -$5.27 |
| DeltaATRs ≥ 7.5 & FullAngle ≥ 26 | 127 | 35.4% | +$0.25 |
| DeltaATRs ≥ 8 & FullAngle ≥ 26 | 93 | 35.5% | +$1.11 |
| DeltaATRs ≥ 9 & FullAngle ≥ 26 | 53 | 39.6% | +$2.03 |

**Shorts:**
| Filter | n | ProfitHit rate | Expectancy |
|---|---|---|---|
| Baseline | 626 | 23.8% | -$4.57 |
| ATRsFromHma ≥ 0.8 | 174 | 30.5% | positive |
| ATRsFromHma ≥ 0.7 & DeltaATRs ≥ 8 | 57 | 36.8% | +$1.05 |

---

## Visual Fingerprint — Winners vs Losers

### Winners (FTNT-6/1, NVDL-6/12 11:53)
- Prior trend: long, steep, clean — RED HMA at 45°+ angle, large separation from CYAN
- PipSpeed panel: **sustained CYAN blocks** across most of the prior trend bars
- Transition bar: YELLOW dot — speed crossed zero cleanly in one bar
- CVDAcel histogram: sustained cyan during prior trend and at the turn
- CVDDelta yellow dots: large concentrated spike at the turn bar
- HMA fast/slow gap: large and consistent throughout the trend (low HMAGapCV)

### Losers (MRNA 11:36, NVDL-6/10 11:56 and 12:49)
- Prior trend: short, shallow, or a minor bounce within an existing opposite trend
- PipSpeed panel: mostly MAGENTA or brief/weak cyan — not sustained
- CVDAcel histogram: dark green or brief spike only
- CVDDelta yellow dots: small and scattered
- HMA fast/slow gap: small — RED barely separated from CYAN (low DeltaATRs)

### Key Loser Pattern — Counter-Trend Bounce
The MRNA 11:36 trade was shorting a small bounce within an already established downtrend.
The CYAN slow HMA was already pointing sharply down. The RED fast barely lifted above the
CYAN before rolling over. This produces low DeltaATRs and shallow FullAngle regardless of
what other conditions show. Primary filters that catch this: DeltaATRs and FullAngle.

**Potential additional filter:** Check CYAN slow HMA direction at signal time — a slow HMA
already pointing strongly in the same direction as the entry means the "reversal" is just
a counter-trend bounce, not a genuine trend reversal.

---

## Parameter Change Log

### Currently Active Parameters
```
PipSpeedLimit    = 0.30
RevATRsPerSecLim = 0.10
CVDAvgLimFactor  = 6
CVDAcelLimFactor = 3
ATRsFromHmaLim   = 1.0
HMinAngleLim     = 25     (applied to FullAngle)
HMinATRs         = 7      (applied to FullDeltaATRs)
HMinHMAGapCV     = 0.40   (maximum allowed — LOW CV = quality trend)
HMinPipSpeedPct  = 40     (minimum % of trend bars above speed limit)
```

### Confirmed Retired / No Signal
| Metric | Reason |
|---|---|
| BarPct (MaxLength-based) | No signal after BarBreak bug fixed |
| BarBreak | No signal after formula restored |
| TrendBarCount as gate | No signal in any dataset |
| Raw CVDAvg / CVDAcel values | Signal is state (cyan/not cyan), not scalar value |
| DeltaPips / FullDeltaPips as gate | No signal after normalization |
| RevATRsPerSec as hard gate | Better as weighted score component |
| MaxLength-based Angle/DeltaATRs | Replaced by FullAngle / FullDeltaATRs |

### Changes Made Across Runs
| Change | Dataset | Outcome |
|---|---|---|
| DeltaATRs >= 5 minimum | 4 | Confirmed direction correct; raised to 7 for FullDeltaATRs |
| BarPct/BarBreak → TrendBarCount | 4 | Both confirmed flat — retired |
| FullDeltaATRs, FullDeltaPips added | 4 | FullDeltaATRs highly significant (p=0.007) longs |
| Trailing stop delay | 4 | Still 90% stop exits; monitoring |
| C2 → FullDeltaATRs, C3 → FullDeltaPips | Current | Removes MaxLength cap artifact |
| C4 → FullAngle | Current | Full trend angle, not capped window |
| PipSpeedPct added (replaces BarPct) | Current | Awaiting first results |
| HMAGapCV added | Current | Awaiting first results |
| C5 redefined as speed flip | Current | Confirmed by visual analysis |
| ATRsFromHmaLim raised to 1.0 | Current | Shorts: higher ATRsFromHma = better (p=0.009) |

---

## Available Tools

### ats_performance_report.py
TradeStation-style performance report from the merged CSV.
```bash
python ats_performance_report.py <csv_file> [--direction long|short|both]
       [--symbol TICKER] [--start mm/dd/yyyy] [--end mm/dd/yyyy] [--output summary.csv]
```
Outputs: Net P/L, Profit Factor, Expectancy, Sharpe, Sortino, Calmar,
Max Drawdown, Max Capital Required, Win/Loss detail, daily breakdown, exit breakdown.

### ats_param_analysis.py  (to be created)
Pre-computes all parameter statistics and outputs a summary table for review.
Designed to minimize token usage — run locally and share output table only.

---

## Open Questions / Next Steps

1. **HMAGapCV and PipSpeedPct thresholds** — first results with new metrics will determine
   if 0.40 and 40% are the right gates or need adjustment.
2. **Weighted scoring implementation** — replace C5/C6/C10/C13 AND-gate with EntryScore.
   Implement after confirming new metrics have data.
3. **Long/short parameter split** — DeltaATRs optimal threshold differs by direction.
   Consider HMinATRs_Long / HMinATRs_Short variants.
4. **HMA Slow direction filter** — detect counter-trend bounce pattern by checking
   if slow HMA is already pointing in the entry direction at signal time.
5. **PipSpeedPct window inconsistency** — HMAGap uses TrendBarCount - TurnBars,
   PipSpeedPct uses full TrendBarCount. Standardize.
6. **TrendBarCount zero guard** — wrap with MaxList(1, MinList(...)) in both branches.
7. **RevATRsPerSec vs P/L magnitude** — test correlation within ProfitHit subset only.
   May predict size of win rather than whether any profit is hit.
8. **AtsPriceBrkout analysis** — CSV available but not yet analyzed in depth.
   Known: same parameters, different C12 (slow already crossed before turn).

---

## Analysis Methodology

- **Outcome:** `ProfitHit = P/L > 0`, `PureLoss = P/L < 0` across ALL exit types
- **Primary test:** Mann-Whitney U (non-parametric, handles non-normal distributions)
- **Secondary:** Quartile buckets, threshold sweeps, Spearman correlation vs P/L
- **Optimize for:** Expectancy = win_rate × avg_win + loss_rate × avg_loss
- **Always split by:** `ind_SignalSent` (1=long, -1=short) before drawing conclusions
- **Validation rule:** All thresholds in-sample only — require forward-test confirmation
- **Overfitting guard:** Minimum 30 trades in any filtered subset before trusting a result
- **Time-series rule:** Train/test splits must be chronological, never random
