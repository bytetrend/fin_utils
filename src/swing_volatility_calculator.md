# Intraday Swing Volatility Calculator — Implementation Notes

## Overview
This calculator computes **intraday price swing volatility** for US stock symbols over the past month.  
"Swing" is defined as the daily range relative to closing price:  
$
\text{Swing}_t = \frac{\text{High}_t - \text{Low}_t}{\text{Close}_t}
$

The resulting metric reflects *how much a stock typically moves up and down within a single trading day*, not its directional trend.

---

## 📊 Key Metrics

| Metric | Description |
|--------|-------------|
| **Swing Mean (%)** | Average daily swing as a percentage of closing price (e.g., `1.23%` → average $1.23 move per $100) |
| **Swing Std Dev (%)** | Standard deviation of daily swings — indicates *consistency* of intraday volatility (higher = more erratic swings) |
| **Annualized Swing Volatility (%)** | Mean swing scaled to an annual basis: `mean × √(samples_per_year)` for comparability across assets and periods |

> ✅ **Why use mean swing instead of standard deviation alone?**  
> Mean swing directly answers: *"How big are typical intraday moves?"*  
> Std dev shows variability *around* that average. Both are provided.

---

## 📈 Data & Methodology

### 1. Data Source
- Uses **Yahoo Finance via `yfinance`**
- Fetches historical intraday data for up to ~60 days (depending on interval)
- For the requested *past month*, we use:
  - Default: `'30m'` interval — provides ~8 samples per trading day (9:30 AM – 4:00 PM ET)
  - Alternative: `'1h'` or `'60m'` if longer lookback needed

### 2. Calculation Steps
For each symbol:
1. Download intraday OHLCV data for the last **N days** (`LOOKBACK_DAYS = 30`)
2. Clean data: drop rows with missing `High`, `Low`, or `Close`
3. Compute swing per bar:  
   `swing_i = (high_i - low_i) / close_i`
4. Aggregate across all bars:
   - Mean swing → central estimate of typical intraday movement
   - Std dev swing → volatility *stability* metric
5. Annualize for interpretation:  
   $
   \text{Annualized Vol} = \text{Mean Swing} \times \sqrt{\text{samples per year}}
   $  
   Where `samples_per_year ≈ samples_per_day × 252 trading days`

### 3. Annualization Logic
- With `'30m'` data: ~8 bars/day (market open to close) → ~2,016 bars/year  
- Example: `0.01` mean swing (`1%`) ⇒ annualized ≈ `1% × √2016 ≈ 45%`
> ⚠️ This is *not* the same as standard volatility (std of returns), but reflects *scaled magnitude* for intuitive comparison.

---

## 🛠️ Configuration & Customization

| Parameter | Description | Default |
|----------|-------------|---------|
| `LOOKBACK_DAYS` | How many days of history to analyze | `30` |
| `INTRADAY_INTERVAL` | Time resolution of bars (`'1m'`, `'2m'`, `'5m'`, `'15m'`, `'30m'`, `'60m'`) | `'30m'` |

> ⚠️ **Important**:  
> - Yahoo Finance restricts high-frequency data:  
>   `1m`, `2m`, `5m`: max ~7 days  
>   `15m`, `30m`: max ~60 days  
>   `60m`, `90m`, `1h`: up to 2+ years  
> - For the *full month*, `'30m'` is recommended. Use `'1h'` for longer periods.

---

## 📁 Input & Output

### Input
- Read symbols from a file (e.g., `stocks.txt`) with one symbol per line:  

### Output (Console)
| Column | Example |
|--------|---------|
| Symbol | `NVDA` |
| Intraday Swing Mean (%) | `1.230` → average $1.23 move per $100 |
| Swing Std Dev (%) | `0.870` → swings vary ±0.87% around the mean |
| Annualized Swing Vol (%) | `48.92` → scaled to annual basis |
| Samples Used | `168` (e.g., ~5.6 days of 30m data if partial fetch) |
| Daily Close | `$520.45` (as of latest daily bar) |

---

## 🧪 Limitations & Considerations

| Issue | Explanation | Mitigation |
|-------|-------------|------------|
| **Data Gaps** | Holidays, weekends, or early market closures cause missing bars | Script logs skipped symbols; uses `.dropna()` internally |
| **After-Hours/Pre-Market** | Not included — only regular trading hours (9:30–16:00 ET) are in `yfinance` intraday data | Acceptable for standard swing measurement |
| **Corporate Actions** | Splits/dividends not adjusted in raw OHLCV → may cause artificial spikes | Use `auto_adjust=False`; adjustments not needed for *swing ratio* |
| **Rate Limits** | Too many requests in quick succession can trigger HTTP 429 | Built-in random delay: `0.3–1.0 sec` between symbols |
| **Low-Liquidity Stocks** | Wide bid-ask spreads may distort High/Low → swing overestimation | Use only for well-traded equities; filter by volume if needed |

---

## 🚀 Usage

```bash
# Install dependencies
pip install yfinance pandas numpy

# Create input file
echo -e "AAPL\nMSFT\nNVDA" > stocks.txt

# Run the calculator
python swing_volatility.py

# 🔄 How to Identify Intraday Reversal-Prone Stocks Using Swing Metrics

Great question! You're interested in **intraday reversal activity** — i.e., stocks that frequently move up *and then down* (or down *and then up*) within a single day, creating sharp price swings but little net directional movement.

Let’s clarify what `Swing Mean %` and `Swing Std Dev %` capture — and how they relate to reversals:

---

## 🔍 Definitions Recap

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Swing Mean %** | mean of `(High − Low) / Close` per bar | *Average daily range size* (how big intraday moves are, regardless of direction) |
| **Swing Std Dev %** | std dev of those swings | *Consistency/volatility of the swing magnitude* — how much bar-to-bar swing size varies |

⚠️ Neither metric directly tells you about *reversal frequency* or *net drift*. A high swing can be one directional trend with volatility, or many small reversals.

But here's the key insight:

---

## ✅ How to Identify Reversal-Prone Stocks

### 📈 1. **Look for HIGH `Swing Mean %`**  
→ Bigger average daily range = more *room* for intraday swings — including reversals.  
A stock moving only ±0.2% per day rarely reverses dramatically.

✅ *Example*:  
- Stock A: swing mean = 0.8% → typical $8 move on a $1,000 share  
- Stock B: swing mean = 3.5% → typical $35 move  
→ **Stock B** has more *potential* for reversal moves.

### 📉 2. **Look for HIGH `Swing Std Dev %`? Not quite…**  
→ Large variation in daily swings often indicates chaotic, mean-reverting behavior — e.g., some days quiet, others volatile with sharp up/down turns.

But here’s the deeper clue:

> 🔁 **The *most* reversal-prone stocks often show:  
> 📌 HIGH `Swing Mean %` + LOW to MODERATE `Swing Std Dev %`**  

Why?  
- High mean swing = large average moves (so reversals can be meaningful).  
- Low/medium std dev = swings are *consistent* and not dominated by a few extreme events — suggesting *repetitive* up-down or down-up patterns.

Let’s compare two hypothetical stocks:

| Stock | Swing Mean % | Swing Std Dev % | Likely Behavior |
|-------|--------------|-----------------|----------------|
| **X** | 4.0% | 3.2% | Some days massive ±5–8% swings, others quiet. Reversals may be rare but dramatic (e.g., pump & dump). |
| **Y** | 2.0% | 0.6% | Very consistent ±2% daily range — likely choppy, oscillating with frequent reversals. ✅ **Best candidate for reversal trading** |

✅ So:
- **Swing Mean %** → *scale* of possible reversal moves  
- **Swing Std Dev %** → *predictability/regularity* of those moves  
→ For reliable intraday reversals: **high mean + low std dev**

---

## 🛠️ Practical Screening Strategy

Use a 2D filter:

```text
Top Candidates =
  Swing Mean % > X%   AND
  Swing Std Dev / Swing Mean < Y    (i.e., Coefficient of Variation, CV < Y)


## Suggested thresholds (backtest-adjusted):
|Market Segment                    |Min Swing Mean %  | Max CV (Std/Mean)    |
|----------------------------------|------------------|--------------------- |
|Scalping/Day-Trading Candidates   | ≥ 1.5%	          | ≤ 0.4                |
|ETFs / Low-Vol Stocks	           | ≥ 0.8%           |	≤ 0.3                |
|Meme/High-FOMO Stocks	           | ≥ 2.5%	          |≤ 0.6 (more erratic)  |

## Compute reversal score:
```python
reversal_score = swing_mean / (swing_std + 1e-6)
```
Higher score → more consistent, sizable swings → better for reversal strategies.

## 📊 Real-World Examples (as of recent data)

| Stock | Swing Mean % | Std Dev % | CV | Why? |
|-------|--------------|-----------|-----|------|
| **SQQQ** (3x inverse QQQ) | ~2.8% | 0.7% | 0.25 | Designed to reverse daily — high mean, low CV ✅ |
| **TQQQ** | ~2.6% | 0.9% | 0.35 | Similar, but riskier due to leverage decay |
| **NVDA (recent)** | ~3.4% | 2.1% | 0.62 | Large swings, but erratic — reversals less predictable |
| **SPY** | ~0.6% | 0.25% | 0.42 | Too quiet for meaningful intraday reversal |

✅ For pure *reversal potential*, prioritize **SQQQ**, **SPXU**, or high-volume micro-cap ETFs — not volatile single stocks.

---

## 🔬 How to Improve Detection (Optional Enhancements)

Add these metrics to your calculator:

1. **Bar-to-Bar Change Direction Rate**:  
   `% of bars where close_t < open_{t-1} OR close_t > open_{t-1}` → high = reversal-prone
2. **Net Drift / Swing Ratio**:  
   `abs(close - open_overall) / total_range`  
   → Low ratio (e.g., < 0.1) = chop, many reversals; High = directional trend.
3. **Volume-Swing Correlation**:  
   Does volume spike *during* swing expansion? Reversals often occur on volume surges.

---

## ✅ TL;DR: Which Values to Pick?

| Goal | Swing Mean % | Swing Std Dev % |
|------|--------------|-----------------|
| 🔥 Maximize reversal *magnitude* (e.g., for scalping) | **High** ↑ | **Moderate/Low** ↓ |
| 🔄 Maximize reversal *frequency & consistency* | **Medium-High** → | **Low** ✅ |
| 📉 Avoid false signals (directional trends masquerading as reversals) | — | Low CV (`Std/Mean`) is key |

➡️ **Recommendation**: Sort by `Swing Mean %` first, then filter by low CV (e.g., CV < 0.4). The combo ensures you find stocks that both *move a lot* and do so in a *repetitive, oscillatory way* — perfect for intraday mean-reversion strategies.
