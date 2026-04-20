#!/usr/bin/env python3
"""
Volatility & Beta Analyzer with Daily Volatility Heatmap
- Calculates daily intra-day swing volatility (last 30 trading days)
- Computes beta vs SPY (with stats)
- Exports: CSV report + HTML heatmap + JSON metadata
"""

import os, time
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats as scipy_stats
from argparse import ArgumentParser

from constants import OUTPUT_DIR

LOOKBACK_DAYS = 30

# Thresholds for reversal detection (as % of swing)
REVERSAL_THRESHOLD_LOW = 0.2   # close within bottom 20% of daily range → bullish
REVERSAL_THRESHOLD_HIGH = 0.2  # close within top    20% of daily range → bearish


def load_symbols(filepath: str) -> list:
    with open(filepath, 'r') as f:
        syms = [l.strip().upper() for l in f if l.strip()]
    return syms


def filter_by_price_volume(symbol: str,
                           min_price=None, max_price=None,
                           min_vol=None, max_vol=None):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.history(period="1mo", interval="1d").iloc[-1:]
        if info.empty:
            return False, {}
        price = float(info['Close'].iloc[0])
        vol_avg = float(info['Volume'].iloc[0])  # fallback

        cond = True
        if min_price and price < min_price: cond = False
        if max_price and price > max_price: cond = False
        if min_vol and vol_avg < min_vol: cond = False
        if max_vol and vol_avg > max_vol: cond = False

        return cond, {"Price": round(price, 2), "Avg Vol (10d)": int(vol_avg)}
    except Exception:
        return False, {}


def get_intraday_data(ticker) -> pd.DataFrame:
    # Fetch 1-day intervals for last 5 days * 6 = 30 trading days approx
    try:
        hist = ticker.history(period="5d", interval="30m")  # market open 9:30–16:00 ET → ~7 bars/day
        if hist.empty:
            return pd.DataFrame()
        # Ensure required columns exist
        for col in ['Open', 'High', 'Low', 'Close']:
            assert col in hist.columns, f"Missing column: {col}"
        return hist
    except Exception as e:
        print(f"[!] Intraday fetch failed: {e}")
        return pd.DataFrame()


def calculate_volatility(df: pd.DataFrame) -> dict:
    # Daily-level metrics (group by trading day)
    df['date'] = df.index.date
    daily_df = df.groupby('date').agg(
        open=('Open', 'first'),
        high=('High', 'max'),
        low=('Low', 'min'),
        close=('Close', 'last')
    ).reset_index(drop=True)

    # Intraday swing per day: (high - low) / close
    daily_df['swing_pct'] = (daily_df['high'] - daily_df['low']) / daily_df['close']
    if len(daily_df) == 0:
        return {
            'swing_mean': 0, 'swing_std': 0,
            'cv_swing': np.nan, 'skew_swing': np.nan,
            'max_swing': 0, 'reversal_ratio': 0,
            'reversal_score': 0, 'annualized_volatility': 0,
            'sample_count': 0
        }

    mean_swing = daily_df['swing_pct'].mean()
    std_swing = daily_df['swing_pct'].std()
    cv = std_swing / mean_swing if mean_swing else np.nan

    skew_val = float(scipy_stats.skew(daily_df['swing_pct'], nan_policy='omit'))

    # Reversal detection (same as before, but using daily data)
    daily_df['close_from_low_pct'] = (daily_df['close'] - daily_df['low']) / (daily_df['high'] - daily_df['low']).replace(0, np.nan)
    daily_df['close_from_high_pct'] = (daily_df['high'] - daily_df['close']) / (daily_df['high'] - daily_df['low']).replace(0, np.nan)

    bullish_rev = (daily_df['close'] < daily_df['open']) & \
                  (daily_df['close_from_low_pct'] <= REVERSAL_THRESHOLD_LOW)
    bearish_rev = (daily_df['close'] > daily_df['open']) & \
                  (daily_df['close_from_high_pct'] <= REVERSAL_THRESHOLD_HIGH)

    reversal_count = bullish_rev.sum() + bearish_rev.sum()
    rev_ratio = reversal_count / len(daily_df) if len(daily_df) else 0

    # Reversal score
    swing_score = min(mean_swing * 200, 100)
    reversal_score = round((rev_ratio * 100) + (abs(skew_val) * 5) + (swing_score / 2), 2)

    # Annualization: daily swings → annualized
    samples_per_day = len(df) / LOOKBACK_DAYS if LOOKBACK_DAYS > 0 else 1
    annual_factor = np.sqrt(samples_per_day * 252)
    ann_vol = mean_swing * annual_factor

    return {
        'swing_mean': mean_swing,
        'swing_std': std_swing,
        'cv_swing': round(cv, 4) if not np.isnan(cv) else None,
        'skew_swing': round(skew_val, 4),
        'max_swing': (daily_df['high'] - daily_df['low']).max() / daily_df['close'].iloc[
            (daily_df['high'] - daily_df['low']).idxmax()] if not daily_df.empty else 0,
        'reversal_ratio': rev_ratio * 100,  # %
        'reversal_score': reversal_score,
        'annualized_volatility': ann_vol,
        'sample_count': len(df),
    }


# ----------------------------
# NEW: Daily volatility heatmap builder
# ----------------------------
def build_daily_heatmap(symbols: list) -> pd.DataFrame:
    """
    Returns DataFrame: rows = dates (most recent first), columns = symbols, values = daily swing % (std dev)
    Uses 1-day intervals for last 30 days per symbol.
    Falls back to monthly data if intraday not available.
    """
    # Collect last N trading days across all symbols
    print("\n🛠️ Building daily volatility heatmap...")

    date_list = []
    result_dict = {}

    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="3mo", interval="1d")
            if hist.empty or len(hist) < 5:
                continue
            # Take most recent N days
            recent = hist.tail(LOOKBACK_DAYS).copy()
            recent['swing'] = (recent['High'] - recent['Low']) / recent['Close']
            dates = pd.to_datetime(recent.index.date)
            for d in dates.unique():
                if d not in date_list:
                    date_list.append(d)

            # Map: {date: daily swing %}
            daily_swings = dict(zip(dates, recent.set_index(dates)['swing']))
            result_dict[sym] = daily_swings

        except Exception as e:
            print(f"⚠️  Error fetching heatmap data for {sym}: {e}")

    if not date_list:
        return pd.DataFrame()

    # Sort dates descending (most recent first)
    date_list.sort(reverse=True)

    # Align all symbols to same dates
    rows = []
    for d in date_list:
        row = {}
        for sym, ddict in result_dict.items():
            row[sym] = ddict.get(d, np.nan) * 100  # %
        rows.append(row)

    return pd.DataFrame(rows, index=pd.to_datetime(date_list))


def export_heatmap(df: pd.DataFrame, output_dir=OUTPUT_DIR):
    """Export HTML heatmap with color-coding (green=low volatility, red=high)."""
    if df.empty:
        print("ℹ️ No data to build heatmap.")
        return

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "daily_volatility_heatmap.html")

    # Define styling
    styles = [
        dict(selector="th", props=[("border", "1px solid #ddd"), ("padding", "8px")]),
        dict(selector="td", props=[("text-align", "center"), ("padding", "6px"),
                                   ("font-size", "9pt"), ("color", "#333")]),
    ]
    cmap = "RdYlGn_r"  # reversed: red (high) → yellow → green (low)

    html = (
        df.style
          .format("{:.2f}%")
          .background_gradient(cmap=cmap, axis=None, vmin=0.5, vmax=3.0)
          .set_caption(f"Daily Intra-Day Volatility Heatmap (Last {LOOKBACK_DAYS} Trading Days)")
          .set_table_styles(styles)
    ).to_html()

    with open(filepath, 'w') as f:
        f.write(html)

    print(f"✅ Exported daily volatility heatmap to: {filepath}")
    return filepath


# ----------------------------
# NEW: Beta vs SPY
# ----------------------------
def compute_beta_vs_spy(symbol: str, days=30) -> dict:
    """
    Compute beta & alpha of symbol vs SPY over last N days.
    Uses daily returns. Falls back to monthly if insufficient data.
    """
    try:
        # Get daily prices (most recent)
        sym_hist = yf.Ticker(symbol).history(period=f"{int(days*1.5)}d", interval="1d")
        spy_hist  = yf.Ticker("SPY").history( period=f"{int(days*1.5)}d", interval="1d")

        if len(sym_hist) < days or len(spy_hist) < days:
            return {"beta": np.nan, "alpha": np.nan, "r2": np.nan,
                    "pvalue_beta": np.nan, "count": 0}

        # Align to same dates
        sym_ret = sym_hist['Close'].pct_change().dropna()
        spy_ret = spy_hist['Close'].pct_change().dropna()

        # Trim both to most recent 'days' daily returns
        n_common = min(len(sym_ret), len(spy_ret))
        if n_common < 10:
            return {"beta": np.nan, "alpha": np.nan, "r2": np.nan,
                    "pvalue_beta": np.nan, "count": n_common}

        sym_r = sym_ret.tail(days).values
        spy_r = spy_ret.tail(days).values

        # OLS regression: stock_ret = alpha + beta * spy_ret + ε
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(spy_r, sym_r)

        return {
            "beta": round(slope, 3),
            "alpha": round(intercept * 252, 4),  # annualized alpha (approx)
            "r2": round(r_value ** 2, 3),
            "pvalue_beta": round(p_value, 4),
            "count": len(sym_r),
        }

    except Exception as e:
        print(f"[!] Beta calc failed for {symbol}: {e}")
        return {"beta": np.nan, "alpha": np.nan, "r2": np.nan,
                "pvalue_beta": np.nan, "count": 0}


# ----------------------------
# MAIN: Main execution loop
# ----------------------------
def main():
    parser = ArgumentParser(description="Calculate intra-day volatility & beta vs SPY")
    parser.add_argument("-f", "--file", required=True, help="File containing symbols (one per line)")
    parser.add_argument("--min-price", type=float, default=5.0, help="Min price filter")
    parser.add_argument("--max-price", type=float, default=None)
    parser.add_argument("--min-vol",   type=int,   default=100_000, help="Min avg volume (3mo)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    symbols = load_symbols(args.file)
    print(f"🔍 Processing {len(symbols)} symbols...")

    results = []
    # --- Step 1: Compute per-stock metrics + beta ---
    for i, sym in enumerate(symbols):
        if (i+1) % 5 == 0:
            print(f"  {i+1}/{len(symbols)} processed...")

        is_valid, info = filter_by_price_volume(
            sym,
            min_price=args.min_price,
            max_price=args.max_price,
            min_vol=args.min_vol
        )
        if not is_valid:
            continue

        df = get_intraday_data(yf.Ticker(sym))
        vol_metrics = calculate_volatility(df) if not df.empty else {}

        # Beta vs SPY (last 30 days)
        beta_info = compute_beta_vs_spy(sym, days=LOOKBACK_DAYS)

        result = {
            "Symbol": sym,
            **info,
            "Volatility (%)": round(vol_metrics.get("swing_mean", np.nan) * 100, 2),
            "Volatility StdDev (%)": round(vol_metrics.get("swing_std", np.nan) * 100, 2),
            "Reversal Ratio (%)": round(vol_metrics.get("reversal_ratio", np.nan), 2),
            "Beta vs SPY (30d)": beta_info["beta"],
            "Alpha (annualized)": beta_info["alpha"],
            "Beta R²": beta_info["r2"],
            "Beta p-value": beta_info["pvalue_beta"],
            "Count of Bars Used": vol_metrics.get("sample_count", 0),
        }
        results.append(result)
        time.sleep(60)  # be nice to API

    # --- Step 2: Export main CSV report ---
    df_results = pd.DataFrame(results)
    # Sort by reversal ratio descending (most reversals first)
    df_sorted = df_results.sort_values(by="Reversal Ratio (%)", ascending=False)
    # Example hybrid score: weight reversal ratio most heavily
    df_sorted['ReversalScore'] = (
            0.6 * df["Reversal Ratio (%)"] +
            0.2 * df["Volatility StdDev (%)"] +
            0.1 * abs(df["Beta vs SPY (30d)"]) +
            0.1 * (df["Alpha (annualized)"] < 0).astype(int) * 5
    )

    # Then sort:
    df = df_sorted.sort_values("ReversalScore", ascending=False)

    if not df_results.empty:
        csv_path = os.path.join(OUTPUT_DIR, "volatility_beta_scored_report.csv")
        df_sorted.to_csv(csv_path, index=False)
        print(f"\n✅ CSV Report saved: {csv_path}")

    # --- Step 3: Build & export heatmap ---
    try:
        heatmap_df = build_daily_heatmap([r["Symbol"] for r in results])
        if not heatmap_df.empty:
            export_heatmap(heatmap_df, OUTPUT_DIR)
    except Exception as e:
        print(f"[!] Heatmap generation skipped: {e}")

    # --- Step 4 (Bonus): Export JSON metadata ---
    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "symbols_processed": len(results),
        "lookback_days": LOOKBACK_DAYS
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), 'w') as f:
        import json; json.dump(meta, f, indent=2)

    print(f"\n🏁 Done! Output files in `{OUTPUT_DIR}/`:")
    if os.path.exists(os.path.join(OUTPUT_DIR, "volatility_beta_report.csv")):
        print("  • volatility_beta_report.csv")
    if os.path.exists(os.path.join(OUTPUT_DIR, "daily_volatility_heatmap.html")):
        print("  • daily_volatility_heatmap.html (open in browser to view)")
    if os.path.exists(os.path.join(OUTPUT_DIR, "metadata.json")):
        print("  • metadata.json")

if __name__ == "__main__":
    main()
