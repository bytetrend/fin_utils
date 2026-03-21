import os
import random
import time
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf

from constants import OUTPUT_DIR

# ----------------------------
# CONFIGURATION
# ----------------------------
LOOKBACK_DAYS = 30  # Look back this many days
INTRADAY_INTERVAL = '60m'  # Valid: '1m', '2m', '5m', '15m', '30m', '60m', '90m'


# Note: >7 days only available for intervals >= 1h or 60m
# For full 30d, we use '1h' or '60m' (best balance)

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def load_symbols(filename):
    """Read stock symbols from a file (one per line)."""
    try:
        with open(filename, 'r') as f:
            symbols = [line.strip().upper() for line in f if line.strip()]
        return list(set(symbols))  # remove duplicates
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{filename}' not found.")


def get_intraday_data(ticker, period_days=LOOKBACK_DAYS):
    """
    Fetches intraday data (up to ~60-75 days max depending on interval).
    Returns DataFrame with columns: ['High', 'Low', 'Close']
    """
    # For > 7 days, yfinance uses daily data unless we use specific intervals
    # But note: '1h' and '60m' give up to ~2 years; '30m' gives ~60 days.
    # We cap period_days <= 60 for safety with 30m interval.

    data = ticker.history(
        period=f"{min(period_days, 60)}d",
        interval=INTRADAY_INTERVAL,
        auto_adjust=False
    )
    if data.empty:
        return pd.DataFrame()

    # Keep only required columns and drop NaN rows in High/Low/Close
    data = data[['High', 'Low', 'Close']].dropna()
    return data


def calculate_volatility(df):
    """
    Calculate swing-based volatility:
      - Swing_i = (High_i - Low_i) / Close_i
      - Volatility = mean(swing) * sqrt(#samples_per_year)

    Also returns std of swings as alternative metric.
    """
    if df.empty or len(df) < 2:
        return {'swing_mean': np.nan, 'swing_std': np.nan, 'sample_count': 0}

    # Intraday swing (as % of close)
    df['swing'] = (df['High'] - df['Low']) / df['Close']

    # Basic stats
    mean_swing = df['swing'].mean()
    std_swing = df['swing'].std()

    # Annualize: assume 252 trading days and ~X samples per day
    # For interval '30m': 8 samples/day (9:30-16:00 EST)
    # We'll use actual count instead of hardcode
    samples_per_day = len(df) / LOOKBACK_DAYS  # avg samples per day
    annual_factor = np.sqrt(samples_per_day * 252) if samples_per_day > 0 else 1

    return {
        'swing_mean': mean_swing,
        'swing_std': std_swing,
        'sample_count': len(df),
        'annualized_volatility': mean_swing * annual_factor
    }


def get_daily_close_from_yahoo(ticker):
    """Quick daily close to help interpret swing vol."""
    try:
        hist = ticker.history(period="1d")
        if not hist.empty and 'Close' in hist.columns:
            return hist['Close'].iloc[-1]
    except Exception:
        pass
    return np.nan


def calculate_symbol_volatility(symbols: List[str]) -> List[Dict[str, Any]]:
    results = []
    print("\n📊 Fetching and calculating volatility for each symbol...\n")
    for i, symbol in enumerate(symbols):
        print(f"[{i + 1}/{len(symbols)}] Processing {symbol}...", end=" ")
        try:
            ticker = yf.Ticker(symbol)
            # Get stock info
            info = ticker.info
            if len(info) <= 1:
                raise ValueError(f"No data found for ticker {ticker}")
            df = get_intraday_data(ticker)
            if not df.empty:
                vol_result = calculate_volatility(df.copy())

                # Add context: recent daily price
                daily_close = get_daily_close_from_yahoo(ticker)

                results.append({
                    'Symbol': symbol,
                    'Intraday Swing Mean (%)': round(vol_result['swing_mean'] * 100, 3),
                    'Swing Std Dev (%)': round(vol_result['swing_std'] * 100, 3) if vol_result['swing_std'] else np.nan,
                    'Annualized Swing Vol (%)': round(vol_result.get('annualized_volatility', np.nan) * 100, 2),
                    'Samples Used': int(vol_result['sample_count']),
                    'Daily Close': round(daily_close, 2),
                    'Beta (5Y Monthly)': info.get('beta', None),
                    '52 Week Change': info.get('52WeekChange', None),
                    'Avg Vol (3 month)': info.get('averageVolume', None),
                    'Avg Vol (10 day)': info.get('averageVolume10days', None),
                    'Shares Outstanding': info.get('sharesOutstanding', None)

                })
                print("✅ Done.")
            else:
                print("⚠️ Skipped (no data).")
        except Exception as e:
            print(f"❌ symbol {symbol} not found Error: {str(e)}")

    # Be polite to API: small random delay
    time.sleep(random.uniform(0.3, 1.0))
    return results

# ----------------------------
# MAIN EXECUTION
#To calculate intraday price swing volatility (the average range between high and low prices over a period),
# we'll need historical intraday data.
#
#Since free APIs typically provide only daily data, and you specifically asked for intra-day swings,
# I will use Yahoo Finance via yfinance with an intraday frequency (1h, 30m, or 15m)
# for the last 30 days, and calculate:
#Copy
#Intraday Swing = (High - Low) / Close
#Volatility (for period) = Mean of daily average swings OR Standard Deviation of returns — I'll provide both.
#Since true 1-minute intraday data for 30 days exceeds free rate limits, we use hourly/30m data as a
# reasonable proxy.
# ----------------------------
if __name__ == "__main__":
    # File paths
    symbols_file = f'{OUTPUT_DIR}/symbols/test_symbols.txt'
    output_file = f'{OUTPUT_DIR}/symbols/stocks_swing_volatility.csv'

    try:
        symbols: List[str] = load_symbols(symbols_file)
        results: List[Dict[str, Any]] = calculate_symbol_volatility(symbols)

        # Output final table
        if results:
            result_df = pd.DataFrame(results)
            result_df = result_df.sort_values(by='Annualized Swing Vol (%)', ascending=False)

            print("\n" + "=" * 90)
            print("📈 Intraday Swing Volatility Report (Last ~{} Days)".format(LOOKBACK_DAYS))
#            print( f"Interval: {INTRADAY_INTERVAL} | Annualization Factor: ~{(252 * len(result_df.iloc[0]['Samples Used'] / LOOKBACK_DAYS)):.0f} samples/yr")
            print("=" * 90)

            pd.set_option('display.float_format', '{:.4f}'.format)
            print(result_df.to_string(index=False))
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Save to CSV
            result_df.to_csv(output_file, index=False)
        else:
            print("\n⚠️ No valid data retrieved for any symbol.")

    except Exception as e:
        print(f"[❌] {e}")
        exit(1)