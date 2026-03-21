import os
import traceback
from argparse import ArgumentParser

import finnhub
import pandas as pd

from constants import OUTPUT_DIR

# This uses the Finnhub API to fetch real-time stock quote data for a given symbol. It handles potential errors
# gracefully and ensures that the data is returned in a structured format.
# https://finnhub.io/docs/api
#

# --- Example ---
FINNHUB_API_KEY = "d6qubb1r01qgdhqcfci0d6qubb1r01qgdhqcfcig"  # 🔑 Get free: https://finnhub.io/
symbol = "AAPL"


def load_symbols(filename: str) -> pd.DataFrame:
    """Read stock symbols from a file (one per line)."""
    try:
        return pd.read_csv(filename)
    except Exception:
        traceback.print_exc(e)
        print(f"File '{filename}' not found.")
        exit(1)


def enhance_with_quote_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enhance quote data with additional calculations if needed."""

    # Sample Output (values are illustrative):
    # Quote for AAPL from Finnhub:
    #   Current Price:   $172.50
    #   High (Today):    $173.00
    #   Low (Today):     $171.80
    #   Open:            $172.15
    #   Previous Close:  $171.90
    #   Volume:          45678912

    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    results = []
    for stock_symbol in df.get('symbol', []):
        quote_data = finnhub_client.quote(symbol)
        results.append(
            {
                'symbol': stock_symbol,
                'current_price': quote_data.get('c', 'N/A'),
                'volume': quote_data.get('v', 'N/A')
            }
        )
    merged_pd = pd.merge(df, pd.DataFrame(results), on='symbol', how='inner')
    if merged_pd.size == df.size:
        print(f'Some symbols were lost: merged df: {merged_pd.size} rows, symbols df: {df.size} rows')
    return merged_pd


def filter_by_volume_and_price(df: pd.DataFrame, min_volume: int = 1000000, min_price: int = 0,
                               max_price: int = 500) -> pd.DataFrame:
    """Filter stocks by minimum volume and price range."""
    return df[df['volume'] >= min_volume and df['current_price'] >= min_price and df['current_price'] <= max_price]


# ----------------------------
# MAIN: Main execution loop
# ----------------------------
def main():
    parser = ArgumentParser(description="Calculate intra-day volatility & beta vs SPY")
    parser.add_argument("-i", "--input-file", required=True, help="File containing symbols (one per line)")
    parser.add_argument("-o", "--output-file", required=True, help="File containing output symbols (one per line)")
    parser.add_argument("--min-price", type=float, default=5.0, help="Min price filter")
    parser.add_argument("--max-price", type=float, default=None)
    parser.add_argument("--min-vol",   type=int,   default=100_000, help="Min avg volume (3mo)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    symbols_file = f'{OUTPUT_DIR}/symbols/us_securities_detailed-20260315.csv'
    output_file = f'{OUTPUT_DIR}/symbols/stocks_swing_volatility.csv'
    symbols_df = load_symbols(args.input_file)
    enhanced_df = enhance_with_quote_data(symbols_df)
    selected_df = filter_by_volume_and_price(enhanced_df, min_volume=args.min_vol, min_price=args.min_price, max_price=args.max_price)
    print(f"🔍 Processing {len(selected_df)} symbols...")
    selected_df.to_csv(output_file, index=False)
    print(f"✅ Output saved to {output_file}")


if __name__ == "__main__":
    # File paths
    main()

