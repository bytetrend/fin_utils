import pandas as pd
import requests
import os
import traceback
from typing import List, Dict

# ----------------------------
# CONFIGURATION
# ----------------------------

# 🔑 Get your free Alpha Vantage API Key: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY = 'B09WTUZ0QPFMTDFA'
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"


# ----------------------------
# HELPER FUNCTIONS
# ----------------------------

def load_symbols(filename: str) -> pd.DataFrame:
    """Read stock symbols from a CSV file."""
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File '{filename}' not found.")
        return pd.read_csv(filename)
    except Exception as e:
        print(f"Error loading symbols from '{filename}': {e}")
        traceback.print_exc()
        exit(1)


def get_stock_data(symbol: str) -> dict:
    """
    Fetch the most recent quote using the free, standard daily data endpoint.
    """
    params = {
        'function': 'TIME_SERIES_DAILY',  # <-- Using free, standard endpoint
        'symbol': symbol,
        'apikey': ALPHA_VANTAGE_API_KEY
    }

    try:
        response = requests.get('https://www.alphavantage.co/query', params=params)
        response.raise_for_status()

        data = response.json()

        # CRITICAL: Check for Alpha Vantage's specific error messages.
        if "Error Message" in data:
            print(f"❌ API Error for '{symbol}': {data['Error Message']}")
            return None
        if "Information" in data and "premium" in data["Information"]:
            print(f"⚠️  Symbol '{symbol}' triggered a premium endpoint. Skipping.")
            return None

        # Validate that we have the expected data structure.
        time_series_key = 'Time Series (Daily)'
        if time_series_key not in data:
            print(f"⚠️  Unexpected response for '{symbol}'. Data: {data}")
            return None

        # Get today's data
        time_series = data[time_series_key]
        if not time_series:
            return None

        latest_date = sorted(time_series.keys())[0]  # Most recent date
        latest_data = time_series[latest_date]

        return {
            "symbol": symbol.upper(),
            "date": latest_date,
            "open": float(latest_data["1. open"]),
            "high": float(latest_data["2. high"]),
            "low": float(latest_data["3. low"]),
            "close": float(latest_data["4. close"]),
            "volume": int(latest_data["5. volume"])
        }

    except Exception as err:
        print(f"❌ Request error for '{symbol}': {err}")
        return None


def get_alpha_vantage_quote(symbol: str) -> Dict[str, float]:
    """
    Fetch the most recent quote (current price and volume) for a given symbol
    using Alpha Vantage's TIME_SERIES_INTRADAY endpoint.
    """
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'interval': '1min',  # Get the most granular data possible
        'apikey': ALPHA_VANTAGE_API_KEY
    }

    try:
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        # Alpha Vantage's structure: "Time Series (1min)" -> most recent timestamp
        time_series_key = 'Time Series (1min)'
        if time_series_key not in data:
            print(f"⚠️ No intraday data found for '{symbol}'. API likely rate-limited or symbol is invalid.")
            return {
                'current_price': None,
                'volume': None
            }

        latest_timestamp = list(data[time_series_key].keys())[0]
        latest_data = data[time_series_key][latest_timestamp]

        # Extract required metrics
        current_price = float(latest_data['4. close'])
        volume = int(latest_data['5. volume'])

        return {
            'current_price': current_price,
            'volume': volume
        }

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error fetching data for '{symbol}': {e}")
        return {'current_price': None, 'volume': None}
    except (KeyError, ValueError) as e:
        print(f"⚠️ Error parsing response for '{symbol}': {e}")
        return {'current_price': None, 'volume': None}


def enhance_with_quote_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enhance the symbol DataFrame with quote data from Alpha Vantage."""

    if ALPHA_VANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_API_KEY":
        print("❌ Error: You must set a valid ALPHA_VANTAGE_API_KEY in the script.")
        exit(1)

    results = []
    total_symbols = len(df)
    print(f"🔍 Fetching quote data for {total_symbols} symbols from Alpha Vantage...")

    # Iterate through the DataFrame
    for i, stock_symbol in enumerate(df['symbol'], 1):
        print(f"   [{i}/{total_symbols}] Fetching data for '{stock_symbol}'...")
        quote_data = get_stock_data(stock_symbol)
        try:
            if quote_data:
                results.append(
                    {
                        'symbol': stock_symbol,
                        'current_price': quote_data.get('current_price', None),
                        'volume': quote_data.get('volume',None)
                    }
            )
        except Exception as e:
            print(f"⚠️ Error processing quote data for '{stock_symbol}': {e}")
            results.append(
                {
                    'symbol': stock_symbol,
                    'current_price': None,
                    'volume': None
                }
            )

    # Convert list of dicts to DataFrame
    quotes_df = pd.DataFrame(results)

    # Merge with original df
    merged_pd = pd.merge(df, quotes_df, on='symbol', how='left')

    print(f"✅ Data fetch complete. {len(merged_pd)} symbols retained.")

    if len(merged_pd) < len(df):
        print(f"⚠️ Some symbols were lost: merged df: {len(merged_pd)} rows, original symbols df: {len(df)} rows")

    return merged_pd


def filter_by_volume_and_price(df: pd.DataFrame, min_volume: int = 1000000, min_price: float = 5.0,
                               max_price: float = 500.0) -> pd.DataFrame:
    """Filter stocks by minimum volume and price range."""

    # Drop rows where critical data is missing
    df_filtered = df.dropna(subset=['current_price', 'volume'])

    return df_filtered[
        (df_filtered['volume'] >= min_volume) &
        (df_filtered['current_price'] >= min_price) &
        (df_filtered['current_price'] <= max_price)
        ]


# ----------------------------
# MAIN: Main execution loop
# ----------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Calculate intra-day volatility & beta vs SPY")
    parser.add_argument("-i", "--input-file", required=True, help="File containing symbols (CSV format)")
    parser.add_argument("-o", "--output-file", required=True, help="Output file for filtered stocks (CSV format)")
    parser.add_argument("--min-price", type=float, default=5.0, help="Min price filter")
    parser.add_argument("--max-price", type=float, default=None)
    parser.add_argument("--min-vol", type=int, default=100_000, help="Min volume (from quote)")

    args = parser.parse_args()

    # Assuming OUTPUT_DIR is defined somewhere or use current directory
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    symbols_df = load_symbols(args.input_file)

    print("🚀 Starting enhancement with Alpha Vantage quote data...")
    enhanced_df = enhance_with_quote_data(symbols_df)

    print(f"📊 Filtering by price (${args.min_price}-${args.max_price}) and volume (>{args.min_vol})...")
    selected_df = filter_by_volume_and_price(enhanced_df, min_volume=args.min_vol, min_price=args.min_price,
                                             max_price=args.max_price)

    print(f"✅ Processing complete. {len(selected_df)} symbols match the criteria.")
    selected_df.to_csv(args.output_file, index=False)
    print(f"📄 Output saved to {args.output_file}")


if __name__ == "__main__":
    main()
