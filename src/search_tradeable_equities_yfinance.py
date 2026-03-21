import json
import time
import warnings

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from constants import OUTPUT_DIR

warnings.filterwarnings('ignore')


def calculate_weekly_atr(df, period_weeks=12):
    """Calculate Weekly Average True Range (ATR) for 3 months (12 weeks)."""
    if len(df) < period_weeks * 5 + 1:  # Need at least 12 weeks of data (60+ trading days)
        return None

    # Resample to weekly data (using Friday as the end of week)
    weekly_df = df.resample('W-FRI').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    if len(weekly_df) < period_weeks + 1:
        return None

    high = weekly_df['High']
    low = weekly_df['Low']
    close = weekly_df['Close']

    # Calculate True Range components
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    # True Range is the maximum of the three
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR is the rolling average of True Range
    atr = true_range.rolling(window=period_weeks).mean()

    # Return the most recent ATR value
    return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else None


def classify_ticker_type(quote_type):
    """Classify ticker type based on quoteType."""
    if quote_type == 'ETF':
        return 'ETF'
    elif quote_type == 'EQUITY':
        return 'STOCK'
    else:
        return 'OTHER'


def get_stock_metrics(ticker, description=""):
    """Get stock metrics including volume, beta, and weekly ATR."""
    try:
        stock = yf.Ticker(ticker)

        # Get basic info
        info = stock.info

        # Get required metrics
        avg_vol_3m = info.get('averageVolume', 0)  # 3 month average volume
        beta = info.get('beta', 0)
        bid = info.get('bid', 0)
        shares_outstanding = info.get('sharesOutstanding', 0)
        quote_type = info.get('quoteType', 'OTHER')

        # Get historical data for Weekly ATR calculation (3 months)
        hist = stock.history(period='3mo')
        if hist.empty or len(hist) < 60:  # Need at least 60 trading days for 3 months
            return None

        # Calculate Weekly ATR for 3 months
        weekly_atr = calculate_weekly_atr(hist, period_weeks=12)

        if weekly_atr is None or avg_vol_3m is None or beta is None:
            return None

        return {
            'Ticker': ticker,
            'Description': description or info.get('longName', 'N/A'),
            'Ticker Type': classify_ticker_type(quote_type),
            'Weekly ATR (3 Months)': round(weekly_atr, 3),
            'Avg Vol (3 month)': avg_vol_3m,
            'Beta (5Y Monthly)': beta,
            'Shares Outstanding': shares_outstanding,
            'Price': bid
        }

    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        return None


def read_tickers_from_json(file_path):
    """Read tickers from JSON file."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

        tickers = []
        for key, value in data.items():
            tickers.append({
                'ticker': value['ticker'],
                'description': value['title']
            })
        return tickers
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return []


def filter_tradeable_equities(tickers_data, min_volume=3000000, min_beta=1.2, min_atr=1):
    """Filter tickers based on trading criteria."""

    tradeable_equities = []

    for i, ticker_info in enumerate(tqdm(tickers_data, desc="Processing")):
        try:
            ticker = ticker_info['ticker']
            description = ticker_info['description']

            metrics = get_stock_metrics(ticker, description)

            if metrics is None:
                continue

            # Apply filtering criteria
            if (metrics['Avg Vol (3 month)'] > min_volume and
                    metrics['Beta (5Y Monthly)'] > min_beta and
                    metrics['Weekly ATR (3 Months)'] > min_atr):
                tradeable_equities.append(metrics)

            # Add delay to avoid overwhelming the API
            time.sleep(0.1)

        except Exception as e:
            continue

    return tradeable_equities


def save_results(tradeable_equities, output_file):
    """Save results to CSV file."""
    if tradeable_equities:
        df = pd.DataFrame(tradeable_equities)

        # Reorder columns as requested
        column_order = [
            'Ticker',
            'Description',
            'Ticker Type',
            'Weekly ATR (3 Months)',
            'Avg Vol (3 month)',
            'Beta (5Y Monthly)',
            'Shares Outstanding',
            'Price'
        ]

        df = df[column_order]

        # Sort by Weekly ATR descending (most volatile first)
        df = df.sort_values('Weekly ATR (3 Months)', ascending=False)

        # Save to CSV
        df.to_csv(output_file, index=False)

        return df
    else:
        return pd.DataFrame()


def main():
    """Main function to search for tradeable equities."""

    # Configuration
    # from https://www.sec.gov/files/company_tickers.json
    stocks_file = f'{OUTPUT_DIR}/symbols_list/company_tickers.json'  # Use the stocks file from classification
    output_file = f'{OUTPUT_DIR}/symbols/most_tradeable_equities.csv'

    # Filter criteria
    MIN_VOLUME = 3_000_000
    MIN_BETA = 1.2
    MIN_ATR = 1.0

    # Read tickers from JSON file
    tickers_data = read_tickers_from_json(stocks_file)

    if not tickers_data:
        return

    # Filter tradeable equities
    tradeable_equities = filter_tradeable_equities(
        tickers_data, MIN_VOLUME, MIN_BETA, MIN_ATR
    )

    # Save results
    df = save_results(tradeable_equities, output_file)


if __name__ == "__main__":
    main()
