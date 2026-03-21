import os
from datetime import datetime
from io import StringIO

import pandas as pd
import requests

from constants import OUTPUT_DIR

# This program uses the Alpha Vantage API to fetch a list of US stock symbols and their details,
# then saves the data to a tab-delimited file. It handles potential errors gracefully and
# ensures that only active stocks are included in the output.
#
# https://www.alphavantage.co/documentation/
#
# Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key
API_KEY = 'KTID5Y01OXARIDHE'
OUTPUT_FILE = os.path.join(OUTPUT_DIR,'symbols', f"us_securities_detailed-{datetime.now().strftime("%Y%m%d")}.csv")


def fetch_us_stock_symbols(api_key):
    """
    Fetches US stock symbols from Alpha Vantage using SYMBOL_SEARCH
    and returns a DataFrame.
    """
    url = f'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={api_key}'

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        csv_data = response.content.decode('utf-8')

        # Check if the API returned an error message
        if not csv_data:
            print(f"API Error: no data returned.")
            return None
        csv_buffer = StringIO(csv_data)
        # Create a DataFrame from the list of matches
        df = pd.read_csv(csv_buffer)
        active_stocks = df[df['status'] == 'Active']
        return active_stocks[["symbol", "name", "exchange", "assetType"]]

    except requests.exceptions.RequestException as e:
        print(f"Network error fetching data: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error processing data: {e}")
        return None


def save_to_tab_delimited(df, filename):
    """
    Saves the DataFrame to a tab-delimited file with specific columns.
    """
    if df is None:
        print("No data to save.")
        return False
    try:
        df.to_csv(filename, sep='\t', index=False)
        print(f"Data successfully saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False


# Main execution
if __name__ == "__main__":
    # TODO: Replace with your API key
    api_key = 'KTID5Y01OXARIDHE'

    stock_df = fetch_us_stock_symbols(api_key)

    if stock_df is not None:
        print("Fetched data sample:")
        print(stock_df.head())
        stock_df.to_csv(OUTPUT_FILE, index=False)
    else:
        print("Failed to retrieve stock symbols.")
