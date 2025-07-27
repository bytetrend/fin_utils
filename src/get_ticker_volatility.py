import pandas as pd
import sys
import os
from tqdm import tqdm
import time
import yfinance as yf

from constants import OUTPUT_DIR


def read_symbols(file_path):
    """Read stock symbols from a text file."""
    try:
        with open(file_path, 'r') as file:
            symbols = [line.strip().upper() for line in file if line.strip()]
        return symbols
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []

def extract_volatility_data(ticker):
    """Extract volatility-related data for a single ticker using yfinance."""
    try:
        # Create yfinance ticker object
        stock = yf.Ticker(ticker)
        
        # Get stock info
        info = stock.info
        
        # Map our target attributes to yfinance keys
        result = {
            'Ticker': ticker,
            'Beta (5Y Monthly)': info.get('beta', None),
            '52 Week Change': info.get('52WeekChange', None),
            'Avg Vol (3 month)': info.get('averageVolume', None),
            'Avg Vol (10 day)': info.get('averageVolume10days', None),
            'Shares Outstanding': info.get('sharesOutstanding', None)
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        # Return a row with None values if there's an error
        result = {
            'Ticker': ticker,
            'Beta (5Y Monthly)': None,
            '52 Week Change': None,
            'Avg Vol (3 month)': None,
            'Avg Vol (10 day)': None,
            'Shares Outstanding': None
        }
        return result

def main():
    """Main function to process all tickers and create the volatility CSV."""
    
    # File paths
    symbols_file = f'{OUTPUT_DIR}/symbols_list/stocks_symbols.txt'
    output_file = f'{OUTPUT_DIR}/symbol_info/stocks_ticker_volatility.csv'
    
    # Read stock symbols
    print("Reading stock symbols...")
    symbols = read_symbols(symbols_file)
    
    if not symbols:
        print("No symbols found or file not accessible.")
        return
    
    print(f"Found {len(symbols)} symbols to process.")
    
    # Process each ticker
    volatility_data = []
    failed_tickers = []
    
    print("Processing tickers...")
    for i, ticker in enumerate(tqdm(symbols, desc="Processing")):
        try:
            data = extract_volatility_data(ticker)
            volatility_data.append(data)
            
            # Add a small delay to avoid overwhelming the API
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Failed to process {ticker}: {str(e)}")
            failed_tickers.append(ticker)
            continue
        
        # Save intermediate results every 100 tickers
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} tickers...")
            # Save partial results
            if volatility_data:
                df_partial = pd.DataFrame(volatility_data)
                cols = ['Ticker', 'Beta (5Y Monthly)', '52 Week Change', 
                        'Avg Vol (3 month)', 'Avg Vol (10 day)', 'Shares Outstanding']
                df_partial = df_partial[cols]
                df_partial.to_csv(output_file.replace('.csv', '_partial.csv'), index=False)
    
    # Create DataFrame
    if volatility_data:
        df = pd.DataFrame(volatility_data)
        
        # Reorder columns to have Ticker first
        cols = ['Ticker', 'Beta (5Y Monthly)', '52 Week Change', 
                'Avg Vol (3 month)', 'Avg Vol (10 day)', 'Shares Outstanding']
        df = df[cols]
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
        print(f"Successfully processed {len(volatility_data)} tickers")
        
        if failed_tickers:
            print(f"Failed to process {len(failed_tickers)} tickers: {failed_tickers[:10]}{'...' if len(failed_tickers) > 10 else ''}")
        
        # Display first few rows
        print("\nFirst 5 rows of the data:")
        print(df.head())
        
        # Display some statistics
        print(f"\nData completeness:")
        for col in cols[1:]:  # Skip 'Ticker' column
            non_null_count = df[col].notna().sum()
            percentage = (non_null_count / len(df)) * 100
            print(f"  {col}: {non_null_count}/{len(df)} ({percentage:.1f}%)")
        
    else:
        print("No data was successfully extracted.")

if __name__ == "__main__":
    main() 