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

def classify_ticker(ticker):
    """Classify a ticker as ETF or Stock using yfinance."""
    try:
        # Create yfinance ticker object
        stock = yf.Ticker(ticker)
        
        # Get stock info
        info = stock.info
        
        # Get the quote type to determine if it's an ETF or stock
        quote_type = info.get('quoteType', 'UNKNOWN')
        
        if quote_type == 'ETF':
            return 'ETF'
        elif quote_type == 'EQUITY':
            return 'STOCK'
        else:
            # For unknown types, try to get more info
            fund_family = info.get('fundFamily', '')
            legal_type = info.get('legalType', '')
            
            if 'ETF' in legal_type.upper() or 'EXCHANGE TRADED' in legal_type.upper():
                return 'ETF'
            elif fund_family and fund_family != 'Unknown':
                return 'ETF'  # If it has a fund family, likely an ETF
            else:
                return 'UNKNOWN'
        
    except Exception as e:
        print(f"Error classifying {ticker}: {str(e)}")
        return 'ERROR'

def write_symbols_to_file(symbols, file_path):
    """Write a list of symbols to a file, one per line."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            for symbol in symbols:
                file.write(f"{symbol}\n")
        print(f"Wrote {len(symbols)} symbols to {file_path}")
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")

def main():
    """Main function to classify all tickers and separate them into stocks and ETFs."""
    
    # File paths
    symbols_file = f'{OUTPUT_DIR}/symbols_list/symbols.txt'
    stocks_output_file = f'{OUTPUT_DIR}/symbols_list/stocks_symbols.txt'
    etfs_output_file = f'{OUTPUT_DIR}/symbols_list/etf_symbols.txt'
    
    # Read stock symbols
    print("Reading stock symbols...")
    symbols = read_symbols(symbols_file)
    
    if not symbols:
        print("No symbols found or file not accessible.")
        return
    
    print(f"Found {len(symbols)} symbols to classify.")
    
    # Lists to store classified symbols
    stocks = []
    etfs = []
    unknown = []
    errors = []
    
    print("Classifying tickers...")
    for i, ticker in enumerate(tqdm(symbols, desc="Classifying")):
        try:
            classification = classify_ticker(ticker)
            
            if classification == 'STOCK':
                stocks.append(ticker)
            elif classification == 'ETF':
                etfs.append(ticker)
            elif classification == 'UNKNOWN':
                unknown.append(ticker)
            else:  # ERROR
                errors.append(ticker)
            
            # Add a small delay to avoid overwhelming the API
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Failed to process {ticker}: {str(e)}")
            errors.append(ticker)
            continue
        
        # Progress update every 100 tickers
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} tickers... "
                  f"Stocks: {len(stocks)}, ETFs: {len(etfs)}, "
                  f"Unknown: {len(unknown)}, Errors: {len(errors)}")
    
    # Write results to files
    print("\nWriting results to files...")
    
    # Write stocks to file
    if stocks:
        write_symbols_to_file(stocks, stocks_output_file)
    else:
        print("No stocks found to write.")
    
    # Write ETFs to file
    if etfs:
        write_symbols_to_file(etfs, etfs_output_file)
    else:
        print("No ETFs found to write.")
    
    # Print summary
    print(f"\nClassification Summary:")
    print(f"  Total symbols processed: {len(symbols)}")
    print(f"  Stocks: {len(stocks)}")
    print(f"  ETFs: {len(etfs)}")
    print(f"  Unknown: {len(unknown)}")
    print(f"  Errors: {len(errors)}")
    
    if unknown:
        print(f"\nUnknown tickers (first 10): {unknown[:10]}")
    
    if errors:
        print(f"\nError tickers (first 10): {errors[:10]}")
    
    # Create a summary CSV file as well
    summary_data = []
    for ticker in stocks:
        summary_data.append({'Ticker': ticker, 'Type': 'STOCK'})
    for ticker in etfs:
        summary_data.append({'Ticker': ticker, 'Type': 'ETF'})
    for ticker in unknown:
        summary_data.append({'Ticker': ticker, 'Type': 'UNKNOWN'})
    for ticker in errors:
        summary_data.append({'Ticker': ticker, 'Type': 'ERROR'})
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        summary_file = '../data/ticker_classification_summary.csv'
        df_summary.to_csv(summary_file, index=False)
        print(f"\nSummary saved to {summary_file}")
        
        # Show some statistics
        print(f"\nType distribution:")
        type_counts = df_summary['Type'].value_counts()
        for type_name, count in type_counts.items():
            percentage = (count / len(df_summary)) * 100
            print(f"  {type_name}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    main() 