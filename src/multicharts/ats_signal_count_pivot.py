#!/usr/bin/env python3
"""
Process price breakout CSV files and pivot them by symbol and interval.

Input files: AtsPriceBrkout-{Interval}.csv
  Columns: Symbol, Interval, BarCount, SignalCount, Time

Output: One row per symbol with columns for each interval's max signal count.
  Columns: Symbol, Interval1, Interval2, Interval3, Interval4, MaxSignalCount
"""

import csv
import sys
import os
import re
from collections import defaultdict


def find_matching_files(input_folder: str) -> list:
    """
    Find all files matching pattern AtsPriceBrkout-{Interval}.csv
    Returns list of (filepath, interval) tuples sorted by interval.
    """
    pattern = r'AtsPriceBrkout-(\d+)\.csv$'
    matching = []

    for filename in os.listdir(input_folder):
        match = re.match(pattern, filename)
        if match:
            interval = int(match.group(1))
            filepath = os.path.join(input_folder, filename)
            matching.append((filepath, interval))

    return sorted(matching, key=lambda x: x[1])


def process_all_files(input_folder: str) -> dict:
    """
    Process all matching files and collect max signal count per (symbol, interval).
    
    Returns: dict with structure:
      {symbol: {interval: max_signal_count, ...}, ...}
    """
    data = defaultdict(dict)  # symbol -> {interval -> max_signal_count}
    
    matching_files = find_matching_files(input_folder)
    if not matching_files:
        print(f"No files matching pattern AtsPriceBrkout-{{Interval}}.csv found.")
        return data
    
    print(f"Found {len(matching_files)} file(s)")
    
    # Process each file
    for file_path, interval in matching_files:
        print(f"  Processing {os.path.basename(file_path)} (interval={interval})...")
        
        try:
            with open(file_path, 'r', newline='', encoding='utf-8-sig') as csvfile:
                reader = csv.reader(csvfile)
                symbol_count = 0
                
                for row in reader:
                    # Skip empty rows
                    if not row or all(cell.strip() == '' for cell in row):
                        continue
                    
                    if len(row) < 4:
                        continue
                    
                    try:
                        symbol = row[0].strip()
                        # interval_from_file = int(row[1].strip())  # In case we need to verify
                        signal_count = int(row[3].strip())
                        
                        if not symbol:
                            continue
                        
                        # Keep track of max signal count for this symbol+interval
                        if interval not in data[symbol]:
                            data[symbol][interval] = signal_count
                        else:
                            data[symbol][interval] = max(data[symbol][interval], signal_count)
                        
                        symbol_count += 1
                    
                    except (ValueError, IndexError):
                        pass
                
                print(f"    Loaded {symbol_count} entries")
        
        except Exception as e:
            print(f"    Error reading {os.path.basename(file_path)}: {e}")
    
    return data


def get_all_intervals(data: dict) -> sorted:
    """Extract all unique intervals from data and return sorted list."""
    intervals = set()
    for symbol_data in data.values():
        intervals.update(symbol_data.keys())
    return sorted(intervals)


def write_output_file(data: dict, intervals: list, output_file_path: str):
    """
    Write pivoted output with one row per symbol.
    Columns: Symbol, Interval1, Interval2, Interval3, Interval4, MaxSignalCount
    """
    try:
        with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            # Create column headers
            headers = ['Symbol'] + [f'Interval{iv}' for iv in intervals] + ['MaxSignalCount']
            
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
            # Write data rows sorted by symbol
            for symbol in sorted(data.keys()):
                symbol_data = data[symbol]
                
                # Get signal counts for each interval, using 0 if not present
                signal_counts = [symbol_data.get(iv, 0) for iv in intervals]
                
                # Max signal count across all intervals for this symbol
                max_signal = max(signal_counts) if signal_counts else 0
                
                # Write row: Symbol, Interval1, Interval2, ..., MaxSignalCount
                row = [symbol] + signal_counts + [max_signal]
                writer.writerow(row)
        
        return len(data)
    
    except Exception as e:
        print(f"Error writing output file: {e}")
        return 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python ats_signal_count_pivot.py <input_folder> [output_file.csv]")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    
    if not os.path.isdir(input_folder):
        print(f"Error: Folder '{input_folder}' not found.")
        sys.exit(1)
    
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        output_file = os.path.join(input_folder, "signal_count_pivot.csv")
    
    print(f"Reading from folder: {input_folder}")
    
    # Process all files
    data = process_all_files(input_folder)
    
    if not data:
        print("No valid data found.")
        sys.exit(1)
    
    # Get list of all intervals
    intervals = get_all_intervals(data)
    print(f"\nFound intervals: {intervals}")
    print(f"Total unique symbols: {len(data)}")
    
    # Write output
    count = write_output_file(data, intervals, output_file)
    print(f"\nOutput written to: {output_file}  ({count} rows)")
    
    # Preview
    print("\nPreview (first 10 rows):")
    with open(output_file, 'r') as f:
        for i, line in enumerate(f):
            if i < 11:  # Header + 10 data rows
                print(f"  {line.rstrip()}")
            else:
                break
    
    if len(data) > 10:
        print(f"  ... and {len(data) - 10} more rows")


if __name__ == "__main__":
    main()
