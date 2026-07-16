import csv
import os
import re
from typing import List, Tuple


def find_matching_files(input_folder: str) -> List[Tuple[str, int]]:
    """
    Find all files in input_folder matching pattern AtsPriceBrkout-{Interval}.csv
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


def process_csv_file(file_path: str, interval: int) -> List[Tuple[str, int, int]]:
    """
    Process a single CSV file and extract records with highest SignalCount per symbol.

    Input CSV columns (no header):
    - Column A: Symbol
    - Column B: Interval value (may appear in data)
    - Column D: Signal count value (largest value to extract)

    Returns a list of tuples: (Symbol, Interval, MaxSignalCount)
    """
    symbol_signals = {}  # symbol -> list of signal counts

    try:
        with open(file_path, 'r', newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.reader(csvfile)

            for row_num, row in enumerate(reader, start=1):
                # Skip empty rows
                if not row or all(cell.strip() == '' for cell in row):
                    continue

                if len(row) < 4:
                    continue

                try:
                    symbol = row[0].strip()
                    if not symbol:
                        continue

                    # Parse column D (index 3) as int (signal count)
                    signal_count = int(row[3].strip())

                    if symbol not in symbol_signals:
                        symbol_signals[symbol] = []
                    symbol_signals[symbol].append(signal_count)

                except (ValueError, IndexError):
                    # Skip malformed rows
                    pass

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []

    # Convert to output format - one record per symbol with max signal count
    result = [(symbol, interval, max(signals)) for symbol, signals in symbol_signals.items()]

    return result


def process_folder(input_folder: str) -> List[Tuple[str, int, int]]:
    """
    Process all AtsPriceBrkout-{Interval}.csv files in the folder.
    Returns combined results from all files.
    """
    all_records = {}  # key: (Symbol, Interval) -> MaxSignalCount

    # Find matching files
    matching_files = find_matching_files(input_folder)
    if not matching_files:
        print(f"No files matching pattern AtsPriceBrkout-{{Interval}}.csv found in {input_folder}")
        return []

    print(f"Found {len(matching_files)} file(s)")

    # Process each file
    for file_path, interval in matching_files:
        print(f"  Processing {os.path.basename(file_path)} (interval={interval})...")
        records = process_csv_file(file_path, interval)
        print(f"    Loaded {len(records)} symbols")

        for symbol, interval_val, signal_count in records:
            key = (symbol, interval_val)
            if key not in all_records or signal_count > all_records[key]:
                all_records[key] = signal_count

    # Convert to output format
    result = [(symbol, interval, all_records[(symbol, interval)])
              for symbol, interval in all_records.keys()]

    return result


def write_output_file(result_data: List[Tuple[str, int, int]], output_file_path: str):
    """Write the results to a CSV file with columns: Symbol, Interval, MaxSignalCount"""
    try:
        with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['Symbol', 'Interval', 'MaxSignalCount'])

            # Sort by Symbol, then by Interval
            sorted_data = sorted(result_data, key=lambda r: (r[0], r[1]))

            # Write data rows
            for row in sorted_data:
                writer.writerow(row)

        return True

    except Exception as e:
        print(f"Error writing output file: {e}")
        return False


def write_max_output_file(result_data: List[Tuple[str, int, int]], output_file_path: str):
    """Write only the highest signal count per symbol to a CSV file."""
    try:
        # Group by symbol and keep only the max
        best = {}  # symbol -> (symbol, interval, signal_count)
        for symbol, interval, signal_count in result_data:
            if symbol not in best or signal_count > best[symbol][2]:
                best[symbol] = (symbol, interval, signal_count)

        with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['Symbol', 'Interval', 'MaxSignalCount'])

            # Sort by Symbol
            sorted_data = sorted(best.values(), key=lambda r: r[0])

            # Write data rows
            for row in sorted_data:
                writer.writerow(row)

        return True

    except Exception as e:
        print(f"Error writing max output file: {e}")
        return False


# Example usage:
if __name__ == "__main__":
    """
    Process multiple AtsPriceBrkout-{Interval}.csv files from a folder.
    This script collects all files matching the pattern and extracts the highest
    signal count value (column D) per symbol for each interval.
    
    Example usage:
    python parse_screener_breakout_results.py <input_folder> [output_file.csv] [max_output_file.csv]
    
    Where input_folder contains files like:
      AtsPriceBrkout-5.csv
      AtsPriceBrkout-10.csv
      AtsPriceBrkout-15.csv
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parse_screener_breakout_results.py <input_folder> [output_file.csv] [max_output_file.csv]")
        sys.exit(1)

    input_folder = sys.argv[1]
    
    # Validate folder exists
    if not os.path.isdir(input_folder):
        print(f"Error: Folder '{input_folder}' not found.")
        sys.exit(1)
    
    # Output files (both optional)
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None
    max_output_file = sys.argv[3] if len(sys.argv) >= 4 else None

    print(f"Reading from folder: {input_folder}")

    # Process all files in folder
    result_data = process_folder(input_folder)

    if result_data:
        # Write the main output
        if output_file:
            if write_output_file(result_data, output_file):
                print(f"Output written to: {output_file}  ({len(result_data)} rows)")
            else:
                print("Failed to generate output file.")
        
        # Write the max output
        if max_output_file:
            if write_max_output_file(result_data, max_output_file):
                print(f"Max output written to: {max_output_file}")
            else:
                print("Failed to generate max output file.")
        
        if not output_file and not max_output_file:
            print(f"Processed {len(result_data)} records. No output file specified.")
    else:
        print("No valid data found.")
        sys.exit(1)
