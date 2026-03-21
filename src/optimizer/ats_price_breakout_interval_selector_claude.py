#!/usr/bin/env python3
"""
Process signal CSV file and output the row with the highest SignalCount per Symbol.

Input CSV columns (no header):
  Symbol, Interval, BarNumber, BarDate, BarTime, SignalCount, ComputerDateTime

Output CSV columns:
  Symbol, Interval, SignalCount, ComputerDateTime
"""

import csv
import sys
import os
from collections import defaultdict


def parse_args():
    if len(sys.argv) < 2:
        print("Usage: python process_signals.py <input_csv> [output_csv]")
        print("  input_csv  : Path to the input CSV file (no header)")
        print("  output_csv : (Optional) Path to output file. Defaults to <input_name>_output.csv")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_output{ext or '.csv'}"

    return input_file, output_file


def read_csv(input_file):
    """
    Read the CSV and return a list of dicts with named columns.
    Handles both comma and semicolon delimiters.
    """
    columns = ["Symbol", "Interval", "BarNumber", "BarDate", "BarTime", "SignalCount", "ComputerDateTime"]
    rows = []

    with open(input_file, newline='', encoding='utf-8-sig') as f:
        # Sniff delimiter
        sample = f.read(2048)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',;\t|')
        except csv.Error:
            dialect = csv.excel  # fallback to comma

        reader = csv.reader(f, dialect)
        for line_num, row in enumerate(reader, start=1):
            if not row or all(cell.strip() == '' for cell in row):
                continue  # skip blank lines

            if len(row) < len(columns):
                print(f"Warning: Line {line_num} has only {len(row)} columns (expected {len(columns)}), skipping.")
                continue

            record = {col: row[i].strip() for i, col in enumerate(columns)}

            # Parse SignalCount as int (skip row on failure)
            try:
                record["SignalCount"] = int(record["SignalCount"])
            except ValueError:
                print(f"Warning: Line {line_num} has non-integer SignalCount '{record['SignalCount']}', skipping.")
                continue

            rows.append(record)

    return rows


def find_latest_row_per_symbol_interval(rows):
    """
    For each (Symbol, Interval) combination, find the row with the latest
    BarDate/BarTime (or highest BarNumber as fallback). This represents the
    most recent snapshot of signal data.
    """
    latest = {}  # key: (Symbol, Interval) -> best row so far

    for row in rows:
        key = (row["Symbol"], row["Interval"])
        if key not in latest:
            latest[key] = row
        else:
            current = latest[key]
            # Compare by BarDate + BarTime first, then BarNumber
            current_dt = (current["BarDate"], current["BarTime"])
            row_dt = (row["BarDate"], row["BarTime"])

            if row_dt > current_dt:
                latest[key] = row
            elif row_dt == current_dt:
                # Tie-break on BarNumber
                try:
                    if int(row["BarNumber"]) > int(current["BarNumber"]):
                        latest[key] = row
                except ValueError:
                    pass

    return list(latest.values())


def find_highest_signalcount_per_symbol(latest_rows):
    """
    From the latest rows (one per Symbol+Interval), find the single row per
    Symbol with the highest SignalCount.
    """
    best = {}  # key: Symbol -> best row

    for row in latest_rows:
        symbol = row["Symbol"]
        if symbol not in best or row["SignalCount"] > best[symbol]["SignalCount"]:
            best[symbol] = row

    return list(best.values())


def write_output(output_file, results):
    """Write the output CSV with header."""
    out_columns = ["Symbol", "Interval", "SignalCount", "ComputerDateTime"]

    # Sort by Symbol for readability
    results.sort(key=lambda r: r["Symbol"])

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=out_columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    return len(results)


def main():
    input_file, output_file = parse_args()

    print(f"Reading: {input_file}")
    rows = read_csv(input_file)
    print(f"  Loaded {len(rows)} valid rows.")

    if not rows:
        print("No valid data found. Exiting.")
        sys.exit(1)

    # Step 1: Per (Symbol, Interval) — keep only the latest bar
    latest_rows = find_latest_row_per_symbol_interval(rows)
    print(f"  Unique Symbol+Interval combinations: {len(latest_rows)}")

    # Step 2: Per Symbol — keep the row with the highest SignalCount
    results = find_highest_signalcount_per_symbol(latest_rows)
    print(f"  Unique Symbols: {len(results)}")

    count = write_output(output_file, results)
    print(f"Output written to: {output_file}  ({count} rows)")

    # Preview
    print("\nPreview:")
    print(f"  {'Symbol':<12} {'Interval':<10} {'SignalCount':<12} {'ComputerDateTime'}")
    print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*20}")
    for r in results:
        print(f"  {r['Symbol']:<12} {r['Interval']:<10} {str(r['SignalCount']):<12} {r['ComputerDateTime']}")


if __name__ == "__main__":
    main()