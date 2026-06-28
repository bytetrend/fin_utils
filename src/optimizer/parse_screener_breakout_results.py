import csv
from typing import List, Tuple


def process_csv(input_file_path: str) -> List[Tuple[str, str, int]]:
    """
    Process CSV file and extract the best record per symbol based on highest SignalCount.

    Assumptions:
    - Input CSV has columns in this exact order (no header):
      Symbol, Interval, BarNumber, BarDate, BarTime, SignalCount, ComputerDateTime
    - Date/time formats: BarDate as 'YYYY-MM-DD', BarTime as 'HH:MM:SS'
      ComputerDateTime as 'YYYY-MM-DD HH:MM:SS' or ISO format

    Returns a list of tuples: (Symbol, Interval, SignalCount, ComputerDateTime)
    """

    # Dictionary to store best record per symbol
    # Key: Symbol, Value: (Interval, SignalCount, ComputerDateTime, BarDate, BarTime, BarNumber)
    best_records = {}

    try:
        with open(input_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)

            for row in reader:
                # Skip empty rows
                if not row or all(cell.strip() == '' for cell in row):
                    continue

                try:
                    # Extract fields based on column order
                    symbol, interval, bar_number, signal_count, computer_datetime = row[:5]

                    # Convert SignalCount to integer for comparison
                    signal_count_int = int(signal_count.strip())

                    record_key = f"{symbol.strip()}-{interval}"

                    if record_key not in best_records:
                        # First record for this symbol
                        best_records[record_key] = (
                            symbol.strip(),
                            int(interval.strip()),
                            signal_count_int
                        )
                    else:
                        current_best = best_records[record_key]

                        # Compare: highest SignalCount wins
                        if signal_count_int > current_best[2]:
                            best_records[record_key] = (
                                symbol.strip(),
                                int(interval.strip()),
                                signal_count_int
                            )

                except ValueError as e:
                    # Skip malformed rows with error logging (optional: add print statement)
                    pass

    except FileNotFoundError:
        print(f"Error: File '{input_file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error processing file: {e}")
        return []

    # Convert to output format
    result = [(record[0], record[1],record[2]) for key, record in best_records.items()]

    return result


def write_output_file(result_data: List[Tuple[str, str, int]], output_file_path: str):
    """Write the results to a CSV file with columns: Symbol, Interval, signalcount, computerdatetime"""
    try:
        with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['Symbol', 'Interval', 'signalcount'])

            # Write data rows
            for row in result_data:
                writer.writerow(row)

        return True

    except Exception as e:
        print(f"Error writing output file: {e}")
        return False


# Example usage:
if __name__ == "__main__":
    """
    This script process a CSV file output from the MC indicator AtsPriceBreakout_screener.
    Such script generate a count of entry signals for the strategy AtsPriceBreakout.
    This script will collect them and pick the one with the largest count per each symbol and interval.
    The script is loaded in charts with multiple data intervals 10, 20, 30 ticks, etc.
    For multiple symbols. 
    Example usage:
    pass input and output file example:
    C:/Invest/logs/screener/AtsPriceBrkout.csv C:/Invest/logs/screener/pricebrkout_analysis.csv
    """
    import sys

    if len(sys.argv) < 3:
        print("Usage: python parse_screener_breakout_results.py <input_file.csv> <output_file.csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Process the file
    result_data = process_csv(input_file)

    # Write the output
    if write_output_file(result_data, output_file):
        print(f"Successfully processed {len(result_data)} symbols. Output written to: {output_file}")
    else:
        print("Failed to generate output file.")
