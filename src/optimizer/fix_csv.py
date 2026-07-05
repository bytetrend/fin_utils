import csv
import os
import sys
import shutil
from pathlib import Path

SIGNAL_SENT_COL = 4   # Column 5, 0-indexed (header: SignalSent)
EXTRA_CELL_COL  = 34  # Column 35, 0-indexed
TEMP_FILENAME   = "junk.csv"

def fix_csv(input_path: Path) -> bool:
    """
    Process a single CSV file. Writes the result to junk.csv in the same
    directory, then overwrites the original if any rows were fixed.
    Returns True if the file was modified.
    """
    temp_path = input_path.parent / TEMP_FILENAME
    modified  = False

    with open(input_path, newline='', encoding='utf-8') as infile, \
         open(temp_path,  'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        try:
            header = next(reader)
        except StopIteration:
            print(f"  [SKIP] {input_path.name} is empty.")
            return False

        writer.writerow(header)
        expected_cols = len(header)

        for line_num, row in enumerate(reader, start=2):
            if len(row) == expected_cols + 1 and row[SIGNAL_SENT_COL].strip() == '1':
                row = row[:EXTRA_CELL_COL] + row[EXTRA_CELL_COL + 1:]
                modified = True
            elif len(row) != expected_cols:
                print(f"  [WARN] {input_path.name} line {line_num}: "
                      f"{len(row)} columns (expected {expected_cols}) — left unchanged.")
            writer.writerow(row)

    if modified:
        shutil.move(str(temp_path), str(input_path))
        print(f"  [FIXED]    {input_path.name} — extra cells removed, original overwritten.")
    else:
        temp_path.unlink()
        print(f"  [NO CHANGE] {input_path.name} — no extra cells found, skipped.")

    return modified


def main():
    if len(sys.argv) != 2:
        print("Usage: python fix_csv.py <directory>")
        sys.exit(1)

    directory = Path(sys.argv[1])
    if not directory.is_dir():
        print(f"Error: '{directory}' is not a valid directory.")
        sys.exit(1)

    csv_files = [f for f in sorted(directory.glob("*.csv"))
                 if f.name != TEMP_FILENAME]

    if not csv_files:
        print(f"No CSV files found in '{directory}'.")
        sys.exit(0)

    print(f"Found {len(csv_files)} CSV file(s) in '{directory}':\n")
    fixed_count = sum(fix_csv(f) for f in csv_files)
    print(f"\nDone. {fixed_count}/{len(csv_files)} file(s) were modified.")


if __name__ == '__main__':
    main()
