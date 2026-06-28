#!/usr/bin/env python3
"""
organize_stock_data.py

Organizes MultiCharts export stock data files into a structured directory hierarchy.

Usage:
    python organize_stock_data.py <source_directory> <destination_directory>

File naming convention (source):
    Symbol-FeedName-Exchange-Category-Resolution-DataType.txt
    Example: AMZN-TradeStation-NASDAQ-Stocks-Minute-Trade.txt

Destination structure:
    <dest>/Exchange/Category/Resolution/DataType/Symbol.txt
    Example: <dest>/NASDAQ/Stocks/Minute/Trade/AMZN.txt
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


EXPECTED_PARTS = 6  # Symbol, Feed, Exchange, Category, Resolution, DataType


def parse_filename(filename: str) -> dict | None:
    """
    Parse a stock data filename into its components.

    Expected format: Symbol-FeedName-Exchange-Category-Resolution-DataType.txt
    Returns a dict with keys: symbol, feed, exchange, category, resolution, data_type
    Returns None if the filename doesn't match the expected pattern.
    """
    stem = Path(filename).stem  # Strip .txt extension
    parts = stem.split("-")

    if len(parts) < EXPECTED_PARTS:
        return None

    # Join any extra dashes back (feed name could theoretically have dashes,
    # but based on the spec the middle fields are fixed — so split from both ends)
    symbol      = parts[0]
    data_type   = parts[-1]
    resolution  = parts[-2]
    category    = parts[-3]
    exchange    = parts[-4]
    # Everything between symbol and exchange is the feed name
    feed        = "-".join(parts[1 : len(parts) - 4])

    if not all([symbol, feed, exchange, category, resolution, data_type]):
        return None

    return {
        "symbol":    symbol,
        "feed":      feed,
        "exchange":  exchange,
        "category":  category,
        "resolution": resolution,
        "data_type": data_type,
    }


def build_dest_path(dest_root: Path, info: dict) -> Path:
    """
    Build the full destination file path from parsed filename components.

    Structure: dest_root/Exchange/Category/Resolution/DataType/Symbol.txt
    """
    return (
        dest_root
        / info["exchange"]
        / info["category"]
        / info["resolution"]
        / info["data_type"]
        / f"{info['symbol']}.txt"
    )


def organize_files(source_dir: str, dest_dir: str, dry_run: bool = False) -> None:
    """
    Walk source_dir, parse each .txt file, and move it to the correct
    subdirectory under dest_dir.
    """
    source_path = Path(source_dir).resolve()
    dest_path   = Path(dest_dir).resolve()

    if not source_path.exists():
        print(f"ERROR: Source directory does not exist: {source_path}")
        sys.exit(1)

    if not source_path.is_dir():
        print(f"ERROR: Source path is not a directory: {source_path}")
        sys.exit(1)

    txt_files = sorted(source_path.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in: {source_path}")
        return

    moved    = 0
    skipped  = 0
    errors   = 0

    print(f"Source : {source_path}")
    print(f"Dest   : {dest_path}")
    print(f"Mode   : {'DRY RUN (no files moved)' if dry_run else 'LIVE'}")
    print(f"Files  : {len(txt_files)} .txt file(s) found\n")
    print("-" * 70)

    for src_file in txt_files:
        info = parse_filename(src_file.name)

        if info is None:
            print(f"  SKIP  {src_file.name}  (filename does not match expected pattern)")
            skipped += 1
            continue

        dst_file = build_dest_path(dest_path, info)

        # Check for collision
        if dst_file.exists():
            print(f"  WARN  {src_file.name}")
            print(f"        Destination already exists: {dst_file}")
            print(f"        Skipping to avoid overwrite.")
            skipped += 1
            continue

        print(f"  MOVE  {src_file.name}")
        print(f"     -> {dst_file.relative_to(dest_path)}")

        if not dry_run:
            try:
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_file), str(dst_file))
                moved += 1
            except Exception as exc:
                print(f"        ERROR: {exc}")
                errors += 1
        else:
            moved += 1  # Count as "would move" in dry-run mode

    print("-" * 70)
    action = "Would move" if dry_run else "Moved"
    print(f"\nDone.  {action}: {moved}  |  Skipped: {skipped}  |  Errors: {errors}")


def main():
    parser = argparse.ArgumentParser(
        description="Organize MultiCharts stock data files into Exchange/Category/Resolution/DataType subdirectories."
    )
    parser.add_argument(
        "source_dir",
        help="Directory containing the source .txt files."
    )
    parser.add_argument(
        "dest_dir",
        help="Root destination directory where the organized structure will be created."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would happen without moving any files."
    )

    args = parser.parse_args()
    organize_files(args.source_dir, args.dest_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()