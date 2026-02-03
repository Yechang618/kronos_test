#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix and reformat timestamp column in basis CSV files.
- Rename 'timestampes' → 'timestamps'
- Parse existing string timestamps (e.g., "4/1/2025 12:00:00 AM")
- Re-format to clean "M/D/YYYY H:MM:SS AM/PM" style
"""

import pandas as pd
from pathlib import Path
import re

# ----------------------------
# Configuration
# ----------------------------
BASE_DIR = Path("./datasets/processed/basis_1min_task7")
OUTPUT_DIR = BASE_DIR  # overwrite in place; change if you want backup

# Ensure output dir exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Helper: Parse flexible timestamp string
# ----------------------------
def parse_timestamp_str(ts_str):
    """
    Parse timestamp string like:
      "4/1/2025 12:00:00 AM"
      "12/29/2025 1:30:45 PM"
    Returns: datetime object (naive, assumed UTC)
    """
    try:
        return pd.to_datetime(ts_str, format='%m/%d/%Y %I:%M:%S %p')
    except ValueError:
        # Fallback: try with single-digit month/day (though %m/%d handles it)
        return pd.to_datetime(ts_str, infer_datetime_format=True)

# ----------------------------
# Helper: Format datetime as "M/D/YYYY H:MM:SS AM/PM"
# ----------------------------
def format_clean_timestamp(dt):
    """Convert datetime to 'M/D/YYYY H:MM:SS AM/PM' without leading zeros."""
    if pd.isna(dt):
        return ""
    hour = dt.hour % 12
    hour = 12 if hour == 0 else hour
    am_pm = "AM" if dt.hour < 12 else "PM"
    return f"{dt.month}/{dt.day}/{dt.year} {hour}:{dt.minute:02d}:{dt.second:02d} {am_pm}"

# ----------------------------
# Main Processing
# ----------------------------
def process_file(file_path: Path):
    print(f"🔧 Processing {file_path.name}...")
    try:
        df = pd.read_csv(file_path, compression='gzip')
    except Exception as e:
        print(f"  ❌ Failed to read {file_path}: {e}")
        return

    # Step 1: Rename 'timestampes' to 'timestamps'
    if 'timestampes' in df.columns:
        df.rename(columns={'timestampes': 'timestamps'}, inplace=True)
    elif 'timestamps' not in df.columns:
        print(f"  ⚠️ No 'timestampes' or 'timestamps' column in {file_path}")
        return

    # Step 2: Parse existing string timestamps to datetime
    try:
        df['timestamps'] = df['timestamps'].apply(parse_timestamp_str)
    except Exception as e:
        print(f"  ❌ Timestamp parsing failed for {file_path}: {e}")
        return

    # Step 3: Re-format to clean string (no leading zeros)
    df['timestamps'] = df['timestamps'].apply(format_clean_timestamp)

    # Step 4: Save back
    output_path = OUTPUT_DIR / file_path.name
    df.to_csv(output_path, index=False, compression='gzip')
    print(f"  ✅ Saved {output_path}")

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    csv_files = list(BASE_DIR.glob("*_basis_1min_*.csv.gz"))
    if not csv_files:
        print(f"❌ No CSV files found in {BASE_DIR}")
    else:
        print(f"Found {len(csv_files)} files to process.")
        for f in sorted(csv_files):
            process_file(f)
        print("\n🎉 All files processed!")