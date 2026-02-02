#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-process book data to generate 1-minute basis OHLCV + amount signals.
"""

import pandas as pd
from pathlib import Path
import numpy as np
import os
from datetime import datetime

# ----------------------------
# Configuration
# ----------------------------
INPUT_BASE = Path("./dataset/market_processed")
OUTPUT_DIR = Path("./datasets/processed/basis_1min_task7")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Date range to process
START_DATE = "20251229"
END_DATE = "20260121"

# Get all dates in range
date_range = pd.date_range(
    start=pd.to_datetime(START_DATE, format="%Y%m%d"),
    end=pd.to_datetime(END_DATE, format="%Y%m%d"),
    freq="D"
).strftime("%Y%m%d").tolist()

# Discover all symbols from the first date (or any date)
sample_date = START_DATE
sample_path = INPUT_BASE / sample_date
if not sample_path.exists():
    raise FileNotFoundError(f"Sample path not found: {sample_path}")

symbols = [d.name for d in sample_path.iterdir() if d.is_dir() and d.name.endswith("USDT")]
print(f"Found symbols: {symbols}")

# ----------------------------
# Processing Function
# ----------------------------

def process_symbol(symbol: str):
    print(f"\n📊 Processing symbol: {symbol}")
    all_dfs = []

    for date_str in date_range:
        book_file = INPUT_BASE / date_str / symbol / f"book_{symbol}_{date_str}.csv.gz"
        if not book_file.exists():
            print(f"  ⚠️ Missing {book_file}")
            continue

        try:
            df = pd.read_csv(book_file, compression='gzip')
        except Exception as e:
            print(f"  ❌ Failed to read {book_file}: {e}")
            continue

        # Keep only required columns
        required_cols = [
            'time_str',
            'spot_bid1_px', 'spot_ask1_px',
            'swap_bid1_px', 'swap_ask1_px',
            'funding_rate', 'index_price'
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"  ⚠️ Missing columns in {book_file}: {missing}")
            continue

        # df = df[required_cols].copy()
        # df['timestamp'] = pd.to_datetime(df['time_str'], utc=True)
        df = df[required_cols].copy()
        # ✅ Fix: Use ISO8601 format to handle variable precision
        df['timestamp'] = pd.to_datetime(df['time_str'], format='ISO8601')
        df.drop(columns=['time_str'], inplace=True)

        # Drop rows with critical NaN
        df = df.dropna(subset=[
            'spot_bid1_px', 'spot_ask1_px',
            'swap_bid1_px', 'swap_ask1_px',
            'index_price'
        ])

        if df.empty:
            continue

        # Compute WAP and basis
        df['wap_bid1_px'] = (df['swap_bid1_px'] + df['swap_ask1_px']) / 2
        df['basis_bid'] = np.log(df['wap_bid1_px']) - np.log(df['spot_ask1_px'])
        df['basis_ask'] = np.log(df['swap_ask1_px']) - np.log(df['spot_bid1_px'])
        df['mid_basis'] = (df['basis_bid'] + df['basis_ask']) / 2
        df['amount'] = np.log((df['spot_bid1_px'] + df['spot_ask1_px']) / 2) - np.log(df['index_price'])

        all_dfs.append(df)

    if not all_dfs:
        print(f"  ❌ No valid data for {symbol}")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df.sort_values('timestamp').set_index('timestamp')

    # Resample to 1-minute bars
    def agg_func(group):
        if group.empty:
            return pd.Series({
                'open': np.nan,
                'high': np.nan,
                'low': np.nan,
                'close': np.nan,
                'volume': np.nan,
                'amount': np.nan
            })

        open_val = group['mid_basis'].iloc[0]
        close_val = group['mid_basis'].iloc[-1]
        high_val = group['basis_bid'].max()
        low_val = group['basis_ask'].min()
        volume_val = group['funding_rate'].dropna().iloc[-1] if not group['funding_rate'].dropna().empty else np.nan
        amount_val = group['amount'].iloc[-1]

        return pd.Series({
            'open': open_val,
            'high': high_val,
            'low': low_val,
            'close': close_val,
            'volume': volume_val,
            'amount': amount_val
        })

    ohlcv = full_df.groupby(pd.Grouper(freq='1T')).apply(agg_func)
    ohlcv = ohlcv.dropna(subset=['open', 'high', 'low', 'close'], how='all')
    ohlcv.index.name = 'timestampes'

    if ohlcv.empty:
        print(f"  ⚠️ No OHLCV generated for {symbol}")
        return

    # Save by year-month
    ohlcv_reset = ohlcv.reset_index()
    ohlcv_reset['year_month'] = ohlcv_reset['timestampes'].dt.to_period('M')

    for period, group in ohlcv_reset.groupby('year_month'):
        year = period.year
        month = str(period.month).zfill(2)
        filename = f"{symbol}_basis_1min_{year}-{month}.csv.gz"
        output_file = OUTPUT_DIR / filename

        # Append if file exists, else write
        if output_file.exists():
            existing = pd.read_csv(output_file, compression='gzip')
            combined = pd.concat([existing, group.drop(columns=['year_month'])], ignore_index=True)
            combined = combined.drop_duplicates(subset=['timestampes']).sort_values('timestampes')
        else:
            combined = group.drop(columns=['year_month'])

        combined.to_csv(output_file, index=False, compression='gzip')
        print(f"  ✅ Saved {len(combined)} rows to {output_file}")

# ----------------------------
# Main Execution
# ----------------------------

if __name__ == "__main__":
    for symbol in symbols:
        try:
            process_symbol(symbol)
        except Exception as e:
            print(f"❌ Fatal error for {symbol}: {e}")

    print("\n🎉 Basis 1-minute processing completed!")