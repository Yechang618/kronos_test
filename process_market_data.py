#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Data Processing Pipeline (v2)
- Reads ALL .parquet files in ./dataset/market/{date}/{symbol}USDT/
- Splits by 'stream' column: 
    - "spot_l5", "future_l5" → book data
    - "spot_trade", "future_trade" → trade data
- Processes and saves to ./dataset/market_processed/{date}/{symbol}/
"""

import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timezone
import time
import sys
import argparse

# ============================
# Utility Functions
# ============================

def clean_book_columns(df, is_spot: bool):
    """Remove unwanted columns and add prefix based on stream type"""
    drop_cols = {
        "update_id", "bid_px", "bid_qty", "ask_px", "ask_qty",
        "lag_ms", "ts_from_last_ms", "event_ts", "transaction_ts", "local_ts"
    }
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    prefix = "spot_" if is_spot else "swap_"
    rename_map = {col: f"{prefix}{col}" for col in df.columns if col not in ["time_str", "symbol"]}
    df = df.rename(columns=rename_map)
    return df[["time_str"] + [c for c in df.columns if c != "time_str"]]

def safe_read_parquet(path):
    try:
        return pd.read_parquet(path, engine='fastparquet')
    except Exception as e:
        print(f"  ⚠️ Failed to read {path.name}: {e}")
        return None

def parse_time_str(series):
    return pd.to_datetime(series, utc=True)

# ============================
# Book Processing
# ============================

def process_book_data(date_str: str):
    print(f"\n{'='*50}\n📖 Processing BOOK data (spot_l5 + future_l5) for {date_str}\n{'='*50}")
    input_dir = Path("./datasets/market_data") / date_str
    output_dir = Path("./datasets/market_processed") / date_str
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return

    dt = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
    start_ts = int(dt.timestamp() * 1000)
    end_ts = start_ts + 86400_000

    for symbol_dir in input_dir.iterdir():
        if not (symbol_dir.is_dir() and symbol_dir.name.endswith("USDT")):
            continue

        symbol = symbol_dir.name
        print(f"\n→ Symbol: {symbol}")

        # Collect ALL parquet files
        all_files = [f for f in symbol_dir.glob("*.parquet") if "_inprogress" not in f.name]
        if not all_files:
            print(f"  ⚠️ No files")
            continue

        spot_dfs, future_dfs = [], []

        for f in sorted(all_files):
            df = safe_read_parquet(f)
            if df is None or "stream" not in df.columns:
                continue

            # Filter by stream type
            spot_part = df[df["stream"] == "spot_l5"]
            future_part = df[df["stream"] == "future_l5"]

            if not spot_part.empty:
                spot_dfs.append(spot_part)
            if not future_part.empty:
                future_dfs.append(future_part)

        spot_df = pd.concat(spot_dfs, ignore_index=True) if spot_dfs else None
        future_df = pd.concat(future_dfs, ignore_index=True) if future_dfs else None

        if spot_df is None and future_df is None:
            print(f"  ⚠️ No book data (spot_l5/future_l5)")
            continue

        # Clean columns
        if spot_df is not None:
            spot_df = clean_book_columns(spot_df, is_spot=True)
            spot_df['time_str'] = parse_time_str(spot_df['time_str'])
        if future_df is not None:
            future_df = clean_book_columns(future_df, is_spot=False)
            future_df['time_str'] = parse_time_str(future_df['time_str'])

        # Merge
        if spot_df is not None and future_df is not None:
            merged = pd.merge_asof(
                spot_df.sort_values('time_str'),
                future_df.sort_values('time_str'),
                on='time_str',
                direction='nearest',
                tolerance=pd.Timedelta('100ms')
            )
        else:
            merged = spot_df if spot_df is not None else future_df

        # Fetch external data
        print(f"  🌐 Fetching funding & index for {symbol}...")
        try:
            fr_resp = requests.get("https://fapi.binance.com/fapi/v1/fundingRate", params={
                "symbol": symbol, "startTime": start_ts, "endTime": end_ts, "limit": 1000
            }, timeout=10)
            fr_data = fr_resp.json() if fr_resp.status_code == 200 else []
            funding_df = pd.DataFrame(fr_data)
            if not funding_df.empty:
                funding_df['fundingTime'] = pd.to_datetime(funding_df['fundingTime'], unit='ms', utc=True)
                funding_df['funding_rate'] = pd.to_numeric(funding_df['fundingRate'])
                funding_df = funding_df[['fundingTime', 'funding_rate']]
            else:
                funding_df = pd.DataFrame(columns=['fundingTime', 'funding_rate'])
        except Exception as e:
            print(f"  ⚠️ Funding fetch error: {e}")
            funding_df = pd.DataFrame(columns=['fundingTime', 'funding_rate'])

        try:
            idx_resp = requests.get("https://fapi.binance.com/fapi/v1/indexPriceKlines", params={
                "pair": symbol, "interval": "1m", "startTime": start_ts, "endTime": end_ts, "limit": 1500
            }, timeout=10)
            idx_data = idx_resp.json() if idx_resp.status_code == 200 else []
            index_df = pd.DataFrame(idx_data)
            if not index_df.empty:
                index_df = index_df.iloc[:, [0, 4]]
                index_df.columns = ['open_time', 'index_price']
                index_df['open_time'] = pd.to_datetime(index_df['open_time'], unit='ms', utc=True)
                index_df['index_price'] = pd.to_numeric(index_df['index_price'])
            else:
                index_df = pd.DataFrame(columns=['open_time', 'index_price'])
        except Exception as e:
            print(f"  ⚠️ Index price fetch error: {e}")
            index_df = pd.DataFrame(columns=['open_time', 'index_price'])

        # Merge external
        merged = merged.sort_values('time_str')
        if not funding_df.empty:
            merged = pd.merge_asof(
                merged, funding_df.sort_values('fundingTime'),
                left_on='time_str', right_on='fundingTime',
                direction='backward'
            ).drop(columns=['fundingTime'])
        else:
            merged['funding_rate'] = pd.NA

        if not index_df.empty:
            merged = pd.merge_asof(
                merged, index_df.sort_values('open_time'),
                left_on='time_str', right_on='open_time',
                direction='nearest',
                tolerance=pd.Timedelta('30s')
            ).drop(columns=['open_time'])
        else:
            merged['index_price'] = pd.NA

        # Save
        symbol_out = output_dir / symbol
        symbol_out.mkdir(exist_ok=True)
        out_file = symbol_out / f"book_{symbol}_{date_str}.csv.gz"
        merged.to_csv(out_file, index=False, compression='gzip')
        print(f"  ✅ Saved book data ({len(merged)} rows)")

        time.sleep(0.1)

# ============================
# Trade Processing
# ============================

def process_trade_data(date_str: str):
    print(f"\n{'='*50}\n🛒 Processing TRADE data (spot_trade + future_trade) for {date_str}\n{'='*50}")
    input_dir = Path("./datasets/market_data") / date_str
    output_dir = Path("./datasets/market_processed") / date_str
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return

    for symbol_dir in input_dir.iterdir():
        if not (symbol_dir.is_dir() and symbol_dir.name.endswith("USDT")):
            continue

        symbol = symbol_dir.name
        print(f"\n→ Symbol: {symbol}")

        all_files = [f for f in symbol_dir.glob("*.parquet") if "_inprogress" not in f.name]
        if not all_files:
            print(f"  ⚠️ No files")
            continue

        trade_dfs = []
        for f in sorted(all_files):
            df = safe_read_parquet(f)
            if df is None or "stream" not in df.columns:
                continue

            spot_trades = df[df["stream"] == "spot_trade"].copy()
            future_trades = df[df["stream"] == "future_trade"].copy()

            if not spot_trades.empty:
                spot_trades["trade_type"] = "spot"
                trade_dfs.append(spot_trades)
            if not future_trades.empty:
                future_trades["trade_type"] = "swap"
                trade_dfs.append(future_trades)

        if not trade_dfs:
            print(f"  ⚠️ No trade data")
            continue

        combined = pd.concat(trade_dfs, ignore_index=True)
        combined['time_str'] = parse_time_str(combined['time_str'])

        symbol_out = output_dir / symbol
        symbol_out.mkdir(exist_ok=True)
        out_file = symbol_out / f"trades_{symbol}_{date_str}.csv.gz"
        combined.to_csv(out_file, index=False, compression='gzip')
        print(f"  ✅ Saved trade data ({len(combined)} rows)")

# ============================
# Main Entry
# ============================

def main():
    parser = argparse.ArgumentParser(description="Process market data for a given date.")
    parser.add_argument("date", help="Date in YYYYMMDD format, e.g., 20260116")
    args = parser.parse_args()

    date_str = args.date
    if not (len(date_str) == 8 and date_str.isdigit()):
        print("❌ Error: Date must be in YYYYMMDD format (e.g., 20260116)")
        sys.exit(1)

    try:
        datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        print("❌ Invalid date")
        sys.exit(1)

    process_book_data(date_str)
    process_trade_data(date_str)

    print(f"\n🎉 All done for {date_str}!")

if __name__ == "__main__":
    main()