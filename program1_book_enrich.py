# program1_book_enrich.py
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timezone
import time

def clean_columns(df, stream_type):
    """删除指定列，并为保留列加前缀"""
    drop_cols = {
        "update_id", "bid_px", "bid_qty", "ask_px", "ask_qty",
        "lag_ms", "ts_from_last_ms", "event_ts", "transaction_ts", "local_ts"
    }
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    prefix = "spot_" if stream_type == "spot_l5" else "swap_"
    rename_map = {col: f"{prefix}{col}" for col in df.columns if col not in ["time_str", "symbol"]}
    df = df.rename(columns=rename_map)
    return df[["time_str"] + [c for c in df.columns if c != "time_str"]]

def fetch_funding_and_index(symbol: str, start_ts: int, end_ts: int):
    # Funding Rate
    try:
        fr = requests.get("https://fapi.binance.com/fapi/v1/fundingRate", params={
            "symbol": symbol, "startTime": start_ts, "endTime": end_ts, "limit": 1000
        }, timeout=10).json()
        funding_df = pd.DataFrame(fr)
        if not funding_df.empty:
            funding_df['fundingTime'] = pd.to_datetime(funding_df['fundingTime'], unit='ms', utc=True)
            funding_df['funding_rate'] = pd.to_numeric(funding_df['fundingRate'])
            funding_df = funding_df[['fundingTime', 'funding_rate']]
        else:
            funding_df = pd.DataFrame(columns=['fundingTime', 'funding_rate'])
    except Exception as e:
        print(f"  ⚠️ Funding fetch failed: {e}")
        funding_df = pd.DataFrame(columns=['fundingTime', 'funding_rate'])

    # Index Price
    try:
        idx = requests.get("https://fapi.binance.com/fapi/v1/indexPriceKlines", params={
            "pair": symbol, "interval": "1m", "startTime": start_ts, "endTime": end_ts, "limit": 1500
        }, timeout=10).json()
        index_df = pd.DataFrame(idx)
        if not index_df.empty:
            index_df = index_df.iloc[:, [0, 4]]
            index_df.columns = ['open_time', 'index_price']
            index_df['open_time'] = pd.to_datetime(index_df['open_time'], unit='ms', utc=True)
            index_df['index_price'] = pd.to_numeric(index_df['index_price'])
        else:
            index_df = pd.DataFrame(columns=['open_time', 'index_price'])
    except Exception as e:
        print(f"  ⚠️ Index price fetch failed: {e}")
        index_df = pd.DataFrame(columns=['open_time', 'index_price'])
    
    return funding_df, index_df

def process_book_data(date_str: str):
    input_base = Path("./dataset/market")
    output_base = Path("./dataset/market_processed")
    
    input_dir = input_base / date_str
    output_dir = output_base / date_str
    output_dir.mkdir(parents=True, exist_ok=True)

    # Time range
    dt = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
    start_ts = int(dt.timestamp() * 1000)
    end_ts = start_ts + 86400_000

    for symbol_dir in input_dir.iterdir():
        if not (symbol_dir.is_dir() and symbol_dir.name.endswith("USDT")):
            continue

        symbol = symbol_dir.name
        print(f"📖 Processing book data for {symbol}...")

        # Find files by stream type (assume filename contains 'spot_l5' or 'future_l5')
        spot_files = []
        future_files = []
        for f in symbol_dir.glob("*.parquet"):
            if "_inprogress" in f.name:
                continue
            if "spot_l5" in f.name:
                spot_files.append(f)
            elif "future_l5" in f.name:
                future_files.append(f)

        spot_df, future_df = None, None

        if spot_files:
            dfs = [pd.read_parquet(f, engine='fastparquet') for f in sorted(spot_files)]
            spot_df = pd.concat(dfs, ignore_index=True)
            spot_df = spot_df[spot_df["stream"] == "spot_l5"]
            spot_df = clean_columns(spot_df, "spot_l5")

        if future_files:
            dfs = [pd.read_parquet(f, engine='fastparquet') for f in sorted(future_files)]
            future_df = pd.concat(dfs, ignore_index=True)
            future_df = future_df[future_df["stream"] == "future_l5"]
            future_df = clean_columns(future_df, "future_l5")

        if spot_df is None and future_df is None:
            print(f"  ⚠️ No book data for {symbol}")
            continue

        # Parse time_str
        for df in [spot_df, future_df]:
            if df is not None:
                df['time_str'] = pd.to_datetime(df['time_str'], utc=True)

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
        funding_df, index_df = fetch_funding_and_index(symbol, start_ts, end_ts)
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
        symbol_out_dir = output_dir / symbol
        symbol_out_dir.mkdir(exist_ok=True)
        out_file = symbol_out_dir / f"book_{symbol}_{date_str}.csv.gz"
        merged.to_csv(out_file, index=False, compression='gzip')
        print(f"  ✅ Saved to {out_file}")

        time.sleep(0.1)

if __name__ == "__main__":
    process_book_data("20260116")