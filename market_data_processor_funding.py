import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timezone
import time

# ----------------------------
# Binance API Helper Functions
# ----------------------------

def get_funding_rate_history(symbol: str, start_time: int, end_time: int, limit: int = 1000):
    """
    Fetch funding rate history from Binance Futures.
    Returns list of {'fundingTime': int (ms), 'fundingRate': str}
    """
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {
        "symbol": symbol,
        "startTime": start_time,
        "endTime": end_time,
        "limit": min(limit, 1000)  # Binance max limit is 1000
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  ❌ Failed to fetch funding rate for {symbol}: {e}")
        return []

def get_index_price_klines(symbol: str, start_time: int, end_time: int, interval: str = "1m"):
    """
    Fetch index price klines (OHLC) at given interval.
    We only need close price as index_price.
    Returns list of [open_time, open, high, low, close, ...]
    """
    url = "https://fapi.binance.com/fapi/v1/indexPriceKlines"
    params = {
        "pair": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": 1500  # max allowed
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  ❌ Failed to fetch index price for {symbol}: {e}")
        return []

def prepare_binance_symbol(symbol: str) -> str:
    """Convert 'BTCUSDT' to 'BTCUSDT' (already correct); ensure no suffix issues."""
    return symbol

# ----------------------------
# Main Processing Function
# ----------------------------

def load_and_enrich_market_data(base_input_dir: str, base_output_dir: str, target_date: str):
    input_date_dir = Path(base_input_dir) / target_date
    if not input_date_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_date_dir}")

    output_base = Path(base_output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Parse date to timestamps (UTC)
    try:
        dt = datetime.strptime(target_date, "%Y%m%d")
        dt = dt.replace(tzinfo=timezone.utc)
        start_ts = int(dt.timestamp() * 1000)          # ms
        end_ts = int((dt.timestamp() + 86400) * 1000)  # next day
    except Exception as e:
        raise ValueError(f"Invalid date format: {target_date}. Use YYYYMMDD.") from e

    for symbol_dir in input_date_dir.iterdir():
        if not (symbol_dir.is_dir() and symbol_dir.name.endswith("USDT")):
            continue

        symbol = symbol_dir.name
        print(f"\n🔄 Processing {symbol}...")

        # --- Step 1: Load local market data ---
        parquet_files = [f for f in symbol_dir.glob("*.parquet") if "_inprogress" not in f.name]
        if not parquet_files:
            print(f"  ⚠️ No valid files for {symbol}")
            continue

        parquet_files.sort()
        dfs = []
        for f in parquet_files:
            try:
                df = pd.read_parquet(f, engine='fastparquet')
                dfs.append(df)
            except Exception as e:
                print(f"  ⚠️ Skip file {f}: {e}")
                continue

        if not dfs:
            print(f"  ⚠️ No data loaded for {symbol}")
            continue

        market_df = pd.concat(dfs, ignore_index=True)
        if 'time_str' not in market_df.columns:
            print(f"  ❌ 'time_str' column missing in {symbol}")
            continue

        # Normalize time_str to datetime with UTC
        try:
            market_df['timestamp'] = pd.to_datetime(market_df['time_str'], utc=True)
            market_df['timestamp_ms'] = market_df['timestamp'].view('int64') // 1_000_000  # nanosec → ms
        except Exception as e:
            print(f"  ❌ Failed to parse time_str: {e}")
            continue

        # --- Step 2: Fetch Binance funding rate ---
        binance_symbol = prepare_binance_symbol(symbol)
        funding_data = get_funding_rate_history(binance_symbol, start_ts, end_ts)
        funding_df = pd.DataFrame(funding_data)
        if not funding_df.empty:
            funding_df['fundingTime'] = pd.to_datetime(funding_df['fundingTime'], unit='ms', utc=True)
            funding_df.rename(columns={'fundingRate': 'funding_rate'}, inplace=True)
            funding_df['funding_rate'] = pd.to_numeric(funding_df['funding_rate'], errors='coerce')
        else:
            funding_df = pd.DataFrame(columns=['fundingTime', 'funding_rate'])

        # --- Step 3: Fetch Binance index price (1m klines) ---
        index_data = get_index_price_klines(binance_symbol, start_ts, end_ts, interval="1m")
        index_df = pd.DataFrame(index_data)
        if not index_df.empty:
            index_df = index_df.iloc[:, [0, 4]]  # [open_time, close]
            index_df.columns = ['open_time', 'index_price']
            index_df['open_time'] = pd.to_datetime(index_df['open_time'], unit='ms', utc=True)
            index_df['index_price'] = pd.to_numeric(index_df['index_price'], errors='coerce')
        else:
            index_df = pd.DataFrame(columns=['open_time', 'index_price'])

        # --- Step 4: Merge all data on timestamp ---
        # Start with market data
        merged = market_df.copy()

        # Merge funding_rate (forward-fill or exact match)
        if not funding_df.empty:
            merged = pd.merge_asof(
                merged.sort_values('timestamp'),
                funding_df[['fundingTime', 'funding_rate']].sort_values('fundingTime'),
                left_on='timestamp',
                right_on='fundingTime',
                direction='backward'  # use latest funding rate before or at timestamp
            ).drop(columns=['fundingTime'])
        else:
            merged['funding_rate'] = pd.NA

        # Merge index_price (exact 1-minute alignment)
        if not index_df.empty:
            merged = pd.merge_asof(
                merged.sort_values('timestamp'),
                index_df.sort_values('open_time'),
                left_on='timestamp',
                right_on='open_time',
                direction='nearest',  # or 'backward' if you prefer
                tolerance=pd.Timedelta('30s')  # max 30s deviation
            ).drop(columns=['open_time'])
        else:
            merged['index_price'] = pd.NA

        # --- Step 5: Save to CSV.GZ ---
        output_symbol_dir = output_base / symbol
        output_symbol_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_symbol_dir / f"market_{symbol}_{target_date}.csv.gz"

        # Drop helper columns
        merged.drop(columns=['timestamp', 'timestamp_ms'], inplace=True, errors='ignore')

        merged.to_csv(output_file, index=False, compression='gzip')
        print(f"  ✅ Saved enriched data to {output_file} ({len(merged)} rows)")

        # Be polite to Binance API
        time.sleep(0.1)

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    BASE_INPUT_DIR = "./datasets/market_data"
    BASE_OUTPUT_DIR = "./datasets/market_processed"
    TARGET_DATE = "20260116"  # Change as needed

    load_and_enrich_market_data(BASE_INPUT_DIR, BASE_OUTPUT_DIR, TARGET_DATE)