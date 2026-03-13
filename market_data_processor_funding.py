import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta
import time

# ----------------------------
# Binance API Helpers (fixed URLs)
# ----------------------------

def get_funding_rate_history(symbol: str, start_time: int, end_time: int, limit: int = 1000):
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": symbol, "startTime": start_time, "endTime": end_time, "limit": min(limit, 1000)}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  ❌ Funding fetch failed for {symbol}: {e}")
        return []

def get_index_price_klines(symbol: str, start_time: int, end_time: int, interval: str = "1m"):
    url = "https://fapi.binance.com/fapi/v1/indexPriceKlines"
    params = {"pair": symbol, "interval": interval, "startTime": start_time, "endTime": end_time, "limit": 1500}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  ❌ Index price fetch failed for {symbol}: {e}")
        return []

def clean_book_columns(df, is_spot: bool):
    drop_cols = {
        "update_id", "bid_px", "bid_qty", "ask_px", "ask_qty",
        "lag_ms", "ts_from_last_ms", "event_ts", "transaction_ts", "local_ts"
    }
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    prefix = "spot_" if is_spot else "swap_"
    rename_map = {col: f"{prefix}{col}" for col in df.columns if col not in ["time_str", "symbol"]}
    return df.rename(columns=rename_map)

def parse_time_str(series):
    return pd.to_datetime(series, utc=True).astype('datetime64[ns, UTC]')

# ----------------------------
# Main Processing Function (Book + Trade)
# ----------------------------

def load_and_enrich_market_data(base_input_dir: str, base_output_dir: str, target_date: str):
    input_date_dir = Path(base_input_dir) / target_date
    if not input_date_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_date_dir}")

    output_base = Path(base_output_dir) / target_date
    output_base.mkdir(parents=True, exist_ok=True)

    dt = datetime.strptime(target_date, "%Y%m%d").replace(tzinfo=timezone.utc)
    start_ts = int(dt.timestamp() * 1000)
    end_ts = start_ts + 86400_000

    for symbol_dir in input_date_dir.iterdir():
        if not (symbol_dir.is_dir() and symbol_dir.name.endswith("USDT")):
            continue

        symbol = symbol_dir.name
        print(f"\n🔄 Processing {symbol}...")

        # Load all files
        parquet_files = [f for f in symbol_dir.glob("*.parquet") if "_inprogress" not in f.name]
        if not parquet_files:
            print(f"  ⚠️ No files")
            continue

        dfs = []
        for f in sorted(parquet_files):
            try:
                df = pd.read_parquet(f, engine='fastparquet')
                dfs.append(df)
            except Exception as e:
                print(f"  ⚠️ Skip {f}: {e}")
                continue

        if not dfs:
            continue
        full_df = pd.concat(dfs, ignore_index=True)

        if 'stream' not in full_df.columns or 'time_str' not in full_df.columns:
            print(f"  ❌ Missing key columns")
            continue

        # ======================
        # 1. Process BOOK data
        # ======================
        spot_l5 = full_df[full_df["stream"] == "spot_l5"].copy()
        future_l5 = full_df[full_df["stream"] == "future_l5"].copy()

        if not spot_l5.empty or not future_l5.empty:
            if not spot_l5.empty:
                spot_l5 = clean_book_columns(spot_l5, is_spot=True)
                spot_l5['time_str'] = parse_time_str(spot_l5['time_str'])
            if not future_l5.empty:
                future_l5 = clean_book_columns(future_l5, is_spot=False)
                future_l5['time_str'] = parse_time_str(future_l5['time_str'])

            if not spot_l5.empty and not future_l5.empty:
                book_merged = pd.merge_asof(
                    spot_l5.sort_values('time_str'),
                    future_l5.sort_values('time_str'),
                    on='time_str',
                    direction='nearest',
                    tolerance=pd.Timedelta('100ms')
                )
            else:
                book_merged = spot_l5 if not spot_l5.empty else future_l5

            # Fetch external data
            funding_data = get_funding_rate_history(symbol, start_ts, end_ts)
            funding_df = pd.DataFrame(funding_data)
            if not funding_df.empty:
                funding_df['fundingTime'] = pd.to_datetime(funding_df['fundingTime'], unit='ms', utc=True).astype('datetime64[ns, UTC]')
                funding_df['funding_rate'] = pd.to_numeric(funding_df['fundingRate'])
                funding_df = funding_df[['fundingTime', 'funding_rate']]
            else:
                funding_df = pd.DataFrame(columns=['fundingTime', 'funding_rate'])

            index_data = get_index_price_klines(symbol, start_ts, end_ts)
            index_df = pd.DataFrame(index_data)
            if not index_df.empty:
                index_df = index_df.iloc[:, [0, 4]]
                index_df.columns = ['open_time', 'index_price']
                index_df['open_time'] = pd.to_datetime(index_df['open_time'], unit='ms', utc=True).astype('datetime64[ns, UTC]')
                index_df['index_price'] = pd.to_numeric(index_df['index_price'])
            else:
                index_df = pd.DataFrame(columns=['open_time', 'index_price'])

            book_merged = book_merged.sort_values('time_str')
            if not funding_df.empty:
                book_merged = pd.merge_asof(
                    book_merged, funding_df.sort_values('fundingTime'),
                    left_on='time_str', right_on='fundingTime',
                    direction='backward'
                ).drop(columns=['fundingTime'])
            else:
                book_merged['funding_rate'] = pd.NA

            if not index_df.empty:
                book_merged = pd.merge_asof(
                    book_merged, index_df.sort_values('open_time'),
                    left_on='time_str', right_on='open_time',
                    direction='nearest',
                    tolerance=pd.Timedelta('30s')
                ).drop(columns=['open_time'])
            else:
                book_merged['index_price'] = pd.NA

            # Save book
            out_dir = output_base / symbol
            out_dir.mkdir(exist_ok=True)
            book_file = out_dir / f"book_{symbol}_{target_date}.csv.gz"
            book_merged.to_csv(book_file, index=False, compression='gzip')
            print(f"  ✅ Saved book data ({len(book_merged)} rows)")

        # ======================
        # 2. Process TRADE data
        # ======================
        spot_trade = full_df[full_df["stream"] == "spot_trade"].copy()
        future_trade = full_df[full_df["stream"] == "future_trade"].copy()

        trade_dfs = []
        if not spot_trade.empty:
            spot_trade["trade_type"] = "spot"
            trade_dfs.append(spot_trade)
        if not future_trade.empty:
            future_trade["trade_type"] = "swap"
            trade_dfs.append(future_trade)
        print(f"Target date: {target_date}, symbol: {symbol}, spot trades: {len(spot_trade)}, future trades: {len(future_trade)}")
        if trade_dfs:
            trades_merged = pd.concat(trade_dfs, ignore_index=True)
            trades_merged['time_str'] = parse_time_str(trades_merged['time_str'])
            trades_merged = trades_merged.sort_values('time_str')

            out_dir = output_base / symbol
            out_dir.mkdir(exist_ok=True)
            trade_file = out_dir / f"trades_{symbol}_{target_date}.csv.gz"
            trades_merged.to_csv(trade_file, index=False, compression='gzip')
            print(f"  ✅ Saved trade data ({len(trades_merged)} rows)")
        else:
            print(f"  ⚠️ No trade data")

        time.sleep(0.1)
# ----------------------------

#     load_and_enrich_market_data(BASE_INPUT_DIR, BASE_OUTPUT_DIR, TARGET_DATE)
if __name__ == "__main__":
    BASE_INPUT_DIR = "D:/market_data"
    BASE_OUTPUT_DIR = "./dataset/market_processed"

    start_date = datetime(2026, 2, 18)
    end_date = datetime(2026, 3, 10)

    current = start_date
    while current <= end_date:
        target_date = current.strftime("%Y%m%d")
        print(f"\n{'='*70}")
        print(f"📅 Processing date: {target_date}")
        print(f"{'='*70}")
        try:
            load_and_enrich_market_data(BASE_INPUT_DIR, BASE_OUTPUT_DIR, target_date)
        except Exception as e:
            print(f"❌ Error on {target_date}: {e}")
        current += timedelta(days=1)

    print("\n🎉 Batch processing completed!")