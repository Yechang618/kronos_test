# program2_trades_stack.py
import pandas as pd
from pathlib import Path

def process_trade_data(date_str: str):
    input_base = Path("./dataset/market")
    output_base = Path("./dataset/market_processed")
    
    input_dir = input_base / date_str
    output_dir = output_base / date_str
    output_dir.mkdir(parents=True, exist_ok=True)

    for symbol_dir in input_dir.iterdir():
        if not (symbol_dir.is_dir() and symbol_dir.name.endswith("USDT")):
            continue

        symbol = symbol_dir.name
        print(f"🛒 Processing trade data for {symbol}...")

        trade_files = []
        for f in symbol_dir.glob("*.parquet"):
            if "_inprogress" in f.name:
                continue
            if "spot_trade" in f.name or "future_trade" in f.name:
                trade_files.append(f)

        if not trade_files:
            print(f"  ⚠️ No trade files for {symbol}")
            continue

        dfs = []
        for f in sorted(trade_files):
            try:
                df = pd.read_parquet(f, engine='fastparquet')
                if "spot_trade" in f.name:
                    df["trade_type"] = "spot"
                elif "future_trade" in f.name:
                    df["trade_type"] = "swap"
                else:
                    continue
                dfs.append(df)
            except Exception as e:
                print(f"  ⚠️ Skip {f}: {e}")

        if not dfs:
            continue

        combined = pd.concat(dfs, ignore_index=True)
        combined['time_str'] = pd.to_datetime(combined['time_str'], utc=True)

        # Save
        symbol_out_dir = output_dir / symbol
        symbol_out_dir.mkdir(exist_ok=True)
        out_file = symbol_out_dir / f"trades_{symbol}_{date_str}.csv.gz"
        combined.to_csv(out_file, index=False, compression='gzip')
        print(f"  ✅ Saved to {out_file}")

if __name__ == "__main__":
    process_trade_data("20260116")