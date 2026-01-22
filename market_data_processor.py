import pandas as pd
from pathlib import Path

def load_and_save_market_data(base_input_dir: str, base_output_dir: str, target_date: str):
    """
    Load Parquet files for a given date, process by symbol, and save as compressed CSV.
    
    Input structure:
        ./datasets/market_data/20260116/ATOMUSDT/*.parquet
        
    Output structure:
        ./dataset/market_processed/ATOMUSDT/market_ATOMUSDT_20260116.csv.gz
    """
    input_date_dir = Path(base_input_dir) / target_date
    if not input_date_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_date_dir}")

    output_base = Path(base_output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Iterate over each symbol subdirectory (e.g., ATOMUSDT/, BTCUSDT/)
    for symbol_dir in input_date_dir.iterdir():
        if not (symbol_dir.is_dir() and symbol_dir.name.endswith("USDT")):
            continue

        symbol = symbol_dir.name
        print(f"Processing {symbol}...")

        # Collect and filter parquet files
        parquet_files = []
        for f in symbol_dir.glob("*.parquet"):
            if "_inprogress" in f.name:
                continue
            parquet_files.append(f)
        
        if not parquet_files:
            print(f"  ⚠️ No valid files for {symbol}")
            continue

        parquet_files.sort()

        # Load all parts
        dfs = []
        for f in parquet_files:
            try:
                df = pd.read_parquet(f, engine='fastparquet')
                dfs.append(df)
            except Exception as e:
                print(f"  ⚠️ Failed to read {f}: {e}")
                continue

        if not dfs:
            print(f"  ⚠️ No data loaded for {symbol}")
            continue

        # Concatenate
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"  → Loaded {len(combined_df)} rows")

        # Define output path
        output_symbol_dir = output_base / symbol
        output_symbol_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_symbol_dir / f"market_{symbol}_{target_date}.csv.gz"

        # Save as compressed CSV
        combined_df.to_csv(output_file, index=False, compression='gzip')
        print(f"  → Saved to {output_file}")

# ----------------------------
# Configuration
# ----------------------------
if __name__ == "__main__":
    BASE_INPUT_DIR = "./datasets/market_data"
    BASE_OUTPUT_DIR = "./datasets/market_processed"
    TARGET_DATE = "20260116"  # 修改为你需要的日期

    load_and_save_market_data(BASE_INPUT_DIR, BASE_OUTPUT_DIR, TARGET_DATE)