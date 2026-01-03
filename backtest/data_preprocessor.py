# backtest/data_preprocessor.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_timestamp_series(series):
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        max_val = s.max()
        if max_val > 1e17:
            unit = 'ns'
        elif max_val > 1e14:
            unit = 'us'
        elif max_val > 1e11:
            unit = 'ms'
        else:
            unit = 's'
        return pd.to_datetime(s, unit=unit)
    else:
        return pd.to_datetime(s)

def process_book_df(df, prefix, levels=3):  # 只处理前3档
    df_out = pd.DataFrame(index=df.index)
    for level in range(levels):
        price_col = f"bids[{level}].price"
        amount_col = f"bids[{level}].amount"
        if price_col in df.columns:
            df_out[f"{prefix}_bid{level}_price"] = df[df[price_col] > 0][price_col]
        if amount_col in df.columns:
            df_out[f"{prefix}_bid{level}_amount"] = df[df[amount_col] > 0][amount_col]
        
        price_col = f"asks[{level}].price"
        amount_col = f"asks[{level}].amount"
        if price_col in df.columns:
            df_out[f"{prefix}_ask{level}_price"] = df[df[price_col] > 0][price_col]
        if amount_col in df.columns:
            df_out[f"{prefix}_ask{level}_amount"] = df[df[amount_col] > 0][amount_col]
    
    return df_out

def calculate_basis_metrics(df):
    df = df.copy()
    df['basis1_price'] = np.log(df['swap_bid0_price']) - np.log(df['spot_ask0_price'])
    df['basis2_price'] = np.log(df['swap_ask0_price']) - np.log(df['spot_bid0_price'])
    df['basis1_volume'] = np.minimum(df['swap_bid0_amount'], df['spot_ask0_amount'])
    df['basis2_volume'] = np.minimum(df['swap_ask0_amount'], df['spot_bid0_amount'])
    df['basis1_volume'] = df['basis1_volume'].clip(lower=0)
    df['basis2_volume'] = df['basis2_volume'].clip(lower=0)
    return df

def plot_basis_comparison(df, symbol, start_time, end_time, output_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['basis1_price'], label='Basis1', alpha=0.8)
    plt.plot(df.index, df['basis2_price'], label='Basis2', alpha=0.8)
    plt.title(f'{symbol} Basis Comparison ({start_time} to {end_time})')
    plt.xlabel('Time')
    plt.ylabel('Basis')
    plt.legend()
    plt.grid(True, alpha=0.7)
    plt.tight_layout()
    filename = f"basis_comparison_{symbol}_{start_time}_{end_time}.png"
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()

def load_and_process_data(symbol, date_range, load_dir, output_dir):
    """逐日处理，避免内存累积"""
    for single_date in date_range:
        date_str = single_date.strftime("%Y-%m-%d")
        print(f"Processing {date_str} for {symbol}...")
        
        pair = f"{symbol}USDT"
        book_path = load_dir / f"book/binance_book_snapshot_25_{date_str}_{pair}.csv.gz"
        fbook_path = load_dir / f"fbook/binance-futures_book_snapshot_25_{date_str}_{pair}.csv.gz"
        
        if not (book_path.exists() and fbook_path.exists()):
            print(f"Skipping {date_str}: missing files")
            continue
        
        # 关键：使用 1秒频率
        full_index = pd.date_range(f"{date_str} 00:00:00", f"{date_str} 23:59:59", freq="100ms")
        
        try:
            # 处理现货
            spot_df = pd.read_csv(book_path)
            spot_df.index = parse_timestamp_series(spot_df["timestamp"])
            spot_df = spot_df.sort_index()
            if spot_df.index.duplicated().any():
                spot_df = spot_df[~spot_df.index.duplicated(keep='last')]
            spot_feat = process_book_df(spot_df, "spot")
            spot_res = spot_feat.reindex(full_index, method='pad')
            
            # 处理合约
            swap_df = pd.read_csv(fbook_path)
            swap_df.index = parse_timestamp_series(swap_df["timestamp"])
            swap_df = swap_df.sort_index()
            if swap_df.index.duplicated().any():
                swap_df = swap_df[~swap_df.index.duplicated(keep='last')]
            swap_feat = process_book_df(swap_df, "swap")
            swap_res = swap_feat.reindex(full_index, method='pad')
            
            df_day = pd.concat([spot_res, swap_res], axis=1)
            df_day = calculate_basis_metrics(df_day)
            df_day.index.name = "timestamp"
            
            if df_day["spot_bid0_price"].first_valid_index() is not None:
                # 保存当日数据
                daily_file = output_dir / f"backtest_{symbol}_{date_str}.csv"
                df_day.to_csv(daily_file)
                print(f"  ✅ Saved {daily_file.name}")
            else:
                print(f"  ⚠️ No valid data for {date_str}")
                
        except Exception as e:
            print(f"  ❌ Error on {date_str}: {e}")

def main():
    symbol = "SOL"
    load_dir = Path("D:/data/datasets")
    output_dir = Path("./backtest/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    time_periods = [("2025-10-01", "2025-10-07")]
    # time_periods = [("2025-10-14", "2025-10-20")]
    # time_periods = [("2025-10-22", "2025-10-28")]    
    
    
    for start_str, end_str in time_periods:
        print(f"\nProcessing: {start_str} to {end_str}")
        date_range = pd.date_range(start=start_str, end=end_str, freq="D")
        
        # 逐日处理
        load_and_process_data(symbol, date_range, load_dir, output_dir)
        
        # 合并数据
        combined_df = pd.DataFrame()
        for date in date_range:
            date_str = date.strftime("%Y-%m-%d")
            file_path = output_dir / f"backtest_{symbol}_{date_str}.csv"
            if file_path.exists():
                daily_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                combined_df = pd.concat([combined_df, daily_df])
        
        if not combined_df.empty:
            start_time = start_str.replace("-", "")
            end_time = end_str.replace("-", "")
            csv_file = output_dir / f"backtest_{symbol}_{start_time}_{end_time}.csv"
            combined_df.to_csv(csv_file)
            plot_basis_comparison(combined_df, symbol, start_time, end_time, output_dir)
            print(f"✅ Completed period {start_str} to {end_str}")

if __name__ == "__main__":
    main()