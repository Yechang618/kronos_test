# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path

# ==============================
# 配置
# ==============================
base_dir = Path("datasets")
# load_dir = base_dir
load_dir = Path("D:/data/datasets")  
processed_dir = base_dir / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

symbols = ["ADA", "AIXBT", "APT", "AVAX", "BCH", "BNB", "BTC",  # 6
           "CHESS", "COMP", "DOGE", "DOT", "ENA", "ETC","ETH", # 13
           "FET", "FORM", "HBAR", "HFT", "KAITO", "LINK", "LTC", # 20
           "NEAR", "OM", "ONDO", "PNUT", "SOL", "TAO", # 26
           "THE", "TON", "TRX", "TURBO",  # 30
           "UNI", "XLM", "XRP", "ZEC", # 34
           ] # 
# symbol = "SOL" # 26
# ==============================
# 工具函数
# ==============================
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

def process_book_df(df, prefix):
    df_out = pd.DataFrame(index=df.index)
    # 提取第0、1、2档的价格和数量
    for level in range(3):
        price_col = f"bids[{level}].price"
        amount_col = f"bids[{level}].amount"
        if price_col in df.columns:
            df_out[f"{prefix}_bid{level}_price"] = df[price_col]
        if amount_col in df.columns:
            df_out[f"{prefix}_bid{level}_amount"] = df[amount_col]

        price_col = f"asks[{level}].price"
        amount_col = f"asks[{level}].amount"
        if price_col in df.columns:
            df_out[f"{prefix}_ask{level}_price"] = df[price_col]
        if amount_col in df.columns:
            df_out[f"{prefix}_ask{level}_amount"] = df[amount_col]
    return df_out

def process_trades_df(df, prefix):
    df = df.copy()
    if 'timestamp' not in df.columns:
        raise KeyError("'timestamp' column missing")

    if 'side' in df.columns:
        side_series = df['side']
    elif 'isBuyerMaker' in df.columns:
        side_series = df['isBuyerMaker'].map({True: 'sell', False: 'buy'})
    elif 'm' in df.columns:
        side_series = df['m'].map({True: 'sell', False: 'buy'})
    else:
        raise KeyError(f"Cannot determine side in {prefix} trades")

    if side_series.dtype == 'object':
        side_series = side_series.str.upper().map({'B': 'buy', 'S': 'sell', 'BUY': 'buy', 'SELL': 'sell'})
    else:
        side_series = side_series.map({1: 'buy', 0: 'sell', -1: 'sell', True: 'sell', False: 'buy'})

    df['side'] = side_series
    df = df[df['side'].isin(['buy', 'sell'])]

    if 'price' not in df.columns and 'p' in df.columns:
        df['price'] = df['p']
    if 'amount' not in df.columns and 'q' in df.columns:
        df['amount'] = df['q']

    required = ['price', 'amount', 'side', 'timestamp']
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing {col} in {prefix} trades")

    df_agg = df.groupby(['side', pd.Grouper(key='timestamp', freq='1s')]).agg(
        price=('price', 'mean'),
        amount=('amount', 'mean')
    ).reset_index()

    df_wide = df_agg.pivot(index='timestamp', columns='side', values=['price', 'amount'])
    df_wide.columns = [f"{prefix}_{side}_{col}" for col, side in df_wide.columns]

    day = df['timestamp'].iloc[0].strftime('%Y-%m-%d')
    full_sec = pd.date_range(start=f"{day} 00:00:00", end=f"{day} 23:59:59", freq='1s')
    return df_wide.reindex(full_sec)

for i in range(32, len(symbols)):
    symbol = symbols[i]
    quote = "USDT"
    pair = f"{symbol}{quote}"

    start_date = "2025-01-01"
    end_date = "2025-10-29"
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    print(f"🚀 Processing {pair} from {start_date} to {end_date}")


    # ==============================
    # 主循环：按天加载并生成秒级数据
    # ==============================
    valid_dfs = []
    valid_dates = []

    for single_date in date_range:
        date_str = single_date.strftime("%Y-%m-%d")
        print(f"\n📆 Processing {pair} {date_str}...")

        patterns = {
            "book":     f"book/binance_book_snapshot_25_{date_str}_{pair}.csv.gz",
            "fbook":    f"fbook/binance-futures_book_snapshot_25_{date_str}_{pair}.csv.gz",
            "ftick":    f"ftick/binance-futures_derivative_ticker_{date_str}_{pair}.csv.gz",
            "ftrades":  f"ftrades/binance-futures_trades_{date_str}_{pair}.csv.gz",
            "trades":   f"trades/binance_trades_{date_str}_{pair}.csv.gz",
        }
        paths = {k: load_dir / v for k, v in patterns.items()}    

        if not (paths["book"].exists() and paths["fbook"].exists()):
            print(f"  ⚠️ Skipping {date_str}: missing spot or futures book")
            continue

        full_second_index = pd.date_range(
            start=f"{date_str} 00:00:00",
            end=f"{date_str} 23:59:59",
            freq="1s"
        )
        dfs_to_merge = []

        try:
            # Spot
            df = pd.read_csv(paths["book"])
            df.index = parse_timestamp_series(df["timestamp"])
            df = df.sort_index()
            if df.index.duplicated().any():
                df = df[~df.index.duplicated(keep='last')]
            df_feat = process_book_df(df, "spot")
            df_res = df_feat.reindex(full_second_index, method='pad')
            dfs_to_merge.append(df_res)

            # Swap (futures)
            df = pd.read_csv(paths["fbook"])
            df.index = parse_timestamp_series(df["timestamp"])
            df = df.sort_index()
            if df.index.duplicated().any():
                df = df[~df.index.duplicated(keep='last')]
            df_feat = process_book_df(df, "swap")
            df_res = df_feat.reindex(full_second_index, method='pad')
            dfs_to_merge.append(df_res)

            # Ticker (optional)
            if paths["ftick"].exists():
                df = pd.read_csv(paths["ftick"])
                df.index = parse_timestamp_series(df["timestamp"])
                df = df.sort_index()
                if df.index.duplicated().any():
                    df = df[~df.index.duplicated(keep='last')]
                df = df[["index_price", "mark_price", "funding_rate"]]
                df_res = df.reindex(full_second_index, method='pad')
                dfs_to_merge.append(df_res)

            # Trades
            if paths["trades"].exists():
                df = pd.read_csv(paths["trades"])
                df["timestamp"] = parse_timestamp_series(df["timestamp"])
                df_res = process_trades_df(df, "spot")
                dfs_to_merge.append(df_res)

            if paths["ftrades"].exists():
                df = pd.read_csv(paths["ftrades"])
                df["timestamp"] = parse_timestamp_series(df["timestamp"])
                df_res = process_trades_df(df, "swap")
                dfs_to_merge.append(df_res)

            df_day = pd.concat(dfs_to_merge, axis=1)
            df_day.index.name = "timestamp"

            first_valid = df_day["spot_bid0_price"].first_valid_index()
            if first_valid is not None:
                df_day = df_day.loc[first_valid:]
            else:
                print(f"  ⚠️ No valid spot book, skipping {date_str}")
                continue

            valid_dfs.append(df_day)
            valid_dates.append(single_date)
            print(f"  ✅ {date_str} processed ({len(df_day)} seconds)")

        except Exception as e:
            print(f"  ❌ Error on {date_str}: {e}")
            continue

    # ==============================
    # 合并所有有效秒级数据
    # ==============================
    if not valid_dfs:
        print("❌ No valid data processed.")
        exit()

    all_df = pd.concat(valid_dfs, axis=0)
    all_df.index.name = "timestamp"
    print(f"\n📊 Total seconds: {len(all_df)}")

    # ==============================
    # 计算新定义的指标
    # ==============================
    # 确保前三档价格/数量存在（缺失则设为 NaN）
    required_cols = []
    for asset in ['spot', 'swap']:
        for side in ['bid', 'ask']:
            for level in range(3):
                required_cols.append(f"{asset}_{side}{level}_price")
                required_cols.append(f"{asset}_{side}{level}_amount")

    for col in required_cols:
        if col not in all_df.columns:
            all_df[col] = np.nan

    # --- 新定义的 basis1 和 basis2 (log price diff) ---
    all_df['basis1'] = np.log(all_df['swap_bid0_price']) - np.log(all_df['spot_ask0_price'])
    all_df['basis2'] = np.log(all_df['swap_ask0_price']) - np.log(all_df['spot_bid0_price'])

    # --- 新定义的 Volume (swap book imbalance) ---
    # swap_bid_sum3 = all_df[['swap_bid0_amount', 'swap_bid1_amount', 'swap_bid2_amount']].sum(axis=1)
    # swap_ask_sum3 = all_df[['swap_ask0_amount', 'swap_ask1_amount', 'swap_ask2_amount']].sum(axis=1)
    # # 避免除零或 log(negative)
    # swap_bid_sum3 = swap_bid_sum3.replace(0, np.nan)
    # swap_ask_sum3 = swap_ask_sum3.replace(0, np.nan)
    # all_df['Volume'] = np.log(swap_bid_sum3) - np.log(swap_ask_sum3)
    all_df = all_df.rename(columns={"funding_rate": "Volume"}, errors="raise")

    # --- 新定义的 Amount (spot book imbalance) ---
    # spot_bid_sum3 = all_df[['spot_bid0_amount', 'spot_bid1_amount', 'spot_bid2_amount']].sum(axis=1)
    # spot_ask_sum3 = all_df[['spot_ask0_amount', 'spot_ask1_amount', 'spot_ask2_amount']].sum(axis=1)
    # spot_bid_sum3 = spot_bid_sum3.replace(0, np.nan)
    # spot_ask_sum3 = spot_ask_sum3.replace(0, np.nan)
    spot_mid_price = (all_df['spot_bid0_price'] + all_df['spot_ask0_price']) / 2
    all_df['Amount'] = np.log(spot_mid_price) - np.log(all_df['index_price'])

    # ==============================
    # 按 10 分钟重采样，聚合新指标
    # ==============================
    def agg_1min(subdf):
        if subdf.empty:
            return pd.Series(
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                index=['Max', 'Min', 'Open', 'Close', 'Volume', 'Amount']
            )
        
        # Max = basis1 的最大值（按你最初要求，而非分位数）
        Max = subdf['basis1'].quantile(0.99, interpolation='nearest')
        # Min = basis2 的最小值
        Min = subdf['basis2'].quantile(0.01, interpolation='nearest')

        # Open / Close: (basis1 + basis2)/2 的首尾非 NaN 值
        mid_basis = (subdf['basis1'] + subdf['basis2']) / 2
        mid_clean = mid_basis.dropna()
        Open = mid_clean.iloc[0] if len(mid_clean) > 0 else np.nan
        Close = mid_clean.iloc[-1] if len(mid_clean) > 0 else np.nan

        # Volume = 该分钟内 Volume 的均值（按原逻辑）
        Volume = subdf['Volume'].iloc[-1] if len(subdf['Volume']) > 0 else np.nan
        # Amount = 该分钟内 Amount 的均值（注意：现在 Amount 是秒级 imbalance，不是交易量）
        Amount = subdf['Amount'].mean()

        return pd.Series({
            'Max': Max,
            'Min': Min,
            'Open': Open,
            'Close': Close,
            'Volume': Volume,
            'Amount': Amount
        })

    print("\n⏳ Resampling to 1-minute intervals with log-based metrics...")

    # 去重（防万一）
    all_df = all_df.loc[~all_df.index.duplicated(keep='first'), :]

    basis_1min = all_df.resample('1min').apply(agg_1min)
    basis_1min = basis_1min.dropna(how='all')

    # ==============================
    # ✅ 关键：修改 index 名称为 'timestamps'
    # ==============================
    basis_1min.index.name = "timestamps"

    # ==============================
    # 保存结果
    # ==============================
    basis_dir = processed_dir / "basis_1min_task6"
    basis_dir.mkdir(exist_ok=True)

    # 确保 basis_1min.index 是 DatetimeIndex（应已是）
    print("Index type after resample:", type(basis_1min.index))

    grouped_basis = basis_1min.groupby(pd.Grouper(freq='MS'))
    for month_start, month_df in grouped_basis:
        if not month_df.empty:
            year_month = month_start.strftime("%Y-%m")
            out_file = basis_dir / f"{pair}_basis_1min_{year_month}.csv.gz"
            print(f"📈 Saving 1-min basis: {year_month}")
            month_df.to_csv(out_file, compression="gzip")

    print(f"\n🎉 Done! Processed {symbol} {len(valid_dates)} days.")
    print(f" → 1-minute basis files saved in '{basis_dir}'")   