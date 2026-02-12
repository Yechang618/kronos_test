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
    for level in range(5):
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
    # start_date = "2025-10-21"
    # end_date = "2025-10-23"
    # start_date = "2025-10-01"
    # end_date = "2025-10-02"    
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
            # print(f"Columns after merge: {df_day.columns.tolist()}")
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
    print("Columns:", all_df.columns.tolist())
    # ==============================
    # 计算新定义的指标
    # ==============================
    # 确保前三档价格/数量存在（缺失则设为 NaN）
    required_cols = []
    for asset in ['spot', 'swap']:
        for side in ['bid', 'ask']:
            for level in range(5):
                required_cols.append(f"{asset}_{side}{level}_price")
                required_cols.append(f"{asset}_{side}{level}_amount")

    for col in required_cols:
        if col not in all_df.columns:
            all_df[col] = np.nan

    # --- 新定义的 basis1 和 basis2 (log price diff) ---
    all_df['basis1'] = np.log(all_df['swap_bid0_price']) - np.log(all_df['spot_ask0_price'])
    all_df['basis2'] = np.log(all_df['swap_ask0_price']) - np.log(all_df['spot_bid0_price'])

    # --- 新定义的 Volume (swap book imbalance) ---
    # all_df = all_df.rename(columns={"funding_rate": "Volume"}, errors="raise")

    # --- 新定义的 Amount (spot book imbalance) ---
    spot_mid_price = (all_df['spot_bid0_price'] + all_df['spot_ask0_price']) / 2
    all_df['spot_index_imbalance'] = np.log(spot_mid_price) - np.log(all_df['index_price'])

    # ==============================
    # 按 10 分钟重采样，聚合新指标
    # ==============================
    def agg_1min(subdf):
        if subdf.empty:
            # print("  ⚠️ Empty subdf, returning NaNs")
            return pd.Series(
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                 np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,  
                 np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                 np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                 np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,  
                 np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                index=['basis_bid', 'basis_ask', 'basis_high', 'basis_low', 'funding_rate', 
                        'index_price', 'spot_index_imbalance', 'mark_price', 
                        'spot_buy_price', 'spot_sell_price', 'spot_buy_amount', 'spot_sell_amount', 
                        'swap_buy_price', 'swap_sell_price', 'swap_buy_amount', 'swap_sell_amount',
                        'spot_bid0_price', 'spot_bid0_amount', 'spot_bid1_price', 'spot_bid1_amount', 
                        'spot_bid2_price', 'spot_bid2_amount', 'spot_bid3_price', 'spot_bid3_amount', 
                        'spot_bid4_price', 'spot_bid4_amount', 
                        'spot_ask0_price', 'spot_ask0_amount', 'spot_ask1_price', 'spot_ask1_amount',
                        'spot_ask2_price', 'spot_ask2_amount', 'spot_ask3_price', 'spot_ask3_amount',
                        'spot_ask4_price', 'spot_ask4_amount',
                        'swap_bid0_price', 'swap_bid0_amount', 'swap_bid1_price', 'swap_bid1_amount', 
                        'swap_bid2_price', 'swap_bid2_amount', 'swap_bid3_price', 'swap_bid3_amount', 
                        'swap_bid4_price', 'swap_bid4_amount', 
                        'swap_ask0_price', 'swap_ask0_amount','swap_ask1_price', 'swap_ask1_amount',
                        'swap_ask2_price', 'swap_ask2_amount','swap_ask3_price', 'swap_ask3_amount',
                        'swap_ask4_price', 'swap_ask4_amount'] # dim = 5 + 3 + 4 * 12 = 56
            )
        
        # Max = basis1 的最大值（按你最初要求，而非分位数）
        Max = subdf['basis1'].quantile(0.95, interpolation='nearest')
        # Min = basis2 的最小值
        Min = subdf['basis2'].quantile(0.05, interpolation='nearest')

        # basis_bid = 该分钟内 basis1 的最后一个非 NaN 值（按原逻辑）
        basis_bid = subdf['basis1'].dropna().iloc[-1] if len(subdf['basis1'].dropna()) > 0 else 0.0
        # basis_ask = 该分钟内 basis2 的最后一个非 NaN 值（按原逻辑）
        basis_ask = subdf['basis2'].dropna().iloc[-1] if len(subdf['basis2'].dropna()) > 0 else 0.0
        # basis_high = 该分钟内 basis1 的 95% 分位数（按你最初要求，而非 Max）
        basis_high = Max
        # basis_low = 该分钟内 basis2 的 5% 分位数（按你最初要求，而非 Min）
        basis_low = Min
        # funding_rate = 该分钟内 funding_rate 的最后一个非 NaN 值（按原逻辑）
        funding_rate = subdf['funding_rate'].dropna().iloc[-1] if len(subdf['funding_rate'].dropna()) > 0 else 0.0
        # index_price = 该分钟内 index_price 的最后一个非 NaN 值（按原逻辑）
        index_price = subdf['index_price'].dropna().iloc[-1] if len(subdf['index_price'].dropna()) > 0 else 0.0
        # spot_index_imbalance = 该分钟内 spot_index_imbalance 的最后一个非 NaN 值（按原逻辑）
        spot_index_imbalance = subdf['spot_index_imbalance'].dropna().iloc[-1] if len(subdf['spot_index_imbalance'].dropna()) > 0 else 0.0
        # mark_price = 该分钟内 mark_price 的最后一个非 NaN 值（按原逻辑）
        mark_price = subdf['mark_price'].dropna().iloc[-1] if len(subdf['mark_price'].dropna()) > 0 else 0.0
        # spot_buy_price = 该分钟内 spot 买价的最后一个非 NaN 值（按原逻辑）
        spot_bid0_price = subdf['spot_bid0_price'].dropna().iloc[-1] if len(subdf['spot_bid0_price'].dropna()) > 0 else 0.0
        # spot_bid0_amount = 该分钟内 spot 买一量的最后一个非 NaN 值（按原逻辑）
        spot_bid0_amount = subdf['spot_bid0_amount'].dropna().iloc[-1] if len(subdf['spot_bid0_amount'].dropna()) > 0 else 0.0
        # spot_ask0_price = 该分钟内 spot 卖一价的最后一个非 NaN 值（按原逻辑）
        spot_ask0_price = subdf['spot_ask0_price'].dropna().iloc[-1] if len(subdf['spot_ask0_price'].dropna()) > 0 else 0.0
        # spot_ask0_amount = 该分钟内 spot 卖一量的最后一个非 NaN 值（按原逻辑）
        spot_ask0_amount = subdf['spot_ask0_amount'].dropna().iloc[-1] if len(subdf['spot_ask0_amount'].dropna()) > 0 else 0.0
        # swap_bid0_price = 该分钟内 swap 买一价的最后一个非 NaN 值（按原逻辑）
        swap_bid0_price = subdf['swap_bid0_price'].dropna().iloc[-1] if len(subdf['swap_bid0_price'].dropna()) > 0 else 0.0
        # swap_bid0_amount = 该分钟内 swap 买一量的最后一个非 NaN 值（按原逻辑）
        swap_bid0_amount = subdf['swap_bid0_amount'].dropna().iloc[-1] if len(subdf['swap_bid0_amount'].dropna()) > 0 else 0.0
        # swap_ask0_price = 该分钟内 swap 卖一价的最后一个非 NaN 值（按原逻辑）
        swap_ask0_price = subdf['swap_ask0_price'].dropna().iloc[-1] if len(subdf['swap_ask0_price'].dropna()) > 0 else 0.0
        # swap_ask0_amount = 该分钟内 swap 卖一量的最后一个非 NaN 值（按原逻辑）
        swap_ask0_amount = subdf['swap_ask0_amount'].dropna().iloc[-1] if len(subdf['swap_ask0_amount'].dropna()) > 0 else 0.0
        # spot_bid1_price = 该分钟内 spot 买二价的最后一个非 NaN 值（按原逻辑）
        spot_bid1_price = subdf['spot_bid1_price'].dropna().iloc[-1] if len(subdf['spot_bid1_price'].dropna()) > 0 else 0.0
        # spot_bid1_amount = 该分钟内 spot 买二量的最后一个非 NaN 值（按原逻辑）
        spot_bid1_amount = subdf['spot_bid1_amount'].dropna().iloc[-1] if len(subdf['spot_bid1_amount'].dropna()) > 0 else 0.0
        # spot_ask1_price = 该分钟内 spot 卖二价的最后一个非 NaN 值（按原逻辑）
        spot_ask1_price = subdf['spot_ask1_price'].dropna().iloc[-1] if len(subdf['spot_ask1_price'].dropna()) > 0 else 0.0
        # spot_ask1_amount = 该分钟内 spot 卖二量的最后一个非 NaN 值（按原逻辑）
        spot_ask1_amount = subdf['spot_ask1_amount'].dropna().iloc[-1] if len(subdf['spot_ask1_amount'].dropna()) > 0 else 0.0
        # spot_bid2_price = 该分钟内 spot 买三价的最后一个非 NaN 值（按原逻辑）
        spot_bid2_price = subdf['spot_bid2_price'].dropna().iloc[-1] if len(subdf['spot_bid2_price'].dropna()) > 0 else 0.0
        # spot_bid2_amount = 该分钟内 spot 买三量的最后一个非 NaN 值（按原逻辑）
        spot_bid2_amount = subdf['spot_bid2_amount'].dropna().iloc[-1] if len(subdf['spot_bid2_amount'].dropna()) > 0 else 0.0
        # spot_ask2_price = 该分钟内 spot 卖三价的最后一个非 NaN 值（按原逻辑）
        spot_ask2_price = subdf['spot_ask2_price'].dropna().iloc[-1] if len(subdf['spot_ask2_price'].dropna()) > 0 else 0.0
        # spot_ask2_amount = 该分钟内 spot 卖三量的最后一个非 NaN 值（按原逻辑）
        spot_ask2_amount = subdf['spot_ask2_amount'].dropna().iloc[-1] if len(subdf['spot_ask2_amount'].dropna()) > 0 else 0.0
        # spot_bid3_price = 该分钟内 spot 买四价的最后一个非 NaN 值（按原逻辑）
        spot_bid3_price = subdf['spot_bid3_price'].dropna().iloc[-1] if len(subdf['spot_bid3_price'].dropna()) > 0 else 0.0
        # spot_bid3_amount = 该分钟内 spot 买四量的最后一个非 NaN 值（按原逻辑）
        spot_bid3_amount = subdf['spot_bid3_amount'].dropna().iloc[-1] if len(subdf['spot_bid3_amount'].dropna()) > 0 else 0.0
        # spot_ask3_price = 该分钟内 spot 卖四价的最后一个非 NaN 值（按原逻辑）
        spot_ask3_price = subdf['spot_ask3_price'].dropna().iloc[-1] if len(subdf['spot_ask3_price'].dropna()) > 0 else 0.0
        # spot_ask3_amount = 该分钟内 spot 卖四量的最后一个非 NaN 值（按原逻辑）
        spot_ask3_amount = subdf['spot_ask3_amount'].dropna().iloc[-1] if len(subdf['spot_ask3_amount'].dropna()) > 0 else 0.0
        # spot_bid4_price = 该分钟内 spot 买五价的最后一个非 NaN 值（按原逻辑）
        spot_bid4_price = subdf['spot_bid4_price'].dropna().iloc[-1] if len(subdf['spot_bid4_price'].dropna()) > 0 else 0.0
        # spot_bid4_amount = 该分钟内 spot 买五量的最后一个非 NaN 值（按原逻辑）
        spot_bid4_amount = subdf['spot_bid4_amount'].dropna().iloc[-1] if len(subdf['spot_bid4_amount'].dropna()) > 0 else 0.0
        # spot_ask4_price = 该分钟内 spot 卖五价的最后一个非 NaN 值（按原逻辑）
        spot_ask4_price = subdf['spot_ask4_price'].dropna().iloc[-1] if len(subdf['spot_ask4_price'].dropna()) > 0 else 0.0
        # spot_ask4_amount = 该分钟内 spot 卖五量的最后一个非 NaN 值（按原逻辑）
        spot_ask4_amount = subdf['spot_ask4_amount'].dropna().iloc[-1] if len(subdf['spot_ask4_amount'].dropna()) > 0 else 0.0
        # swap_bid1_price = 该分钟内 swap 买二价的最后一个非 NaN 值（按原逻辑）
        swap_bid1_price = subdf['swap_bid1_price'].dropna().iloc[-1] if len(subdf['swap_bid1_price'].dropna()) > 0 else 0.0
        # swap_bid1_amount = 该分钟内 swap 买二量的最后一个非 NaN 值（按原逻辑）
        swap_bid1_amount = subdf['swap_bid1_amount'].dropna().iloc[-1] if len(subdf['swap_bid1_amount'].dropna()) > 0 else 0.0
        # swap_ask1_price = 该分钟内 swap 卖二价的最后一个非 NaN 值（按原逻辑）
        swap_ask1_price = subdf['swap_ask1_price'].dropna().iloc[-1] if len(subdf['swap_ask1_price'].dropna()) > 0 else 0.0
        # swap_ask1_amount = 该分钟内 swap 卖二量的最后一个非 NaN 值（按原逻辑）
        swap_ask1_amount = subdf['swap_ask1_amount'].dropna().iloc[-1] if len(subdf['swap_ask1_amount'].dropna()) > 0 else 0.0
        # swap_bid2_price = 该分钟内 swap 买三价的最后一个非 NaN 值（按原逻辑）
        swap_bid2_price = subdf['swap_bid2_price'].dropna().iloc[-1] if len(subdf['swap_bid2_price'].dropna()) > 0 else 0.0
        # swap_bid2_amount = 该分钟内 swap 买三量的最后一个非 NaN 值（按原逻辑）
        swap_bid2_amount = subdf['swap_bid2_amount'].dropna().iloc[-1] if len(subdf['swap_bid2_amount'].dropna()) > 0 else 0.0
        # swap_ask2_price = 该分钟内 swap 卖三价的最后一个非 NaN 值（按原逻辑）
        swap_ask2_price = subdf['swap_ask2_price'].dropna().iloc[-1] if len(subdf['swap_ask2_price'].dropna()) > 0 else 0.0
        # swap_ask2_amount = 该分钟内 swap 卖三量的最后一个非 NaN 值（按原逻辑）
        swap_ask2_amount = subdf['swap_ask2_amount'].dropna().iloc[-1] if len(subdf['swap_ask2_amount'].dropna()) > 0 else 0.0
        # swap_bid3_price = 该分钟内 swap 买四价的最后一个非 NaN 值（按原逻辑）
        swap_bid3_price = subdf['swap_bid3_price'].dropna().iloc[-1] if len(subdf['swap_bid3_price'].dropna()) > 0 else 0.0
        # swap_bid3_amount = 该分钟内 swap 买四量的最后一个非 NaN 值（按原逻辑）
        swap_bid3_amount = subdf['swap_bid3_amount'].dropna().iloc[-1] if len(subdf['swap_bid3_amount'].dropna()) > 0 else 0.0
        # swap_ask3_price = 该分钟内 swap 卖四价的最后一个非 NaN 值（按原逻辑）
        swap_ask3_price = subdf['swap_ask3_price'].dropna().iloc[-1] if len(subdf['swap_ask3_price'].dropna()) > 0 else 0.0
        # swap_ask3_amount = 该分钟内 swap 卖四量的最后一个非 NaN 值（按原逻辑）
        swap_ask3_amount = subdf['swap_ask3_amount'].dropna().iloc[-1] if len(subdf['swap_ask3_amount'].dropna()) > 0 else 0.0
        # swap_bid4_price = 该分钟内 swap 买五价的最后一个非 NaN 值（按原逻辑）
        swap_bid4_price = subdf['swap_bid4_price'].dropna().iloc[-1] if len(subdf['swap_bid4_price'].dropna()) > 0 else 0.0
        # swap_bid4_amount = 该分钟内 swap 买五量的最后一个非 NaN 值（按原逻辑）
        swap_bid4_amount = subdf['swap_bid4_amount'].dropna().iloc[-1] if len(subdf['swap_bid4_amount'].dropna()) > 0 else 0.0
        # swap_ask4_price = 该分钟内 swap 卖五价的最后一个非 NaN 值（按原逻辑）
        swap_ask4_price = subdf['swap_ask4_price'].dropna().iloc[-1] if len(subdf['swap_ask4_price'].dropna()) > 0 else 0.0
        # swap_ask4_amount = 该分钟内 swap 卖五量的最后一个非 NaN 值（按原逻辑）
        swap_ask4_amount = subdf['swap_ask4_amount'].dropna().iloc[-1] if len(subdf['swap_ask4_amount'].dropna()) > 0 else 0.0
        # spot_buy_price = 该分钟内 spot 买单价格的加权平均（按原逻辑）
        spot_buy_price = (subdf['spot_bid0_price'] * subdf['spot_bid0_amount']).sum() / subdf['spot_bid0_amount'].sum() if subdf['spot_bid0_amount'].sum() > 0 else spot_ask0_price
        # spot_sell_price = 该分钟内 spot 卖单价格的加权平均（按原逻辑）
        spot_sell_price = (subdf['spot_ask0_price'] * subdf['spot_ask0_amount']).sum() / subdf['spot_ask0_amount'].sum() if subdf['spot_ask0_amount'].sum() > 0 else spot_bid0_price
        # spot_buy_amount = 该分钟内 spot 买单数量的总和（按原逻辑）
        spot_buy_amount = subdf['spot_bid0_amount'].sum()        # spot_sell_amount = 该分钟内 spot 卖单数量的总和（按原逻辑）
        spot_sell_amount = subdf['spot_ask0_amount'].sum()       # swap_buy_price = 该分钟内 swap 买单价格的加权平均（按原逻辑）
        swap_buy_price = (subdf['swap_bid0_price'] * subdf['swap_bid0_amount']).sum() / subdf['swap_bid0_amount'].sum() if subdf['swap_bid0_amount'].sum() > 0 else swap_ask0_price
        # swap_sell_price = 该分钟内 swap 卖单价格的加权平均（按原逻辑）
        swap_sell_price = (subdf['swap_ask0_price'] * subdf['swap_ask0_amount']).sum() / subdf['swap_ask0_amount'].sum() if subdf['swap_ask0_amount'].sum() > 0 else swap_bid0_price
        # swap_buy_amount = 该分钟内 swap 买单数量的总和（按原逻辑）
        swap_buy_amount = subdf['swap_bid0_amount'].sum()       # swap_sell_amount = 该分钟内 swap 卖单数量的总和（按原逻辑）
        swap_sell_amount = subdf['swap_ask0_amount'].sum()      # spot_bid0_price = 该分钟内 spot 买一价的最后一个非 NaN 值（按原逻辑）

        return pd.Series({
            'basis_bid': basis_bid, 
            'basis_ask': basis_ask, 
            'basis_high': basis_high, 
            'basis_low': basis_low,
            'funding_rate': funding_rate,
            'index_price': index_price,
            'spot_index_imbalance': spot_index_imbalance,
            'mark_price': mark_price,
            'spot_buy_price': spot_buy_price,
            'spot_sell_price': spot_sell_price,
            'spot_buy_amount': spot_buy_amount,
            'spot_sell_amount': spot_sell_amount,
            'swap_buy_price': swap_buy_price,
            'swap_sell_price': swap_sell_price,
            'swap_buy_amount': swap_buy_amount,
            'swap_sell_amount': swap_sell_amount,
            'spot_bid0_price': spot_bid0_price,
            'spot_bid0_amount': spot_bid0_amount,
            'spot_bid1_price': spot_bid1_price,
            'spot_bid1_amount': spot_bid1_amount,
            'spot_bid2_price': spot_bid2_price,
            'spot_bid2_amount': spot_bid2_amount,
            'spot_bid3_price': spot_bid3_price,
            'spot_bid3_amount': spot_bid3_amount,
            'spot_bid4_price': spot_bid4_price,
            'spot_bid4_amount': spot_bid4_amount,
            'spot_ask0_price': spot_ask0_price,
            'spot_ask0_amount': spot_ask0_amount,
            'spot_ask1_price': spot_ask1_price,
            'spot_ask1_amount': spot_ask1_amount,
            'spot_ask2_price': spot_ask2_price,
            'spot_ask2_amount': spot_ask2_amount,
            'spot_ask3_price': spot_ask3_price,
            'spot_ask3_amount': spot_ask3_amount,
            'spot_ask4_price': spot_ask4_price,
            'spot_ask4_amount': spot_ask4_amount,
            'swap_bid0_price': swap_bid0_price,
            'swap_bid0_amount': swap_bid0_amount,
            'swap_bid1_price': swap_bid1_price,
            'swap_bid1_amount': swap_bid1_amount,
            'swap_bid2_price': swap_bid2_price,
            'swap_bid2_amount': swap_bid2_amount,
            'swap_bid3_price': swap_bid3_price,
            'swap_bid3_amount': swap_bid3_amount,
            'swap_bid4_price': swap_bid4_price,
            'swap_bid4_amount': swap_bid4_amount,
            'swap_ask0_price': swap_ask0_price,
            'swap_ask0_amount': swap_ask0_amount,
            'swap_ask1_price': swap_ask1_price,
            'swap_ask1_amount': swap_ask1_amount,
            'swap_ask2_price': swap_ask2_price,
            'swap_ask2_amount': swap_ask2_amount,
            'swap_ask3_price': swap_ask3_price,
            'swap_ask3_amount': swap_ask3_amount,
            'swap_ask4_price': swap_ask4_price,
            'swap_ask4_amount': swap_ask4_amount
        })

    print("\n⏳ Resampling to 10-minute intervals with log-based metrics...")

    # 去重（防万一）
    # all_df = all_df.loc[~all_df.index.duplicated(keep='first'), :]
    basis_1min = all_df.resample('1min').apply(agg_1min)
    basis_1min = basis_1min.dropna(how='all')
    print(f"Resampled 1-minute basis shape: {basis_1min.shape}")

    # ==============================
    # ✅ 关键：修改 index 名称为 'timestamps'
    # ==============================
    basis_1min.index.name = "timestamps"

    # ==============================
    # 保存结果
    # ==============================
    basis_dir = processed_dir / "basis_1min_task8"
    basis_dir.mkdir(exist_ok=True)

    # CRITICAL VALIDATION
    print(f"Resampled shape: {basis_1min.shape}")
    print(f"Is Series? {isinstance(basis_1min, pd.Series)}")
    print(f"Index type: {type(basis_1min.index)}")
    print(f"Index sample: {basis_1min.index[:3]}")

    print(f"basis_1min: {basis_1min.head(10)}")
    # Safety check before grouping
    # if not isinstance(basis_1min.index, pd.DatetimeIndex):
    #     print(basis_1min.index)
    #     raise ValueError(f"Index is not DatetimeIndex! Got: {type(basis_1min.index)}")    
    if not isinstance(basis_1min.index, pd.DatetimeIndex):
        basis_1min.index = pd.to_datetime(basis_1min.index)

    grouped_basis = basis_1min.groupby(pd.Grouper(freq='MS'))
    for month_start, month_df in grouped_basis:
        if not month_df.empty:
            year_month = month_start.strftime("%Y-%m")
            out_file = basis_dir / f"{pair}_basis_1min_{year_month}.csv.gz"
            print(f"📈 Saving 1-min basis: {year_month}")
            month_df.to_csv(out_file, compression="gzip")

    print(f"\n🎉 Done! Processed {symbol} {len(valid_dates)} days.")
    print(f" → 1-minute basis files saved in '{basis_dir}'")   