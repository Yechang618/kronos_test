import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from datetime import datetime, timezone, timedelta
import warnings
import argparse

warnings.filterwarnings('ignore')

# ================= 配置区域 =================
TRADE_RECORD_DIR = "./dataset/bn_trade"
MARKET_DATA_ROOT = "D:/market_data"
DEFAULT_OUTPUT_DATASET_PATH = "./dataset/processed_training_set.csv"
DEFAULT_LOOKBACK_SECONDS = 60
# 移动均线需要更长的历史窗口（秒）
MA_10MIN_SECONDS = 600
MA_30MIN_SECONDS = 1800
MA_1H_SECONDS = 3600
# ===========================================

def load_trade_records(base_dir, target_date=None, target_symbol=None):
    """加载所有交易记录 CSV，可选过滤日期和 symbol"""
    dfs = []
    csv_files = glob.glob(os.path.join(base_dir, "combined_*.csv"))
    print(f"Found {len(csv_files)} trade record files.")
    
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            
            if target_date:
                if 'date' in df.columns:
                    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
                    df['date_str'] = df['date_parsed'].dt.strftime('%Y%m%d')
                    df = df[df['date_str'] == target_date]
                    df = df.drop(columns=['date_parsed', 'date_str'], errors='ignore')
            
            if target_symbol and 'symbol' in df.columns:
                df = df[df['symbol'] == target_symbol]
            
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not dfs:
        raise FileNotFoundError("No trade records found after filtering.")

    return pd.concat(dfs, ignore_index=True)

def load_market_data_for_date_symbol(date_str, symbol, market_data_root):
    """根据日期和符号加载市场数据 parquet"""
    input_date_dir = Path(market_data_root) / date_str
    symbol_dir = input_date_dir / symbol
    
    if not symbol_dir.exists():
        return None
        
    parquet_files = [f for f in symbol_dir.glob("*.parquet") if "_inprogress" not in f.name]
    if not parquet_files:
        return None
        
    dfs = []
    for f in sorted(parquet_files):
        try:
            df = pd.read_parquet(f, engine='fastparquet')
            dfs.append(df)
        except Exception as e:
            continue
            
    if not dfs:
        return None
    
    full_df = pd.concat(dfs, ignore_index=True)
    
    if 'time_str' in full_df.columns:
        full_df['time_dt'] = pd.to_datetime(full_df['time_str'], utc=True)
    else:
        return None
        
    return full_df

def calculate_factors(market_df, trade_ts, lookback_seconds):
    """
    基于交易前市场数据计算因子
    新增：换手率、买卖量、移动均线等因子
    """
    trade_dt = pd.to_datetime(trade_ts, unit='ms', utc=True)
    start_dt = trade_dt - timedelta(seconds=lookback_seconds)
    
    # 筛选 1 分钟窗口数据
    window_df = market_df[(market_df['time_dt'] >= start_dt) & 
                          (market_df['time_dt'] < trade_dt)].copy()
    
    # 筛选更长的历史数据用于移动均线计算
    history_10min = market_df[(market_df['time_dt'] >= trade_dt - timedelta(seconds=MA_10MIN_SECONDS)) & 
                              (market_df['time_dt'] < trade_dt)].copy()
    history_30min = market_df[(market_df['time_dt'] >= trade_dt - timedelta(seconds=MA_30MIN_SECONDS)) & 
                              (market_df['time_dt'] < trade_dt)].copy()
    history_1h = market_df[(market_df['time_dt'] >= trade_dt - timedelta(seconds=MA_1H_SECONDS)) & 
                           (market_df['time_dt'] < trade_dt)].copy()
    
    if window_df.empty:
        return None
    
    l5_df = window_df[window_df['stream'].isin(['spot_l5', 'future_l5'])].copy()
    trade_df = window_df[window_df['stream'].isin(['spot_trade', 'future_trade'])].copy()
    
    # 分离买卖方向
    buy_trade_df = trade_df[trade_df.get('side', '').str.contains('buy|BUY|Buy', na=False)].copy() if 'side' in trade_df.columns else trade_df.copy()
    sell_trade_df = trade_df[trade_df.get('side', '').str.contains('sell|SELL|Sell', na=False)].copy() if 'side' in trade_df.columns else pd.DataFrame()
    
    factors = {}
    
    # ==================== 1. 基础价格因子 ====================
    if not l5_df.empty:
        l5_df = l5_df.sort_values('time_dt')
        l5_df.columns = l5_df.columns.str.strip()
        
        if 'bid_px' in l5_df.columns and 'ask_px' in l5_df.columns:
            l5_df['mid_price'] = (l5_df['bid_px'] + l5_df['ask_px']) / 2
            
            open_price = l5_df['mid_price'].iloc[0]
            close_price = l5_df['mid_price'].iloc[-1]
            high_price = l5_df['mid_price'].max()
            low_price = l5_df['mid_price'].min()
            
            factors['factor_price_return'] = (close_price - open_price) / (open_price + 1e-10)
            factors['factor_volatility'] = (high_price - low_price) / (low_price + 1e-10)
            factors['factor_user_custom'] = (close_price - open_price) / (high_price - low_price + 1e-10)
            
            # 订单簿 imbalance
            last_row = l5_df.iloc[-1]
            if 'bid_qty' in last_row and 'ask_qty' in last_row:
                bid_qty = last_row['bid_qty']
                ask_qty = last_row['ask_qty']
                factors['factor_ob_imbalance'] = (bid_qty - ask_qty) / (bid_qty + ask_qty + 1e-10)
            else:
                factors['factor_ob_imbalance'] = 0
        else:
            return None
    else:
        return None
    
    # ==================== 2. 交易流因子（买卖量） ====================
    if not trade_df.empty:
        trade_df = trade_df.sort_values('time_dt')
        trade_df.columns = trade_df.columns.str.strip()
        
        # 总交易强度
        factors['factor_trade_intensity'] = len(trade_df) / lookback_seconds
        
        # 买卖量计算（假设有 qty 列）
        if 'qty' in trade_df.columns:
            total_buy_qty = buy_trade_df['qty'].sum() if not buy_trade_df.empty else 0
            total_sell_qty = sell_trade_df['qty'].sum() if not sell_trade_df.empty else 0
            total_qty = total_buy_qty + total_sell_qty
            
            # 买入卖出量比率
            factors['factor_buy_sell_ratio'] = (total_buy_qty - total_sell_qty) / (total_qty + 1e-10)
            factors['factor_buy_volume'] = total_buy_qty / lookback_seconds
            factors['factor_sell_volume'] = total_sell_qty / lookback_seconds
            factors['factor_total_volume'] = total_qty / lookback_seconds
            
            # 大单检测（假设大于平均成交量的 2 倍为大单）
            avg_trade_size = trade_df['qty'].mean()
            large_trades = trade_df[trade_df['qty'] > 2 * avg_trade_size]
            factors['factor_large_trade_ratio'] = len(large_trades) / (len(trade_df) + 1e-10)
        else:
            # 没有 qty 列时用交易次数估算
            factors['factor_buy_sell_ratio'] = (len(buy_trade_df) - len(sell_trade_df)) / (len(trade_df) + 1e-10)
            factors['factor_buy_volume'] = len(buy_trade_df) / lookback_seconds
            factors['factor_sell_volume'] = len(sell_trade_df) / lookback_seconds
            factors['factor_total_volume'] = len(trade_df) / lookback_seconds
            factors['factor_large_trade_ratio'] = 0
    else:
        factors['factor_trade_intensity'] = 0
        factors['factor_buy_sell_ratio'] = 0
        factors['factor_buy_volume'] = 0
        factors['factor_sell_volume'] = 0
        factors['factor_total_volume'] = 0
        factors['factor_large_trade_ratio'] = 0
    
    # ==================== 3. 移动均线因子 ====================
    if not history_10min.empty and 'mid_price' in l5_df.columns:
        # 10 分钟移动均线
        ma_10min = history_10min[history_10min['stream'].isin(['spot_l5', 'future_l5'])].copy()
        if not ma_10min.empty:
            ma_10min = ma_10min.sort_values('time_dt')
            ma_10min.columns = ma_10min.columns.str.strip()
            if 'bid_px' in ma_10min.columns and 'ask_px' in ma_10min.columns:
                ma_10min['mid_price'] = (ma_10min['bid_px'] + ma_10min['ask_px']) / 2
                factors['factor_ma_10min'] = ma_10min['mid_price'].mean()
                # 当前价格与 10 分钟均线的偏离
                factors['factor_price_ma10m_deviation'] = (close_price - factors['factor_ma_10min']) / (factors['factor_ma_10min'] + 1e-10)
            else:
                factors['factor_ma_10min'] = close_price
                factors['factor_price_ma10m_deviation'] = 0
        else:
            factors['factor_ma_10min'] = close_price
            factors['factor_price_ma10m_deviation'] = 0
    else:
        factors['factor_ma_10min'] = close_price
        factors['factor_price_ma10m_deviation'] = 0
    
    if not history_30min.empty:
        # 30 分钟移动均线
        ma_30min = history_30min[history_30min['stream'].isin(['spot_l5', 'future_l5'])].copy()
        if not ma_30min.empty:
            ma_30min = ma_30min.sort_values('time_dt')
            ma_30min.columns = ma_30min.columns.str.strip()
            if 'bid_px' in ma_30min.columns and 'ask_px' in ma_30min.columns:
                ma_30min['mid_price'] = (ma_30min['bid_px'] + ma_30min['ask_px']) / 2
                factors['factor_ma_30min'] = ma_30min['mid_price'].mean()
                factors['factor_price_ma30m_deviation'] = (close_price - factors['factor_ma_30min']) / (factors['factor_ma_30min'] + 1e-10)
            else:
                factors['factor_ma_30min'] = close_price
                factors['factor_price_ma30m_deviation'] = 0
        else:
            factors['factor_ma_30min'] = close_price
            factors['factor_price_ma30m_deviation'] = 0
    else:
        factors['factor_ma_30min'] = close_price
        factors['factor_price_ma30m_deviation'] = 0
    
    if not history_1h.empty:
        # 1 小时移动均线
        ma_1h = history_1h[history_1h['stream'].isin(['spot_l5', 'future_l5'])].copy()
        if not ma_1h.empty:
            ma_1h = ma_1h.sort_values('time_dt')
            ma_1h.columns = ma_1h.columns.str.strip()
            if 'bid_px' in ma_1h.columns and 'ask_px' in ma_1h.columns:
                ma_1h['mid_price'] = (ma_1h['bid_px'] + ma_1h['ask_px']) / 2
                factors['factor_ma_1h'] = ma_1h['mid_price'].mean()
                factors['factor_price_ma1h_deviation'] = (close_price - factors['factor_ma_1h']) / (factors['factor_ma_1h'] + 1e-10)
            else:
                factors['factor_ma_1h'] = close_price
                factors['factor_price_ma1h_deviation'] = 0
        else:
            factors['factor_ma_1h'] = close_price
            factors['factor_price_ma1h_deviation'] = 0
    else:
        factors['factor_ma_1h'] = close_price
        factors['factor_price_ma1h_deviation'] = 0
    
    # ==================== 4. 换手率相关因子 ====================
    # 估算换手率 = 成交量 / 流通量（这里用平均订单簿深度作为流通量的代理）
    if not l5_df.empty and 'bid_qty' in l5_df.columns:
        # 计算窗口内的平均订单簿深度
        avg_bid_depth = l5_df['bid_qty'].mean()
        avg_ask_depth = l5_df['ask_qty'].mean()
        avg_book_depth = (avg_bid_depth + avg_ask_depth) / 2
        
        # 换手率代理指标
        if 'factor_total_volume' in factors and avg_book_depth > 0:
            factors['factor_turnover_proxy'] = factors['factor_total_volume'] / (avg_book_depth + 1e-10)
        else:
            factors['factor_turnover_proxy'] = 0
        
        # 订单簿深度变化率
        if len(l5_df) > 1:
            first_depth = (l5_df.iloc[0]['bid_qty'] + l5_df.iloc[0]['ask_qty']) / 2
            last_depth = (l5_df.iloc[-1]['bid_qty'] + l5_df.iloc[-1]['ask_qty']) / 2
            factors['factor_book_depth_change'] = (last_depth - first_depth) / (first_depth + 1e-10)
        else:
            factors['factor_book_depth_change'] = 0
    else:
        factors['factor_turnover_proxy'] = 0
        factors['factor_book_depth_change'] = 0
    
    # ==================== 5. 价格动量因子 ====================
    # 短期动量（1 分钟内）
    if len(l5_df) > 1:
        factors['factor_momentum_1m'] = (l5_df['mid_price'].iloc[-1] - l5_df['mid_price'].iloc[0]) / l5_df['mid_price'].iloc[0]
    else:
        factors['factor_momentum_1m'] = 0
    
    # 多周期均线交叉信号
    if 'factor_ma_10min' in factors and 'factor_ma_30min' in factors:
        factors['factor_ma_cross_10m_30m'] = 1 if factors['factor_ma_10min'] > factors['factor_ma_30min'] else -1
    else:
        factors['factor_ma_cross_10m_30m'] = 0
    
    if 'factor_ma_30min' in factors and 'factor_ma_1h' in factors:
        factors['factor_ma_cross_30m_1h'] = 1 if factors['factor_ma_30min'] > factors['factor_ma_1h'] else -1
    else:
        factors['factor_ma_cross_30m_1h'] = 0
    
    return factors

def calculate_labels(row):
    """计算标签：滑点率"""
    try:
        place_price = row.get('taker/swap_place_price', 0)
        exec_price = row.get('taker/swap_executed_price', 0)
        
        if place_price == 0:
            place_price = row.get('maker/spot_anticipated_price', 0)
        if exec_price == 0:
            exec_price = row.get('maker/spot_executed_price', 0)
            
        if place_price == 0:
            return np.nan
            
        slippage = (exec_price - place_price) / place_price
        return slippage
    except:
        return np.nan

def main():
    parser = argparse.ArgumentParser(description='生成交易因子数据集（增强版）')
    parser.add_argument('--date', type=str, default=None, 
                        help='指定日期 (格式：YYYYMMDD)，例如 20260303')
    parser.add_argument('--symbol', type=str, default=None, 
                        help='指定交易对，例如 GPSUSDT')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DATASET_PATH,
                        help='输出文件路径')
    parser.add_argument('--lookback', type=int, default=DEFAULT_LOOKBACK_SECONDS,
                        help='回看窗口秒数')
    parser.add_argument('--market-root', type=str, default=MARKET_DATA_ROOT,
                        help='市场数据根目录')
    parser.add_argument('--trade-dir', type=str, default=TRADE_RECORD_DIR,
                        help='交易记录目录')
    
    args = parser.parse_args()
    
    output_dataset_path = args.output
    lookback_seconds = args.lookback
    market_data_root = args.market_root
    trade_record_dir = args.trade_dir
    
    print("=" * 70)
    print("📊 交易因子数据集生成器（增强版）")
    print("=" * 70)
    print(f"📅 目标日期：{args.date if args.date else '全部日期'}")
    print(f"💹 目标 Symbol: {args.symbol if args.symbol else '全部 Symbol'}")
    print(f"⏱️  回看窗口：{lookback_seconds} 秒")
    print(f"📁 输出路径：{output_dataset_path}")
    print(f"📂 市场数据目录：{market_data_root}")
    print(f"📂 交易记录目录：{trade_record_dir}")
    print("=" * 70)
    print("🔧 新增因子类型:")
    print("   • 换手率代理指标")
    print("   • 买入卖出量比率")
    print("   • 10 分钟/30 分钟/1 小时移动均线")
    print("   • 价格与均线偏离度")
    print("   • 均线交叉信号")
    print("   • 订单簿深度变化率")
    print("=" * 70)
    
    print("Loading trade records...")
    trades_df = load_trade_records(trade_record_dir, 
                                   target_date=args.date, 
                                   target_symbol=args.symbol)
    print(f"Total trades loaded: {len(trades_df)}")
    
    ts_col = 'taker/swap/haircut_executed_ts'
    if ts_col not in trades_df.columns:
        ts_col = 'maker/spot_executed_ts'
        
    trades_df = trades_df[['symbol', ts_col, 'taker/swap_place_price', 'taker/swap_executed_price', 
                           'maker/spot_anticipated_price', 'maker/spot_executed_price', 'trade_mode']].copy()
    trades_df = trades_df.dropna(subset=[ts_col])
    trades_df['trade_ts'] = trades_df[ts_col].astype(float)
    trades_df['label_slippage'] = trades_df.apply(calculate_labels, axis=1)
    trades_df = trades_df.dropna(subset=['label_slippage'])
    
    trades_df['date_str'] = pd.to_datetime(trades_df['trade_ts'], unit='ms').dt.strftime('%Y%m%d')
    
    processed_samples = []
    
    grouped = trades_df.groupby(['date_str', 'symbol'])
    total_groups = len(grouped)
    
    for i, (name, group) in enumerate(grouped):
        date_str, symbol = name
        if i % 50 == 0:
            print(f"Processing group {i}/{total_groups}: {date_str} {symbol}")
            
        market_df = load_market_data_for_date_symbol(date_str, symbol, market_data_root)
        if market_df is None:
            print(f"  ⚠️ No market data for {date_str} {symbol}")
            continue
            
        for _, trade in group.iterrows():
            factors = calculate_factors(market_df, trade['trade_ts'], lookback_seconds)
            if factors:
                sample = {
                    'timestamp': trade['trade_ts'],
                    'symbol': symbol,
                    'date': date_str,
                    'place_price': trade['taker/swap_place_price'] if trade['taker/swap_place_price'] > 0 else trade['maker/spot_anticipated_price'],
                    'exec_price': trade['taker/swap_executed_price'] if trade['taker/swap_executed_price'] > 0 else trade['maker/spot_executed_price'],
                    'label_slippage': trade['label_slippage']
                }
                sample.update(factors)
                processed_samples.append(sample)
                
    if not processed_samples:
        print("❌ No valid samples generated. Check data paths and columns.")
        return
    
    result_df = pd.DataFrame(processed_samples)
    result_df.to_csv(output_dataset_path, index=False)
    print(f"\n✅ Dataset saved to {output_dataset_path} with {len(result_df)} samples.")
    print(f"📋 Total columns: {len(result_df.columns)}")
    print(f"📋 Factor columns: {[c for c in result_df.columns if c.startswith('factor_')]}")
    print("=" * 70)

if __name__ == "__main__":
    main()