#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高频因子提取系统 v3（修复版）
修复：100ms 重采样 + 基差计算 + 所有因子标量化
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime, timedelta
import warnings
import argparse
warnings.filterwarnings('ignore')

# ============================
# 配置区域
# ============================
class Config:
    MARKET_DATA_ROOT = Path("D:/market_data")
    OUTPUT_DIR = Path("./dataset/factors/hf_factors_1min")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    START_DATE = "20260101"
    END_DATE = "20260131"
    
    LOOKBACK_SECONDS = 60
    MA_10MIN_SECONDS = 600
    MA_30MIN_SECONDS = 1800
    MA_1H_SECONDS = 3600
    
    # 🔧 修复：重采样频率改为 100ms
    RESAMPLE_FREQUENCY = '100ms'
    
    MIN_DATA_POINTS = 100
    OUTLIER_STD_THRESHOLD = 5

config = Config()

# ============================
# A 类基础量定义
# ============================
class ClassA_Features:
    @staticmethod
    def extract(df: pd.DataFrame) -> dict:
        features = {}
        spot_l5 = df[df['stream'] == 'spot_l5'].copy() if 'stream' in df.columns else df.copy()
        swap_l5 = df[df['stream'] == 'future_l5'].copy() if 'stream' in df.columns else pd.DataFrame()
        
        if not spot_l5.empty and 'bid_px' in spot_l5.columns and 'ask_px' in spot_l5.columns:
            features['spot_bid0_px'] = spot_l5['bid_px'].values
            features['spot_ask0_px'] = spot_l5['ask_px'].values
        if not swap_l5.empty and 'bid_px' in swap_l5.columns and 'ask_px' in swap_l5.columns:
            features['swap_bid0_px'] = swap_l5['bid_px'].values
            features['swap_ask0_px'] = swap_l5['ask_px'].values
            
        return features

# ============================
# B 类基础量定义
# ============================
class ClassB_Features:
    @staticmethod
    def extract(df: pd.DataFrame, class_a: dict) -> dict:
        features = {}
        
        # 🔧 修复：确保使用数组长度对齐
        n_samples = len(df)
        
        # 4 种基差定义
        try:
            if all(k in class_a for k in ['swap_bid0_px', 'spot_ask0_px']):
                swap_bid = class_a['swap_bid0_px']
                spot_ask = class_a['spot_ask0_px']
                if len(swap_bid) == n_samples and len(spot_ask) == n_samples:
                    features['basis_ba'] = np.log(swap_bid + 1e-10) - np.log(spot_ask + 1e-10)
            if all(k in class_a for k in ['swap_ask0_px', 'spot_bid0_px']):
                swap_ask = class_a['swap_ask0_px']
                spot_bid = class_a['spot_bid0_px']
                if len(swap_ask) == n_samples and len(spot_bid) == n_samples:
                    features['basis_ab'] = np.log(swap_ask + 1e-10) - np.log(spot_bid + 1e-10)
            if all(k in class_a for k in ['swap_bid0_px', 'spot_bid0_px']):
                swap_bid = class_a['swap_bid0_px']
                spot_bid = class_a['spot_bid0_px']
                if len(swap_bid) == n_samples and len(spot_bid) == n_samples:
                    features['basis_bb'] = np.log(swap_bid + 1e-10) - np.log(spot_bid + 1e-10)
            if all(k in class_a for k in ['swap_ask0_px', 'spot_ask0_px']):
                swap_ask = class_a['swap_ask0_px']
                spot_ask = class_a['spot_ask0_px']
                if len(swap_ask) == n_samples and len(spot_ask) == n_samples:
                    features['basis_aa'] = np.log(swap_ask + 1e-10) - np.log(spot_ask + 1e-10)
        except Exception as e:
            print(f"    ⚠️ 基差计算失败：{e}")
            
        # 4 种买卖一量
        spot_l5 = df[df['stream'] == 'spot_l5'].copy() if 'stream' in df.columns else df.copy()
        swap_l5 = df[df['stream'] == 'future_l5'].copy() if 'stream' in df.columns else pd.DataFrame()
        
        if not spot_l5.empty and 'bid_qty' in spot_l5.columns:
            features['spot_bid0_amt'] = spot_l5['bid_qty'].values
        if not spot_l5.empty and 'ask_qty' in spot_l5.columns:
            features['spot_ask0_amt'] = spot_l5['ask_qty'].values
        if not swap_l5.empty and 'bid_qty' in swap_l5.columns:
            features['swap_bid0_amt'] = swap_l5['bid_qty'].values
        if not swap_l5.empty and 'ask_qty' in swap_l5.columns:
            features['swap_ask0_amt'] = swap_l5['ask_qty'].values
            
        return features

    @staticmethod
    def calculate_1min_factors(window_df: pd.DataFrame, feature_name: str) -> dict:
        factors = {}
        if feature_name not in window_df.columns:
            return factors
            
        series = window_df[feature_name].dropna()
        if len(series) < 2:
            return factors
            
        start_val = series.iloc[0]
        end_val = series.iloc[-1]
        high_val = series.max()
        low_val = series.min()
        
        factors[f'{feature_name}_return_1m'] = float(np.log(end_val + 1e-10) - np.log(start_val + 1e-10))
        factors[f'{feature_name}_volatility_1m'] = float(np.log(high_val + 1e-10) - np.log(low_val + 1e-10))
        
        price_range = high_val - low_val
        if price_range > 1e-10:
            factors[f'{feature_name}_candle_body_1m'] = float((end_val - start_val) / price_range)
        else:
            factors[f'{feature_name}_candle_body_1m'] = 0.0
            
        return factors

# ============================
# 1. 基础价格与波动因子
# ============================
class PriceFactors:
    @staticmethod
    def calculate(window_df: pd.DataFrame, lookback_seconds: int) -> dict:
        factors = {}
        if window_df.empty:
            return factors
            
        if all(col in window_df.columns for col in ['spot_bid0_px', 'spot_ask0_px']):
            window_df = window_df.copy()
            window_df['mid_price'] = (window_df['spot_bid0_px'] + window_df['spot_ask0_px']) / 2
            
            if len(window_df) > 1:
                open_price = float(window_df['mid_price'].iloc[0])
                close_price = float(window_df['mid_price'].iloc[-1])
                high_price = float(window_df['mid_price'].max())
                low_price = float(window_df['mid_price'].min())
                
                factors['factor_price_return'] = float((close_price - open_price) / (open_price + 1e-10))
                factors['factor_volatility'] = float((high_price - low_price) / (low_price + 1e-10))
                
                price_range = high_price - low_price
                if price_range > 1e-10:
                    factors['factor_user_custom'] = float((close_price - open_price) / price_range)
                else:
                    factors['factor_user_custom'] = 0.0
                    
                factors['factor_momentum_1m'] = float((close_price - open_price) / (open_price + 1e-10))
                
        return factors

# ============================
# 2. 订单簿 (Order Book) 因子
# ============================
class OrderBookFactors:
    @staticmethod
    def calculate(window_df: pd.DataFrame, trade_df: pd.DataFrame, lookback_seconds: int) -> dict:
        factors = {}
        if window_df.empty:
            return factors
            
        window_df = window_df.copy()
        
        if all(col in window_df.columns for col in ['spot_bid0_amt', 'spot_ask0_amt']):
            last_row = window_df.iloc[-1]
            bid_qty = float(last_row['spot_bid0_amt'])
            ask_qty = float(last_row['spot_ask0_amt'])
            factors['factor_ob_imbalance'] = float((bid_qty - ask_qty) / (bid_qty + ask_qty + 1e-10))
            
        if len(window_df) > 1 and all(col in window_df.columns for col in ['spot_bid0_amt', 'spot_ask0_amt']):
            first_depth = (window_df.iloc[0]['spot_bid0_amt'] + window_df.iloc[0]['spot_ask0_amt']) / 2
            last_depth = (window_df.iloc[-1]['spot_bid0_amt'] + window_df.iloc[-1]['spot_ask0_amt']) / 2
            if first_depth > 1e-10:
                factors['factor_book_depth_change'] = float((last_depth - first_depth) / first_depth)
            else:
                factors['factor_book_depth_change'] = 0.0
                
        if not trade_df.empty and 'qty' in trade_df.columns:
            total_volume = float(trade_df['qty'].sum())
            avg_bid_depth = float(window_df['spot_bid0_amt'].mean())
            avg_ask_depth = float(window_df['spot_ask0_amt'].mean())
            avg_book_depth = (avg_bid_depth + avg_ask_depth) / 2
            if avg_book_depth > 1e-10:
                factors['factor_turnover_proxy'] = float(total_volume / avg_book_depth)
            else:
                factors['factor_turnover_proxy'] = 0.0
                
        return factors

# ============================
# 3. 交易流 (Trade Flow) 因子
# ============================
class TradeFlowFactors:
    @staticmethod
    def calculate(trade_df: pd.DataFrame, lookback_seconds: int) -> dict:
        factors = {}
        if trade_df.empty:
            factors['factor_trade_intensity'] = 0.0
            factors['factor_buy_sell_ratio'] = 0.0
            factors['factor_buy_volume'] = 0.0
            factors['factor_sell_volume'] = 0.0
            factors['factor_total_volume'] = 0.0
            factors['factor_large_trade_ratio'] = 0.0
            return factors
            
        factors['factor_trade_intensity'] = float(len(trade_df) / lookback_seconds)
        
        if 'side' in trade_df.columns:
            buy_df = trade_df[trade_df['side'].str.contains('buy|BUY|Buy', na=False)]
            sell_df = trade_df[trade_df['side'].str.contains('sell|SELL|Sell', na=False)]
        else:
            buy_df = trade_df.copy()
            sell_df = pd.DataFrame()
            
        if 'qty' in trade_df.columns:
            total_buy_qty = float(buy_df['qty'].sum()) if not buy_df.empty else 0.0
            total_sell_qty = float(sell_df['qty'].sum()) if not sell_df.empty else 0.0
            total_qty = total_buy_qty + total_sell_qty
            
            factors['factor_buy_sell_ratio'] = float((total_buy_qty - total_sell_qty) / (total_qty + 1e-10))
            factors['factor_buy_volume'] = float(total_buy_qty / lookback_seconds)
            factors['factor_sell_volume'] = float(total_sell_qty / lookback_seconds)
            factors['factor_total_volume'] = float(total_qty / lookback_seconds)
            
            avg_trade_size = float(trade_df['qty'].mean())
            large_trades = trade_df[trade_df['qty'] > 2 * avg_trade_size]
            factors['factor_large_trade_ratio'] = float(len(large_trades) / (len(trade_df) + 1e-10))
        else:
            factors['factor_buy_sell_ratio'] = float((len(buy_df) - len(sell_df)) / (len(trade_df) + 1e-10))
            factors['factor_buy_volume'] = float(len(buy_df) / lookback_seconds)
            factors['factor_sell_volume'] = float(len(sell_df) / lookback_seconds)
            factors['factor_total_volume'] = float(len(trade_df) / lookback_seconds)
            factors['factor_large_trade_ratio'] = 0.0
            
        return factors

# ============================
# 4. 移动均线 (Moving Average) 因子
# ============================
class MovingAverageFactors:
    @staticmethod
    def calculate(history_dfs: dict, current_price: float) -> dict:
        factors = {}
        periods = {'10min': 600, '30min': 1800, '1h': 3600}
        
        for name, seconds in periods.items():
            ma_col = f'factor_ma_{name}'
            dev_col = f'factor_price_ma{name}_deviation'
            history_df = history_dfs.get(name, pd.DataFrame())
            
            if history_df.empty:
                factors[ma_col] = float(current_price)
                factors[dev_col] = 0.0
                continue
                
            if all(col in history_df.columns for col in ['spot_bid0_px', 'spot_ask0_px']):
                mid_prices = (history_df['spot_bid0_px'] + history_df['spot_ask0_px']) / 2
                # 🔧 修复：确保返回标量
                ma_value = float(mid_prices.mean())
                factors[ma_col] = ma_value
                factors[dev_col] = float((current_price - ma_value) / (ma_value + 1e-10))
            else:
                factors[ma_col] = float(current_price)
                factors[dev_col] = 0.0
                
        return factors

# ============================
# 5. 均线交叉信号因子
# ============================
class MACrossFactors:
    @staticmethod
    def calculate(factors: dict) -> dict:
        cross_factors = {}
        
        if 'factor_ma_10min' in factors and 'factor_ma_30min' in factors:
            ma_10_val = float(factors['factor_ma_10min'])
            ma_30_val = float(factors['factor_ma_30min'])
            cross_factors['factor_ma_cross_10m_30m'] = 1 if ma_10_val > ma_30_val else 0
            
        if 'factor_ma_30min' in factors and 'factor_ma_1h' in factors:
            ma_30_val = float(factors['factor_ma_30min'])
            ma_1h_val = float(factors['factor_ma_1h'])
            cross_factors['factor_ma_cross_30m_1h'] = 1 if ma_30_val > ma_1h_val else 0
            
        return cross_factors

# ============================
# 高频因子计算引擎
# ============================
class HighFrequencyFactorEngine:
    def __init__(self, df: pd.DataFrame, lookback_seconds: int = 60):
        self.df = df.copy()
        self.lookback_seconds = lookback_seconds
        self.class_a = {}
        self.class_b = {}
        self.factors = {}
        
    def compute_all_factors(self, trade_timestamp: pd.Timestamp):
        print("  🔧 计算高频因子...")
        
        # 1. 提取 A 类基础量
        self.class_a = ClassA_Features.extract(self.df)
        
        # 2. 提取 B 类基础量
        if self.class_a:
            self.class_b = ClassB_Features.extract(self.df, self.class_a)
        
        # 3. 准备时间窗口
        window_df, trade_df, history_dfs = self._prepare_windows(trade_timestamp)
        
        # 🔧 修复：将 B 类基础量添加到 window_df 中（作为列）
        for key, value in self.class_b.items():
            if isinstance(value, np.ndarray) and len(value) == len(window_df):
                window_df[key] = value
            elif isinstance(value, list) and len(value) == len(window_df):
                window_df[key] = value
        
        # 4. 计算 B 类 1 分钟因子
        self._calculate_class_b_1min_factors(window_df)
        
        # 5. 计算 5 大类因子
        current_price = 0.0
        if 'spot_bid0_px' in self.class_a and 'spot_ask0_px' in self.class_a:
            bid_px = self.class_a['spot_bid0_px']
            ask_px = self.class_a['spot_ask0_px']
            if len(bid_px) > 0 and len(ask_px) > 0:
                current_price = float((bid_px[-1] + ask_px[-1]) / 2)
        
        # 5.1 基础价格因子
        price_factors = PriceFactors.calculate(window_df, self.lookback_seconds)
        self.factors.update(price_factors)
        
        # 5.2 订单簿因子
        ob_factors = OrderBookFactors.calculate(window_df, trade_df, self.lookback_seconds)
        self.factors.update(ob_factors)
        
        # 5.3 交易流因子
        trade_factors = TradeFlowFactors.calculate(trade_df, self.lookback_seconds)
        self.factors.update(trade_factors)
        
        # 5.4 移动均线因子
        ma_factors = MovingAverageFactors.calculate(history_dfs, current_price)
        self.factors.update(ma_factors)
        
        # 5.5 均线交叉信号
        cross_factors = MACrossFactors.calculate(self.factors)
        self.factors.update(cross_factors)
        
        self._clean_factors()
        print(f"  ✅ 完成 {len(self.factors)} 个因子计算")
        return self.factors
        
    def _calculate_class_b_1min_factors(self, window_df: pd.DataFrame):
        b_class_features = ['basis_ba', 'basis_ab', 'basis_bb', 'basis_aa',
                            'spot_bid0_amt', 'spot_ask0_amt', 'swap_bid0_amt', 'swap_ask0_amt']
        for feature in b_class_features:
            if feature in window_df.columns:
                b_factors = ClassB_Features.calculate_1min_factors(window_df, feature)
                self.factors.update(b_factors)
                
    def _prepare_windows(self, trade_timestamp: pd.Timestamp):
        if 'time_dt' not in self.df.columns:
            return self.df.tail(100), pd.DataFrame(), {'10min': pd.DataFrame(), '30min': pd.DataFrame(), '1h': pd.DataFrame()}
            
        self.df = self.df.sort_values('time_dt')
        
        window_start = trade_timestamp - timedelta(seconds=self.lookback_seconds)
        window_df = self.df[(self.df['time_dt'] >= window_start) & 
                            (self.df['time_dt'] < trade_timestamp)].copy()
        
        trade_df = window_df[window_df.get('stream', '') == 'spot_trade'].copy()
        
        history_dfs = {}
        history_dfs['10min'] = self.df[(self.df['time_dt'] >= trade_timestamp - timedelta(seconds=600)) & 
                                       (self.df['time_dt'] < trade_timestamp)].copy()
        history_dfs['30min'] = self.df[(self.df['time_dt'] >= trade_timestamp - timedelta(seconds=1800)) & 
                                       (self.df['time_dt'] < trade_timestamp)].copy()
        history_dfs['1h'] = self.df[(self.df['time_dt'] >= trade_timestamp - timedelta(seconds=3600)) & 
                                    (self.df['time_dt'] < trade_timestamp)].copy()
                                    
        return window_df, trade_df, history_dfs
        
    def _clean_factors(self):
        for key, value in self.factors.items():
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    self.factors[key] = 0.0
                elif abs(value) > config.OUTLIER_STD_THRESHOLD * 10:
                    self.factors[key] = float(np.sign(value) * config.OUTLIER_STD_THRESHOLD * 10)
                    
    def get_factors(self) -> dict:
        return self.factors.copy()

# ============================
# 数据加载
# ============================
def load_market_data_for_date_symbol(date_str: str, symbol: str, market_data_root: Path) -> pd.DataFrame:
    input_date_dir = market_data_root / date_str
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

def resample_data(df: pd.DataFrame, freq: str = '100ms') -> pd.DataFrame:
    """🔧 修复：重采样到 100ms"""
    if df.empty:
        return df
        
    df = df.set_index('time_dt')
    
    if 'stream' in df.columns:
        l5_df = df[df['stream'].isin(['spot_l5', 'future_l5'])].copy()
        trade_df = df[df['stream'].isin(['spot_trade', 'future_trade'])].copy()
    else:
        l5_df = df.copy()
        trade_df = pd.DataFrame()
        
    price_cols = [c for c in l5_df.columns if 'px' in c or 'price' in c]
    qty_cols = [c for c in l5_df.columns if 'qty' in c or 'amt' in c]
    
    resampled_l5 = pd.DataFrame()
    if price_cols:
        resampled_l5[price_cols] = l5_df[price_cols].resample(freq).last()
    if qty_cols:
        resampled_l5[qty_cols] = l5_df[qty_cols].resample(freq).last()
    if 'stream' in l5_df.columns:
        resampled_l5['stream'] = l5_df['stream'].resample(freq).last()
        
    resampled_l5 = resampled_l5.ffill().bfill().dropna()
    
    if not trade_df.empty:
        resampled = pd.concat([resampled_l5, trade_df], ignore_index=False).sort_index()
    else:
        resampled = resampled_l5
        
    return resampled.reset_index()

# ============================
# 主处理流程
# ============================
def process_symbol(symbol: str, date_str: str, config: Config) -> dict:
    print(f"\n{'='*60}")
    print(f"📊 处理：{symbol} @ {date_str}")
    print(f"{'='*60}")
    
    market_df = load_market_data_for_date_symbol(date_str, symbol, config.MARKET_DATA_ROOT)
    if market_df is None or market_df.empty:
        print(f"  ❌ 无数据")
        return {'status': 'failed', 'reason': 'no_data'}
        
    print(f"  📥 原始数据：{len(market_df):,} 条")
    
    # 🔧 修复：使用 100ms 重采样
    market_df = resample_data(market_df, config.RESAMPLE_FREQUENCY)
    print(f"  🔄 重采样后：{len(market_df):,} 条 ({config.RESAMPLE_FREQUENCY})")
    
    if len(market_df) < config.MIN_DATA_POINTS:
        print(f"  ⚠️ 数据点不足")
        return {'status': 'failed', 'reason': 'insufficient_data'}
        
    all_factors = []
    sample_interval = max(1, len(market_df) // 100)
    
    for i in range(config.LOOKBACK_SECONDS * 10, len(market_df), sample_interval):
        trade_ts = market_df.iloc[i]['time_dt']
        window_df = market_df.iloc[max(0, i - config.LOOKBACK_SECONDS * 10):i].copy()
        
        engine = HighFrequencyFactorEngine(window_df, config.LOOKBACK_SECONDS)
        factors = engine.compute_all_factors(trade_ts)
        
        if factors:
            sample = {
                'timestamp': trade_ts.value // 10**6,
                'symbol': symbol,
                'date': date_str,
            }
            sample.update(factors)
            all_factors.append(sample)
            
    if not all_factors:
        print(f"  ❌ 无有效样本")
        return {'status': 'failed', 'reason': 'no_samples'}
        
    result_df = pd.DataFrame(all_factors)
    symbol_output_dir = config.OUTPUT_DIR / symbol
    symbol_output_dir.mkdir(parents=True, exist_ok=True)
    output_file = symbol_output_dir / f"{symbol}_{date_str}_factors.csv.gz"
    result_df.to_csv(output_file, index=False, compression='gzip')
    
    # 🔧 修复：准确统计因子数
    factor_cols = [c for c in result_df.columns if c not in ['timestamp', 'symbol', 'date']]
    print(f"  💾 保存：{output_file}")
    print(f"  ✅ 样本数：{len(result_df):,}")
    print(f"  ✅ 因子数：{len(factor_cols)}")
    print(f"  📋 因子列表：{factor_cols[:10]}...")
    
    return {
        'symbol': symbol,
        'date': date_str,
        'n_samples': len(result_df),
        'n_factors': len(result_df.columns),
        'status': 'success'
    }

def discover_symbols(config: Config) -> list:
    sample_date = config.START_DATE
    sample_path = config.MARKET_DATA_ROOT / sample_date
    if not sample_path.exists():
        print(f"⚠️ 样本路径不存在：{sample_path}")
        return []
    symbols = [d.name for d in sample_path.iterdir() if d.is_dir()]
    print(f"🔍 发现 {len(symbols)} 个交易对")
    return symbols

def generate_summary(summaries: list, config: Config):
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(config.OUTPUT_DIR / "extraction_summary.csv", index=False)
    if not summary_df.empty:
        success = summary_df[summary_df['status'] == 'success']
        print(f"\n📋 汇总：{len(success)}/{len(summary_df)} 成功")
        if len(success) > 0:
            print(f"   总样本数：{success['n_samples'].sum():,}")
            print(f"   平均因子数：{success['n_factors'].mean():.1f}")

# ============================
# 主程序入口
# ============================
def main():
    parser = argparse.ArgumentParser(description='高频因子提取系统 v3（修复版）')
    parser.add_argument('--date', type=str, default=None, help='指定日期 (YYYYMMDD)')
    parser.add_argument('--symbol', type=str, default=None, help='指定交易对')
    parser.add_argument('--lookback', type=int, default=60, help='回看窗口秒数')
    parser.add_argument('--market-root', type=str, default=str(config.MARKET_DATA_ROOT), help='市场数据根目录')
    parser.add_argument('--output', type=str, default=str(config.OUTPUT_DIR), help='输出目录')
    args = parser.parse_args()
    
    config.MARKET_DATA_ROOT = Path(args.market_root)
    config.OUTPUT_DIR = Path(args.output)
    config.LOOKBACK_SECONDS = args.lookback
    
    print("="*60)
    print("🚀 高频因子提取系统 v3（修复版）")
    print("="*60)
    print(f"📁 市场数据目录：{config.MARKET_DATA_ROOT}")
    print(f"📁 输出目录：{config.OUTPUT_DIR}")
    print(f"⏱️  回看窗口：{config.LOOKBACK_SECONDS}秒")
    print(f"📅 日期范围：{config.START_DATE} ~ {config.END_DATE}")
    print(f"🔄 重采样：{config.RESAMPLE_FREQUENCY}")
    print("="*60)
    
    symbols = discover_symbols(config)
    if not symbols:
        print("❌ 未发现交易对")
        return
        
    if args.date:
        dates = [args.date]
    else:
        dates = pd.date_range(
            start=pd.to_datetime(config.START_DATE, format="%Y%m%d"),
            end=pd.to_datetime(config.END_DATE, format="%Y%m%d"),
            freq="D"
        ).strftime("%Y%m%d").tolist()
        
    if args.symbol:
        symbols = [s for s in symbols if args.symbol in s]
        
    print(f"📊 待处理：{len(symbols)} 交易对 × {len(dates)} 天 = {len(symbols)*len(dates)} 任务")
    
    all_summaries = []
    total = len(symbols) * len(dates)
    current = 0
    
    for symbol in symbols:
        for date_str in dates:
            current += 1
            print(f"\n[{current}/{total}]")
            try:
                summary = process_symbol(symbol, date_str, config)
                all_summaries.append(summary)
            except Exception as e:
                print(f"❌ 失败：{e}")
                import traceback
                traceback.print_exc()
                all_summaries.append({
                    'symbol': symbol,
                    'date': date_str,
                    'status': 'failed',
                    'error': str(e)
                })
                
    generate_summary(all_summaries, config)
    print("\n" + "="*60)
    print("🎉 因子提取完成!")
    print("="*60)

if __name__ == "__main__":
    main()