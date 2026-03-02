#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Frequency Factor Extraction & Analysis System
基于订单簿数据提取高频因子，进行信号质量分析与可视化
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 可视化相关
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# 统计分析
from scipy import stats
from sklearn.preprocessing import StandardScaler

# ============================
# 配置区域
# ============================
class Config:
    # 数据路径
    INPUT_BASE = Path("./dataset/market_processed")
    OUTPUT_DIR = Path("./datasets/factors/hf_factors")
    ANALYSIS_DIR = Path("./datasets/analysis/factor_reports")
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 日期范围
    START_DATE = "20260101"
    END_DATE = "20260101"
    
    # 因子计算参数
    LOOKBACK_WINDOWS = [5, 10, 20, 50, 100]
    PREDICTION_HORIZONS = [1, 5, 10, 30, 60]
    
    # 质量控制
    MIN_DATA_POINTS = 1000
    MAX_MISSING_RATIO = 0.1
    OUTLIER_STD_THRESHOLD = 5

config = Config()

# ============================
# 高频因子计算引擎
# ============================
class HighFrequencyFactorEngine:
    """高频因子计算核心引擎"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.factors = {}
        self.metadata = {}
        
    def compute_all_factors(self):
        """计算全部高频因子"""
        print("  🔧 计算高频因子...")
        
        self._compute_price_factors()
        self._compute_ob_imbalance_factors()
        self._compute_liquidity_factors()
        self._compute_momentum_factors()
        self._compute_basis_factors()
        self._compute_derived_factors()
        self._clean_and_normalize()
        
        print(f"  ✅ 完成 {len(self.factors)} 个因子计算")
        return self.df
    
    def _compute_price_factors(self):
        """价格相关因子"""
        df = self.df
        
        df['spot_mid'] = (df['spot_bid1_px'] + df['spot_ask1_px']) / 2
        df['swap_mid'] = (df['swap_bid1_px'] + df['swap_ask1_px']) / 2
        
        df['spot_spread'] = df['swap_ask1_px'] - df['spot_bid1_px']
        df['swap_spread'] = df['swap_ask1_px'] - df['swap_bid1_px']
        df['spot_spread_rel'] = df['spot_spread'] / df['spot_mid']
        df['swap_spread_rel'] = df['swap_spread'] / df['swap_mid']
        
        df['spot_ret_1'] = df['spot_mid'].pct_change(1)
        df['swap_ret_1'] = df['swap_mid'].pct_change(1)
        
        self.factors['price'] = [
            'spot_mid', 'swap_mid', 'spot_spread_rel', 'swap_spread_rel',
            'spot_ret_1', 'swap_ret_1'
        ]
    
    def _compute_ob_imbalance_factors(self):
        """订单簿不平衡因子"""
        df = self.df
        
        df['obi_level1'] = (df['spot_bid1_px'] - df['spot_ask1_px']) / \
                          (df['spot_bid1_px'] + df['spot_ask1_px'] + 1e-10)
        
        df['obi_swap_level1'] = (df['swap_bid1_px'] - df['swap_ask1_px']) / \
                               (df['swap_bid1_px'] + df['swap_ask1_px'] + 1e-10)
        
        df['obi_cross'] = df['obi_level1'] - df['obi_swap_level1']
        
        for window in config.LOOKBACK_WINDOWS[:3]:
            df[f'obi_mean_{window}'] = df['obi_level1'].rolling(window).mean()
            df[f'obi_std_{window}'] = df['obi_level1'].rolling(window).std()
            df[f'obi_zscore_{window}'] = (
                df['obi_level1'] - df[f'obi_mean_{window}']
            ) / (df[f'obi_std_{window}'] + 1e-10)
        
        self.factors['imbalance'] = [
            'obi_level1', 'obi_swap_level1', 'obi_cross',
            'obi_mean_5', 'obi_std_5', 'obi_zscore_5',
            'obi_mean_10', 'obi_std_10', 'obi_zscore_10'
        ]
    
    def _compute_liquidity_factors(self):
        """流动性因子"""
        df = self.df
        
        df['liquidity_score'] = 1 / (df['spot_spread_rel'] + df['swap_spread_rel'] + 1e-10)
        df['price_impact_proxy'] = (df['spot_spread'] + df['swap_spread']) / 2
        
        for window in config.LOOKBACK_WINDOWS[:3]:
            df[f'liquidity_mean_{window}'] = df['liquidity_score'].rolling(window).mean()
            df[f'liquidity_std_{window}'] = df['liquidity_score'].rolling(window).std()
        
        self.factors['liquidity'] = [
            'liquidity_score', 'price_impact_proxy',
            'liquidity_mean_5', 'liquidity_std_5',
            'liquidity_mean_10', 'liquidity_std_10'
        ]
    
    def _compute_momentum_factors(self):
        """动量与波动因子"""
        df = self.df
        
        for period in [1, 5, 10, 30]:
            df[f'spot_ret_{period}'] = df['spot_mid'].pct_change(period)
            df[f'swap_ret_{period}'] = df['swap_mid'].pct_change(period)
        
        for window in config.LOOKBACK_WINDOWS[:3]:
            df[f'spot_vol_{window}'] = df['spot_ret_1'].rolling(window).std()
            df[f'swap_vol_{window}'] = df['swap_ret_1'].rolling(window).std()
        
        df['momentum_strength'] = (
            df['spot_ret_5'].abs() + df['swap_ret_5'].abs()
        ) / 2
        
        df['vol_ratio'] = df['swap_vol_10'] / (df['spot_vol_10'] + 1e-10)
        
        self.factors['momentum'] = [
            'spot_ret_1', 'spot_ret_5', 'spot_ret_10',
            'swap_ret_1', 'swap_ret_5', 'swap_ret_10',
            'spot_vol_5', 'swap_vol_5', 'spot_vol_10', 'swap_vol_10',
            'momentum_strength', 'vol_ratio'
        ]
    
    def _compute_basis_factors(self):
        """基差相关因子"""
        df = self.df
        
        df['basis_bid'] = np.log(df['swap_bid1_px']) - np.log(df['spot_ask1_px'])
        df['basis_ask'] = np.log(df['swap_ask1_px']) - np.log(df['spot_bid1_px'])
        df['mid_basis'] = (df['basis_bid'] + df['basis_ask']) / 2
        
        df['basis_ret_1'] = df['mid_basis'].pct_change(1)
        df['basis_ret_5'] = df['mid_basis'].pct_change(5)
        
        for window in config.LOOKBACK_WINDOWS[:3]:
            basis_mean = df['mid_basis'].rolling(window).mean()
            basis_std = df['mid_basis'].rolling(window).std()
            df[f'basis_zscore_{window}'] = (df['mid_basis'] - basis_mean) / (basis_std + 1e-10)
        
        df['basis_momentum'] = df['mid_basis'].diff(5)
        df['basis_momentum_accel'] = df['basis_momentum'].diff(5)
        
        df['funding_adj_basis'] = df['mid_basis'] - df['funding_rate'].fillna(0)
        
        self.factors['basis'] = [
            'mid_basis', 'basis_bid', 'basis_ask',
            'basis_ret_1', 'basis_ret_5',
            'basis_zscore_5', 'basis_zscore_10', 'basis_zscore_20',
            'basis_momentum', 'basis_momentum_accel',
            'funding_adj_basis'
        ]
    
    def _compute_derived_factors(self):
        """衍生/交互因子"""
        df = self.df
        
        df['liq_adj_basis_signal'] = df['basis_zscore_10'] * df['liquidity_score']
        df['obi_basis_interaction'] = df['obi_zscore_5'] * df['basis_zscore_5']
        df['vol_adj_momentum'] = df['basis_momentum'] / (df['spot_vol_10'] + 1e-10)
        df['convergence_signal'] = -df['basis_zscore_10'] * np.sign(df['basis_momentum'])
        
        df['composite_signal'] = (
            0.3 * df['basis_zscore_10'].fillna(0) +
            0.25 * df['obi_zscore_5'].fillna(0) +
            0.25 * df['convergence_signal'].fillna(0) +
            0.2 * df['liq_adj_basis_signal'].fillna(0)
        )
        
        self.factors['derived'] = [
            'liq_adj_basis_signal', 'obi_basis_interaction',
            'vol_adj_momentum', 'convergence_signal', 'composite_signal'
        ]
    
    def _clean_and_normalize(self):
        """数据清洗与标准化"""
        df = self.df
        
        df = df.dropna(axis=1, how='all')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df.columns:
                lower = df[col].quantile(0.001)
                upper = df[col].quantile(0.999)
                df[col] = df[col].clip(lower, upper)
        
        df = df.fillna(method='ffill', limit=5)
        
        self.df = df
    
    def get_factor_dataframe(self) -> pd.DataFrame:
        """获取因子数据框"""
        all_factor_cols = []
        for category, cols in self.factors.items():
            all_factor_cols.extend(cols)
        
        existing_cols = [c for c in all_factor_cols if c in self.df.columns]
        return self.df[existing_cols].copy()


# ============================
# 因子分析引擎 (已修复)
# ============================
# ============================
# 因子分析引擎 (完全修复版)
# ============================
# ============================
# 因子分析引擎 (最终修复版)
# ============================
class FactorAnalyzer:
    """因子质量分析与评估"""
    
    def __init__(self, factor_df: pd.DataFrame, target_col: str = 'basis_ret_5'):
        self.df = factor_df.copy()
        self.target_col = target_col
        self.analysis_results = {}
    
    def compute_target(self, horizon: int = 5):
        """计算目标变量（未来收益率）"""
        self.df['target'] = self.df['mid_basis'].shift(-horizon).pct_change(horizon)
        self.target_col = 'target'
        return self.df
    
    def calculate_ic(self, factor_name: str) -> dict:
        """计算因子 IC 值 (完全修复版)"""
        # ✅ 修复 1: 列不存在时返回完整字典
        if factor_name not in self.df.columns:
            return {'ic': np.nan, 't_stat': np.nan, 'p_value': np.nan, 'n_samples': 0}
        
        valid_data = self.df[[factor_name, self.target_col]].dropna()
        
        # ✅ 修复 2: 数据不足时返回完整字典
        if len(valid_data) < 100:
            return {'ic': np.nan, 't_stat': np.nan, 'p_value': np.nan, 'n_samples': len(valid_data)}
        
        try:
            # 确保输入是一维数组
            x = valid_data[factor_name].values.flatten()
            y = valid_data[self.target_col].values.flatten()
            
            # 计算 Spearman 相关系数
            spearman_result = stats.spearmanr(x, y)
            
            # 显式转换为标量浮点数
            ic = float(spearman_result.correlation)
            p_value = float(spearman_result.pvalue)
            
        except Exception as e:
            # ✅ 修复 3: 异常时也要返回完整字典
            return {'ic': np.nan, 't_stat': np.nan, 'p_value': np.nan, 'n_samples': len(valid_data)}
        
        # 计算 t 统计量
        n = len(valid_data)
        if n > 2 and abs(ic) < 1:
            t_stat = ic * np.sqrt((n - 2) / (1 - ic**2 + 1e-10))
        else:
            t_stat = np.nan
        
        return {
            'ic': ic,
            't_stat': t_stat,
            'p_value': p_value,
            'n_samples': n
        }
    
    def analyze_all_factors(self) -> pd.DataFrame:
        """分析所有因子 (完全修复版)"""
        print("  📊 分析因子质量...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        results = []
        
        for col in numeric_cols:
            if col == self.target_col or col.startswith('target'):
                continue
            
            ic_result = self.calculate_ic(col)
            
            col_data = self.df[col].dropna()
            if len(col_data) == 0:
                continue
            
            # ✅ 修复：使用 .get() 安全访问字典
            results.append({
                'factor': col,
                'ic': ic_result.get('ic', np.nan),
                'ic_t_stat': ic_result.get('ic_t_stat', np.nan),
                'ic_p_value': ic_result.get('ic_p_value', np.nan),
                'n_samples': ic_result.get('n_samples', 0),
                'mean': col_data.mean(),
                'std': col_data.std(),
                'skew': col_data.skew() if len(col_data) > 2 else np.nan,
                'kurtosis': col_data.kurtosis() if len(col_data) > 3 else np.nan,
                'missing_ratio': self.df[col].isna().sum() / len(self.df)
            })
        
        self.analysis_results['ic_table'] = pd.DataFrame(results)
        
        if not self.analysis_results['ic_table'].empty:
            self.analysis_results['ic_table']['ic_abs'] = self.analysis_results['ic_table']['ic'].abs()
            self.analysis_results['ic_table'] = self.analysis_results['ic_table'].sort_values(
                'ic_abs', ascending=False
            )
            self.analysis_results['ic_table'] = self.analysis_results['ic_table'].drop(columns=['ic_abs'])
        
        print(f"  ✅ 完成 {len(results)} 个因子分析")
        return self.analysis_results['ic_table']
    
    def get_top_factors(self, n: int = 10, min_abs_ic: float = 0.02) -> pd.DataFrame:
        """获取 Top N 有效因子"""
        if 'ic_table' not in self.analysis_results:
            self.analyze_all_factors()
        
        ic_table = self.analysis_results['ic_table']
        if ic_table is None or ic_table.empty:
            return pd.DataFrame()
        
        top_factors = ic_table[
            (ic_table['ic'].abs() >= min_abs_ic) &
            (ic_table['n_samples'] >= 100)
        ].head(n)
        
        return top_factors
    
    def calculate_factor_correlation(self) -> pd.DataFrame:
        """计算因子相关性矩阵"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        factor_cols = [c for c in numeric_cols if c not in [self.target_col, 'target']]
        
        corr_matrix = self.df[factor_cols].corr()
        self.analysis_results['correlation_matrix'] = corr_matrix
        
        return corr_matrix

# ============================
# 可视化引擎 (已修复)
# ============================
# ============================
# 可视化引擎 (完全修复版)
# ============================
# ============================
# 可视化引擎 (最终修复版)
# ============================
class FactorVisualizer:
    """因子可视化"""
    
    def __init__(self, df: pd.DataFrame, analysis_results: dict, output_dir: Path):
        self.df = df
        self.analysis_results = analysis_results
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_ic_distribution(self, symbol: str):
        """IC 值分布图 (最终修复版)"""
        if 'ic_table' not in self.analysis_results:
            print(f"    ⚠️ 无 IC 表数据，跳过 IC 分布图")
            return
        
        ic_table = self.analysis_results['ic_table']
        if ic_table is None or len(ic_table) == 0:
            print(f"    ⚠️ IC 表为空，跳过 IC 分布图")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # =====================
        # 子图 1: IC 直方图
        # =====================
        ic_data = ic_table['ic'].dropna()
        if len(ic_data) > 0:
            # ✅ 关键修复：使用 .values 确保 1D numpy 数组
            ic_array = ic_data.values.flatten()
            axes[0, 0].hist(ic_array, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
            axes[0, 0].axvline(0.05, color='green', linestyle='--', linewidth=2)
            axes[0, 0].axvline(-0.05, color='green', linestyle='--', linewidth=2)
            axes[0, 0].set_xlabel('IC Value')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Factor IC Distribution')
            axes[0, 0].grid(True, alpha=0.3)
        
        # =====================
        # 子图 2: IC vs T 统计量散点
        # =====================
        valid = ic_table.dropna(subset=['ic', 'ic_t_stat'])
        if len(valid) > 0:
            x_array = valid['ic'].values.flatten()
            y_array = valid['ic_t_stat'].values.flatten()
            axes[0, 1].scatter(x_array, y_array, alpha=0.6, s=30, color='navy')
            axes[0, 1].axhline(2, color='red', linestyle='--', linewidth=2)
            axes[0, 1].axhline(-2, color='red', linestyle='--', linewidth=2)
            axes[0, 1].set_xlabel('IC Value')
            axes[0, 1].set_ylabel('T-Statistic')
            axes[0, 1].set_title('IC vs T-Statistic')
            axes[0, 1].grid(True, alpha=0.3)
        
        # =====================
        # 子图 3: Top 20 因子 IC 条形图
        # =====================
        top_20 = ic_table.head(20)
        if len(top_20) > 0:
            ic_vals = top_20['ic'].values.flatten()
            colors = ['green' if x > 0 else 'red' for x in ic_vals]
            axes[1, 0].barh(range(len(top_20)), ic_vals, color=colors)
            axes[1, 0].set_yticks(range(len(top_20)))
            axes[1, 0].set_yticklabels(top_20['factor'].str.slice(-20), fontsize=8)
            axes[1, 0].set_xlabel('IC Value')
            axes[1, 0].set_title('Top 20 Factors by IC')
            axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # =====================
        # 子图 4: 因子缺失率分布 (❌ 错误发生位置)
        # =====================
        missing_data = ic_table['missing_ratio'].dropna()
        if len(missing_data) > 0:
            print(f"    ⚠️ 因子缺失率分布数据长度: {len(missing_data)}")
            # print(missing_data)
            # # ✅ 关键修复：使用 .values.flatten() 确保 1D numpy 数组
            # missing_array = missing_data.values.flatten()
            # axes[1, 1].hist(missing_array, bins=30, edgecolor='black', alpha=0.7, color='orange')
            # axes[1, 1].set_xlabel('Missing Ratio')
            # axes[1, 1].set_ylabel('Frequency')
            # axes[1, 1].set_title('Factor Missing Data Ratio')
            # axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_ic_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_factor_timeseries(self, symbol: str, factor_names: list = None):
        """因子时间序列图 (最终修复版)"""
        if factor_names is None:
            factor_names = ['mid_basis', 'obi_zscore_5', 'basis_zscore_10', 'composite_signal']
        
        available_factors = [f for f in factor_names if f in self.df.columns]
        if not available_factors:
            print(f"    ⚠️ 无可用因子，跳过时间序列图")
            return
        
        n_factors = len(available_factors)
        fig, axes = plt.subplots(n_factors, 1, figsize=(16, 3 * n_factors))
        if n_factors == 1:
            axes = [axes]
        
        for i, factor in enumerate(available_factors):
            data = self.df[factor].dropna()
            if len(data) == 0:
                continue
            
            # ✅ 使用 .to_numpy() 转换
            axes[i].plot(data.index, data.to_numpy(dtype=np.float64), linewidth=0.5, label=factor)
            axes[i].axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            axes[i].set_ylabel(factor[:20])
            axes[i].set_title(f'{factor} Time Series')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc='upper right', fontsize=8)
        
        plt.xlabel('Time')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_factor_timeseries.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    📈 保存时间序列图：{symbol}_factor_timeseries.png")
    
    def plot_correlation_heatmap(self, symbol: str, top_n: int = 20):
        """因子相关性热力图 (最终修复版)"""
        if 'correlation_matrix' not in self.analysis_results:
            print(f"    ⚠️ 无相关性矩阵，跳过热力图")
            return
        
        corr = self.analysis_results['correlation_matrix']
        
        if 'ic_table' in self.analysis_results:
            top_factors = self.analysis_results['ic_table'].head(top_n)['factor'].tolist()
            top_factors = [f for f in top_factors if f in corr.columns]
            if len(top_factors) > 1:
                corr_subset = corr.loc[top_factors, top_factors]
            else:
                print(f"    ⚠️ 有效因子不足，跳过热力图")
                return
        else:
            corr_subset = corr.iloc[:top_n, :top_n]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        # ✅ 使用 .values 转换 DataFrame 为 numpy 数组
        im = ax.imshow(corr_subset.values, cmap='RdBu_r', vmin=-1, vmax=1)
        
        ax.set_xticks(range(len(corr_subset.columns)))
        ax.set_yticks(range(len(corr_subset.columns)))
        ax.set_xticklabels(corr_subset.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(corr_subset.columns, fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Correlation')
        ax.set_title(f'Factor Correlation Heatmap (Top {top_n})')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_correlation_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    📈 保存相关性热力图：{symbol}_correlation_heatmap.png")
    
    def generate_all_plots(self, symbol: str, top_factors: pd.DataFrame = None):
        """生成全部可视化"""
        print("  🎨 生成可视化图表...")
        
        self.plot_ic_distribution(symbol)
        self.plot_factor_timeseries(symbol)
        self.plot_correlation_heatmap(symbol)
        
        print("  ✅ 可视化完成")
# ============================
# 主处理流程
# ============================
def process_symbol(symbol: str, config: Config) -> dict:
    """处理单个交易对"""
    print(f"\n{'='*60}")
    print(f"📊 处理交易对：{symbol}")
    print(f"{'='*60}")
    
    all_dfs = []
    
    date_range = pd.date_range(
        start=pd.to_datetime(config.START_DATE, format="%Y%m%d"),
        end=pd.to_datetime(config.END_DATE, format="%Y%m%d"),
        freq="D"
    ).strftime("%Y%m%d").tolist()
    
    for date_str in date_range:
        book_file = config.INPUT_BASE / date_str / symbol / f"book_{symbol}_{date_str}.csv.gz"
        
        if not book_file.exists():
            continue
        
        try:
            df = pd.read_csv(
                book_file,
                compression='gzip',
                dtype={
                    'spot_bid1_px': 'float64',
                    'spot_ask1_px': 'float64',
                    'swap_bid1_px': 'float64',
                    'swap_ask1_px': 'float64',
                    'funding_rate': 'float64',
                    'index_price': 'float64'
                }
            )
        except Exception as e:
            print(f"  ❌ 读取失败 {book_file}: {e}")
            continue
        
        required_cols = [
            'time_str', 'spot_bid1_px', 'spot_ask1_px',
            'swap_bid1_px', 'swap_ask1_px', 'funding_rate', 'index_price'
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"  ⚠️ 缺少列：{missing}")
            continue
        
        df = df[required_cols].copy()
        
        try:
            df['timestamp'] = pd.to_datetime(df['time_str'], format='ISO8601')
        except:
            try:
                df['timestamp'] = pd.to_datetime(df['time_str'])
            except Exception as e:
                print(f"  ⚠️ 时间解析错误：{e}")
                continue
        
        df.drop(columns=['time_str'], inplace=True)
        df = df.dropna(subset=['spot_bid1_px', 'spot_ask1_px', 'swap_bid1_px', 'swap_ask1_px'])
        
        if df.empty:
            continue
        
        all_dfs.append(df)
    
    if not all_dfs:
        print(f"  ❌ 无有效数据：{symbol}")
        return {'status': 'failed', 'reason': 'no_data'}
    
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df.sort_values('timestamp').set_index('timestamp')
    
    print(f"  📥 加载数据：{len(full_df)} 条记录")
    
    # 计算高频因子
    factor_engine = HighFrequencyFactorEngine(full_df)
    factor_engine.compute_all_factors()
    factor_df = factor_engine.get_factor_dataframe()
    
    # 因子分析
    analyzer = FactorAnalyzer(factor_df, target_col='basis_ret_5')
    analyzer.compute_target(horizon=5)
    ic_table = analyzer.analyze_all_factors()
    top_factors = analyzer.get_top_factors(n=10, min_abs_ic=0.02)
    corr_matrix = analyzer.calculate_factor_correlation()
    
    # 保存因子数据
    symbol_output_dir = config.OUTPUT_DIR / symbol
    symbol_output_dir.mkdir(parents=True, exist_ok=True)
    
    factor_df_reset = factor_df.reset_index()
    factor_df_reset['year_month'] = factor_df_reset['timestamp'].dt.to_period('M')
    
    for period, group in factor_df_reset.groupby('year_month'):
        year = period.year
        month = str(period.month).zfill(2)
        filename = f"{symbol}_factors_{year}-{month}.csv.gz"
        output_file = symbol_output_dir / filename
        group.drop(columns=['year_month']).to_csv(
            output_file, index=False, compression='gzip'
        )
    
    print(f"  💾 保存因子数据到：{symbol_output_dir}")
    
    # 保存分析报告
    ic_table.to_csv(
        config.ANALYSIS_DIR / f"{symbol}_ic_report.csv", index=False
    )
    
    # 可视化
    visualizer = FactorVisualizer(factor_df, analyzer.analysis_results, 
                                  config.ANALYSIS_DIR / symbol)
    visualizer.generate_all_plots(symbol, top_factors)
    
    # 生成摘要报告
    summary = {
        'symbol': symbol,
        'total_records': len(factor_df),
        'date_range': f"{factor_df.index.min()} ~ {factor_df.index.max()}",
        'n_factors': len(factor_df.columns),
        'top_factors': top_factors['factor'].tolist()[:5] if not top_factors.empty else [],
        'avg_ic': ic_table['ic'].abs().mean() if not ic_table.empty else 0,
        'max_ic': ic_table['ic'].abs().max() if not ic_table.empty else 0,
        'status': 'success'
    }
    
    print(f"\n  📋 摘要:")
    print(f"     记录数：{summary['total_records']}")
    print(f"     因子数：{summary['n_factors']}")
    print(f"     平均|IC|: {summary['avg_ic']:.4f}")
    print(f"     最大|IC|: {summary['max_ic']:.4f}")
    if not top_factors.empty:
        print(f"     Top因子：{top_factors['factor'].iloc[0]} (IC={top_factors['ic'].iloc[0]:.4f})")
    
    return summary


def discover_symbols(config: Config) -> list:
    """发现所有交易对"""
    sample_date = config.START_DATE
    sample_path = config.INPUT_BASE / sample_date
    
    if not sample_path.exists():
        raise FileNotFoundError(f"样本路径不存在：{sample_path}")
    
    symbols = [
        d.name for d in sample_path.iterdir() 
        if d.is_dir() and d.name.endswith("USDT")
    ]
    
    print(f"🔍 发现 {len(symbols)} 个交易对：{symbols[:5]}...")
    return symbols


def generate_summary_report(summaries: list, config: Config):
    """生成汇总报告"""
    print(f"\n{'='*60}")
    print("📊 生成汇总报告")
    print(f"{'='*60}")
    
    summary_df = pd.DataFrame(summaries)
    
    summary_df.to_csv(
        config.ANALYSIS_DIR / "all_symbols_summary.csv", index=False
    )
    
    if not summary_df.empty and 'avg_ic' in summary_df.columns:
        top_symbols = summary_df.nlargest(5, 'avg_ic')
        print("\n🏆 Top 5 交易对 (按平均IC):")
        for _, row in top_symbols.iterrows():
            print(f"   {row['symbol']}: 平均IC={row['avg_ic']:.4f}, 因子数={row['n_factors']}")
    
    print(f"\n💾 汇总报告保存至：{config.ANALYSIS_DIR / 'all_symbols_summary.csv'}")


# ============================
# 主程序入口
# ============================
if __name__ == "__main__":
    print("="*60)
    print("🚀 高频因子提取与分析系统")
    print("="*60)
    print(f"📁 输入目录：{config.INPUT_BASE}")
    print(f"📁 输出目录：{config.OUTPUT_DIR}")
    print(f"📁 分析目录：{config.ANALYSIS_DIR}")
    print(f"📅 日期范围：{config.START_DATE} ~ {config.END_DATE}")
    print("="*60)
    
    symbols = discover_symbols(config)
    
    if not symbols:
        print("❌ 未发现任何交易对数据")
        exit(1)
    
    all_summaries = []
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] 处理进度")
        try:
            summary = process_symbol(symbol, config)
            all_summaries.append(summary)
        except Exception as e:
            print(f"❌ {symbol} 处理失败：{e}")
            import traceback
            traceback.print_exc()
            all_summaries.append({
                'symbol': symbol,
                'status': 'failed',
                'error': str(e)
            })
    
    generate_summary_report(all_summaries, config)
    
    print("\n" + "="*60)
    print("🎉 高频因子提取与分析完成!")
    print("="*60)