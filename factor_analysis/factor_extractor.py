#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Frequency Factor Extraction & Analysis System
基于订单簿数据提取高频因子，进行信号质量分析与可视化
包含 MODWT 小波变换因子用于 Lead-Lag 效应研究
支持 500ms 重采样和前向填充
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
from scipy.signal import correlate
from sklearn.preprocessing import StandardScaler

# 小波变换相关
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("⚠️ 警告：pywt 库未安装，MODWT 因子将不可用。请运行：pip install PyWavelets")

# ============================
# 配置区域
# ============================
class Config:
    # 数据路径
    INPUT_BASE = Path("./dataset/market_processed")
    OUTPUT_DIR = Path("./datasets/factors/hf_factors_30s")
    ANALYSIS_DIR = Path("./datasets/analysis/factor_reports")
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 日期范围
    START_DATE = "20260101"
    END_DATE = "20260131"
    
    # 因子计算参数
    LOOKBACK_WINDOWS = [5, 10, 20, 50, 100]
    PREDICTION_HORIZONS = [1, 5, 10, 30, 60]
    
    # MODWT 参数
    MODWT_WAVELET = 'db4'  # 小波基
    MODWT_LEVELS = 4       # 分解层数
    
    # ✅ 新增：重采样配置
    # RESAMPLE_FREQUENCY = '500ms'  # 重采样频率
    RESAMPLE_FREQUENCY = '30s'  # 重采样频率
    RESAMPLE_METHOD = 'last'      # 价格列重采样方法 (last/mean/ohlc)
    FILL_METHOD = 'ffill'         # 填充方法 (ffill/bfill/interpolate)
    
    # 质量控制
    MIN_DATA_POINTS = 1000
    MAX_MISSING_RATIO = 0.1
    OUTLIER_STD_THRESHOLD = 5

config = Config()
# ============================
# Kalman Filter 工具类 (新增)
# ============================
class KalmanFilterUtils:
    """Kalman Filter 工具类 - 用于状态估计和信号去噪"""
    
    @staticmethod
    def kalman_filter_1d(observations, process_variance=1e-5, measurement_variance=1e-2):
        """
        一维 Kalman Filter
        用于价格/收益率去噪和状态估计
        
        参数:
            observations: 观测值序列
            process_variance: 过程噪声方差 (Q)
            measurement_variance: 测量噪声方差 (R)
        
        返回:
            filtered_values: 滤波后的状态估计
            kalman_gain: 卡尔曼增益序列
            estimation_error: 估计误差协方差
        """
        n = len(observations)
        filtered_values = np.zeros(n)
        kalman_gain = np.zeros(n)
        estimation_error = np.zeros(n)
        
        # 初始化
        x_est = observations[0]  # 初始状态估计
        p_est = 1.0  # 初始估计误差协方差
        
        for i in range(n):
            if np.isnan(observations[i]):
                filtered_values[i] = filtered_values[i-1] if i > 0 else 0
                kalman_gain[i] = 0
                estimation_error[i] = p_est
                continue
            
            # 预测步骤
            x_pred = x_est
            p_pred = p_est + process_variance
            
            # 更新步骤
            kalman_gain[i] = p_pred / (p_pred + measurement_variance)
            x_est = x_pred + kalman_gain[i] * (observations[i] - x_pred)
            p_est = (1 - kalman_gain[i]) * p_pred
            
            filtered_values[i] = x_est
            estimation_error[i] = p_est
        
        return filtered_values, kalman_gain, estimation_error
    
    @staticmethod
    def kalman_filter_trend(series, process_variance=1e-4, measurement_variance=1e-2):
        """
        趋势跟踪 Kalman Filter (带速度估计)
        状态向量：[位置，速度]
        
        返回:
            trend: 估计的趋势值
            velocity: 估计的速度 (变化率)
        """
        n = len(series)
        trend = np.zeros(n)
        velocity = np.zeros(n)
        
        # 状态向量 [x, v]
        x_est = np.array([series[0] if not np.isnan(series[0]) else 0, 0.0])
        # 状态协方差矩阵
        p_est = np.eye(2)
        
        # 状态转移矩阵 (恒定速度模型)
        F = np.array([[1, 1], [0, 1]])
        # 观测矩阵
        H = np.array([1, 0])
        # 过程噪声协方差
        Q = np.eye(2) * process_variance
        # 测量噪声协方差
        R = measurement_variance
        
        for i in range(n):
            if np.isnan(series[i]):
                trend[i] = trend[i-1] if i > 0 else 0
                velocity[i] = velocity[i-1] if i > 0 else 0
                continue
            
            # 预测
            x_pred = F @ x_est
            p_pred = F @ p_est @ F.T + Q
            
            # 更新
            y = series[i] - H @ x_pred  # 残差
            S = H @ p_pred @ H.T + R  # 残差协方差
            K = p_pred @ H.T / S  # 卡尔曼增益
            
            x_est = x_pred + K * y
            p_est = (np.eye(2) - K @ H) @ p_pred
            
            trend[i] = x_est[0]
            velocity[i] = x_est[1]
        
        return trend, velocity
    
    @staticmethod
    def kalman_volatility(returns, window=50, process_variance=1e-3):
        """
        基于 Kalman Filter 的动态波动率估计
        使用指数加权移动方差作为观测
        """
        n = len(returns)
        volatility = np.zeros(n)
        
        # 计算滚动方差作为观测
        ewm_var = returns.ewm(span=window).var().values
        
        # 对波动率进行 Kalman 滤波
        log_vol = np.log(ewm_var + 1e-10)
        filtered_log_vol, _, _ = KalmanFilterUtils.kalman_filter_1d(
            log_vol, 
            process_variance=process_variance,
            measurement_variance=0.1
        )
        
        volatility = np.exp(filtered_log_vol)
        return volatility
    
    @staticmethod
    def kalman_mean_reversion(series, window=100):
        """
        均值回归信号 (基于 Kalman Filter 估计的均衡价格)
        
        返回:
            reversion_signal: 均值回归信号 (负=高估，正=低估)
            equilibrium: 估计的均衡价格
        """
        # 使用 Kalman Filter 估计均衡价格
        equilibrium, _, _ = KalmanFilterUtils.kalman_filter_1d(
            series.values if hasattr(series, 'values') else series,
            process_variance=1e-5,
            measurement_variance=1e-2
        )
        
        # 计算偏离程度
        current_price = series.values if hasattr(series, 'values') else series
        reversion_signal = (equilibrium - current_price) / (equilibrium + 1e-10)
        
        return reversion_signal, equilibrium
    
# ============================
# MODWT 工具函数
# ============================
class MODWTUtils:
    """MODWT 小波变换工具类"""
    
    @staticmethod
    def modwt_decompose(series, wavelet='db4', level=4):
        """
        执行 MODWT 分解
        返回：小波系数列表 (D1, D2, ..., Dn) 和尺度系数 (Sn)
        """
        if not PYWT_AVAILABLE:
            return None, None
        
        try:
            # 处理 NaN 值
            series_clean = series.fillna(method='ffill').fillna(method='bfill')
            
            # 执行 MODWT
            coeffs = pywt.wavedec(series_clean.values, wavelet=wavelet, level=level, mode='periodization')
            
            # coeffs[0] 是尺度系数，coeffs[1:] 是小波系数（从粗到细）
            # 反转顺序使其从细到粗 (D1, D2, ..., Dn, Sn)
            wavelet_coeffs = coeffs[1:][::-1]  # D1, D2, ..., Dn
            scale_coeff = coeffs[0]            # Sn
            
            return wavelet_coeffs, scale_coeff
        except Exception as e:
            print(f"    ⚠️ MODWT 分解失败：{e}")
            return None, None
    
    @staticmethod
    def modwt_reconstruct(wavelet_coeffs, scale_coeff, wavelet='db4'):
        """从 MODWT 系数重构序列"""
        if not PYWT_AVAILABLE or wavelet_coeffs is None:
            return None
        
        try:
            # 恢复原始顺序 (Sn, Dn, ..., D1)
            coeffs = [scale_coeff] + wavelet_coeffs[::-1]
            reconstructed = pywt.waverec(coeffs, wavelet=wavelet, mode='periodization')
            return reconstructed[:len(wavelet_coeffs[0])]
        except Exception as e:
            print(f"    ⚠️ MODWT 重构失败：{e}")
            return None
    
    @staticmethod
    def calculate_cross_wavelet_correlation(series1, series2, wavelet='db4', level=4, lag_range=10):
        """
        计算跨序列小波相关性，用于 Lead-Lag 分析
        返回：不同尺度下的最优滞后和相关性
        """
        if not PYWT_AVAILABLE:
            return None
        
        try:
            results = {}
            
            # 分解两个序列
            coeffs1, scale1 = MODWTUtils.modwt_decompose(series1, wavelet, level)
            coeffs2, scale2 = MODWTUtils.modwt_decompose(series2, wavelet, level)
            
            if coeffs1 is None or coeffs2 is None:
                return None
            
            # 对每个尺度计算互相关
            for i, (d1, d2) in enumerate(zip(coeffs1, coeffs2)):
                scale_name = f'D{i+1}'
                
                # 计算互相关
                correlation = correlate(d1, d2, mode='full')
                lags = np.arange(-len(d1) + 1, len(d1))
                
                # 在指定滞后范围内找最优
                valid_mask = (lags >= -lag_range) & (lags <= lag_range)
                if valid_mask.sum() > 0:
                    valid_corr = correlation[valid_mask]
                    valid_lags = lags[valid_mask]
                    
                    # 归一化
                    max_corr = np.sqrt(np.sum(d1**2) * np.sum(d2**2))
                    if max_corr > 0:
                        valid_corr = valid_corr / max_corr
                    
                    # 找到最优滞后
                    best_idx = np.argmax(np.abs(valid_corr))
                    best_lag = valid_lags[best_idx]
                    best_corr = valid_corr[best_idx]
                    
                    results[scale_name] = {
                        'optimal_lag': best_lag,
                        'correlation': best_corr,
                        'lag_range': lag_range
                    }
            
            return results
        except Exception as e:
            print(f"    ⚠️ 跨小波相关计算失败：{e}")
            return None
    
    @staticmethod
    def calculate_wavelet_energy(coeffs):
        """计算小波能量（波动率代理）"""
        if coeffs is None:
            return None
        
        energies = {}
        for i, coeff in enumerate(coeffs):
            energies[f'D{i+1}_energy'] = np.mean(coeff**2)
        
        return energies

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
        # ✅ 新增：Kalman Filter 相关因子
        self._compute_kalman_factors()        
        # ✅ 新增：MODWT 相关因子
        if PYWT_AVAILABLE:
            self._compute_modwt_factors()
            self._compute_lead_lag_factors()
        else:
            print("  ⚠️ 跳过 MODWT 因子（pywt 未安装）")
        
        self._compute_derived_factors()
        self._clean_and_normalize()
        print(f"  ✅ 完成 {len(self.factors)} 个因子计算")
        return self.df
    
    def _compute_price_factors(self):
        """价格相关因子"""
        df = self.df
        df['spot_mid'] = (df['spot_bid1_px'] + df['spot_ask1_px']) / 2
        df['swap_mid'] = (df['swap_bid1_px'] + df['swap_ask1_px']) / 2
        df['spot_spread'] = df['spot_ask1_px'] - df['spot_bid1_px']
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
    
    # ============================
    # ✅ 新增：Kalman Filter 相关因子
    # ============================
    def _compute_kalman_factors(self):
        """Kalman Filter 相关因子"""
        print("    🔮 计算 Kalman Filter 因子...")
        df = self.df
        
        # 1. 现货价格 Kalman 去噪
        spot_mid = df['spot_mid'].values
        spot_filtered, spot_kg, spot_err = KalmanFilterUtils.kalman_filter_1d(
            spot_mid, 
            process_variance=1e-5, 
            measurement_variance=1e-2
        )
        df['kalman_spot_filtered'] = spot_filtered
        df['kalman_spot_gain'] = spot_kg
        df['kalman_spot_error'] = spot_err
        
        # 2. 合约价格 Kalman 去噪
        swap_mid = df['swap_mid'].values
        swap_filtered, swap_kg, swap_err = KalmanFilterUtils.kalman_filter_1d(
            swap_mid,
            process_variance=1e-5,
            measurement_variance=1e-2
        )
        df['kalman_swap_filtered'] = swap_filtered
        df['kalman_swap_gain'] = swap_kg
        df['kalman_swap_error'] = swap_err
        
        # 3. 基差 Kalman 去噪
        basis = df['mid_basis'].values
        basis_filtered, basis_kg, basis_err = KalmanFilterUtils.kalman_filter_1d(
            basis,
            process_variance=1e-5,
            measurement_variance=1e-3
        )
        df['kalman_basis_filtered'] = basis_filtered
        df['kalman_basis_gain'] = basis_kg
        df['kalman_basis_error'] = basis_err
        
        # 4. 趋势跟踪 Kalman (带速度估计)
        spot_trend, spot_velocity = KalmanFilterUtils.kalman_filter_trend(
            df['spot_mid'],
            process_variance=1e-4,
            measurement_variance=1e-2
        )
        df['kalman_spot_trend'] = spot_trend
        df['kalman_spot_velocity'] = spot_velocity
        
        swap_trend, swap_velocity = KalmanFilterUtils.kalman_filter_trend(
            df['swap_mid'],
            process_variance=1e-4,
            measurement_variance=1e-2
        )
        df['kalman_swap_trend'] = swap_trend
        df['kalman_swap_velocity'] = swap_velocity
        
        # 5. 动态波动率估计
        spot_ret = df['spot_ret_1'].fillna(0)
        swap_ret = df['swap_ret_1'].fillna(0)
        
        spot_kalman_vol = KalmanFilterUtils.kalman_volatility(
            spot_ret, window=50, process_variance=1e-3
        )
        swap_kalman_vol = KalmanFilterUtils.kalman_volatility(
            swap_ret, window=50, process_variance=1e-3
        )
        df['kalman_spot_vol'] = spot_kalman_vol
        df['kalman_swap_vol'] = swap_kalman_vol
        df['kalman_vol_ratio'] = swap_kalman_vol / (spot_kalman_vol + 1e-10)
        
        # 6. 均值回归信号
        basis_mr_signal, basis_equilibrium = KalmanFilterUtils.kalman_mean_reversion(
            df['mid_basis'], window=100
        )
        df['kalman_basis_mr_signal'] = basis_mr_signal
        df['kalman_basis_equilibrium'] = basis_equilibrium
        
        spot_mr_signal, spot_equilibrium = KalmanFilterUtils.kalman_mean_reversion(
            df['spot_mid'], window=100
        )
        df['kalman_spot_mr_signal'] = spot_mr_signal
        
        swap_mr_signal, swap_equilibrium = KalmanFilterUtils.kalman_mean_reversion(
            df['swap_mid'], window=100
        )
        df['kalman_swap_mr_signal'] = swap_mr_signal
        
        # 7. Kalman 创新序列 (新息，用于检测异常)
        df['kalman_spot_innovation'] = df['spot_mid'] - df['kalman_spot_filtered']
        df['kalman_swap_innovation'] = df['swap_mid'] - df['kalman_swap_filtered']
        df['kalman_basis_innovation'] = df['mid_basis'] - df['kalman_basis_filtered']
        
        # 8. 创新标准化 (Z-Score)
        for col in ['kalman_spot_innovation', 'kalman_swap_innovation', 'kalman_basis_innovation']:
            rolling_mean = df[col].rolling(50).mean()
            rolling_std = df[col].rolling(50).std()
            df[f'{col}_zscore'] = (df[col] - rolling_mean) / (rolling_std + 1e-10)
        
        # 9. 趋势强度 (速度/位置比率)
        df['kalman_spot_trend_strength'] = (
            df['kalman_spot_velocity'] / (df['kalman_spot_trend'].abs() + 1e-10)
        )
        df['kalman_swap_trend_strength'] = (
            df['kalman_swap_velocity'] / (df['kalman_swap_trend'].abs() + 1e-10)
        )
        
        # 10. 跨市场趋势分歧
        df['kalman_trend_divergence'] = (
            df['kalman_spot_velocity'] - df['kalman_swap_velocity']
        )
        
        # 11. Kalman 增益变化率 (反映市场稳定性)
        df['kalman_spot_gain_change'] = df['kalman_spot_gain'].diff(1)
        df['kalman_basis_gain_change'] = df['kalman_basis_gain'].diff(1)
        
        # 12. 估计误差变化 (不确定性指标)
        df['kalman_basis_error_change'] = df['kalman_basis_error'].diff(1)
        df['kalman_error_ratio'] = (
            df['kalman_basis_error'] / (df['kalman_spot_error'] + df['kalman_swap_error'] + 1e-10)
        )
        
        self.factors['kalman'] = [
            # 滤波值
            'kalman_spot_filtered', 'kalman_swap_filtered', 'kalman_basis_filtered',
            # 卡尔曼增益
            'kalman_spot_gain', 'kalman_swap_gain', 'kalman_basis_gain',
            # 估计误差
            'kalman_spot_error', 'kalman_swap_error', 'kalman_basis_error',
            # 趋势跟踪
            'kalman_spot_trend', 'kalman_spot_velocity',
            'kalman_swap_trend', 'kalman_swap_velocity',
            # 波动率
            'kalman_spot_vol', 'kalman_swap_vol', 'kalman_vol_ratio',
            # 均值回归
            'kalman_basis_mr_signal', 'kalman_basis_equilibrium',
            'kalman_spot_mr_signal', 'kalman_swap_mr_signal',
            # 创新序列
            'kalman_spot_innovation', 'kalman_swap_innovation', 'kalman_basis_innovation',
            'kalman_spot_innovation_zscore', 'kalman_swap_innovation_zscore', 'kalman_basis_innovation_zscore',
            # 趋势强度
            'kalman_spot_trend_strength', 'kalman_swap_trend_strength',
            # 跨市场
            'kalman_trend_divergence',
            # 增益和误差变化
            'kalman_spot_gain_change', 'kalman_basis_gain_change',
            'kalman_basis_error_change', 'kalman_error_ratio'
        ]
        
        print(f"    ✅ 完成 {len(self.factors['kalman'])} 个 Kalman 因子计算")

    # ============================
    # MODWT 相关因子
    # ============================
    def _compute_modwt_factors(self):
        """MODWT 小波分解因子"""
        if not PYWT_AVAILABLE:
            return
        
        print("    📊 计算 MODWT 因子...")
        df = self.df
        
        # 对现货和合约收益率进行小波分解
        spot_ret = df['spot_ret_1'].fillna(0)
        swap_ret = df['swap_ret_1'].fillna(0)
        
        # 使用滚动窗口进行 MODWT 分解（避免未来数据泄露）
        window_size = 256  # MODWT 需要足够的数据点
        min_window = 128
        
        # 初始化 MODWT 因子列
        for level in range(1, config.MODWT_LEVELS + 1):
            df[f'spot_modwt_D{level}'] = np.nan
            df[f'swap_modwt_D{level}'] = np.nan
            df[f'spot_modwt_energy_D{level}'] = np.nan
            df[f'swap_modwt_energy_D{level}'] = np.nan
        
        df['spot_modwt_scale'] = np.nan
        df['swap_modwt_scale'] = np.nan
        
        # 滚动计算 MODWT
        for i in range(window_size, len(df)):
            window_start = max(0, i - window_size)
            window_end = i
            
            spot_window = spot_ret.iloc[window_start:window_end]
            swap_window = swap_ret.iloc[window_start:window_end]
            
            if len(spot_window) < min_window:
                continue
            
            # 现货 MODWT 分解
            spot_coeffs, spot_scale = MODWTUtils.modwt_decompose(
                spot_window, 
                wavelet=config.MODWT_WAVELET, 
                level=config.MODWT_LEVELS
            )
            
            # 合约 MODWT 分解
            swap_coeffs, swap_scale = MODWTUtils.modwt_decompose(
                swap_window, 
                wavelet=config.MODWT_WAVELET, 
                level=config.MODWT_LEVELS
            )
            
            if spot_coeffs is not None:
                for level, coeff in enumerate(spot_coeffs, 1):
                    if len(coeff) > 0:
                        df.iloc[i, df.columns.get_loc(f'spot_modwt_D{level}')] = coeff[-1]
                        df.iloc[i, df.columns.get_loc(f'spot_modwt_energy_D{level}')] = np.mean(coeff**2)
                if spot_scale is not None and len(spot_scale) > 0:
                    df.iloc[i, df.columns.get_loc('spot_modwt_scale')] = spot_scale[-1]
            
            if swap_coeffs is not None:
                for level, coeff in enumerate(swap_coeffs, 1):
                    if len(coeff) > 0:
                        df.iloc[i, df.columns.get_loc(f'swap_modwt_D{level}')] = coeff[-1]
                        df.iloc[i, df.columns.get_loc(f'swap_modwt_energy_D{level}')] = np.mean(coeff**2)
                if swap_scale is not None and len(swap_scale) > 0:
                    df.iloc[i, df.columns.get_loc('swap_modwt_scale')] = swap_scale[-1]
        
        # 计算小波能量比率（多尺度波动率）
        for level in range(1, config.MODWT_LEVELS + 1):
            df[f'modwt_energy_ratio_D{level}'] = (
                df[f'swap_modwt_energy_D{level}'] / 
                (df[f'spot_modwt_energy_D{level}'] + 1e-10)
            )
        
        # 总能量比率
        spot_total_energy = sum(df[f'spot_modwt_energy_D{i}'] for i in range(1, config.MODWT_LEVELS + 1))
        swap_total_energy = sum(df[f'swap_modwt_energy_D{i}'] for i in range(1, config.MODWT_LEVELS + 1))
        df['modwt_total_energy_ratio'] = swap_total_energy / (spot_total_energy + 1e-10)
        
        self.factors['modwt'] = [
            f'spot_modwt_D{i}' for i in range(1, config.MODWT_LEVELS + 1)
        ] + [
            f'swap_modwt_D{i}' for i in range(1, config.MODWT_LEVELS + 1)
        ] + [
            f'spot_modwt_energy_D{i}' for i in range(1, config.MODWT_LEVELS + 1)
        ] + [
            f'swap_modwt_energy_D{i}' for i in range(1, config.MODWT_LEVELS + 1)
        ] + [
            f'modwt_energy_ratio_D{i}' for i in range(1, config.MODWT_LEVELS + 1)
        ] + [
            'spot_modwt_scale', 'swap_modwt_scale', 'modwt_total_energy_ratio'
        ]
    
    def _compute_lead_lag_factors(self):
        """Lead-Lag 效应因子（基于 MODWT 互相关）"""
        if not PYWT_AVAILABLE:
            return
        
        print("    🔗 计算 Lead-Lag 因子...")
        df = self.df
        
        # 滚动计算跨市场小波互相关
        window_size = 256
        min_window = 128
        lag_range = 10
        
        # 初始化 Lead-Lag 因子
        for level in range(1, config.MODWT_LEVELS + 1):
            df[f'lead_lag_D{level}_optimal_lag'] = np.nan
            df[f'lead_lag_D{level}_correlation'] = np.nan
        
        df['lead_lag_aggregate_lag'] = np.nan
        df['lead_lag_aggregate_corr'] = np.nan
        df['lead_lag_direction'] = np.nan  # 正=合约领先，负=现货领先
        
        spot_ret = df['spot_ret_1'].fillna(0)
        swap_ret = df['swap_ret_1'].fillna(0)
        
        for i in range(window_size, len(df)):
            window_start = max(0, i - window_size)
            window_end = i
            
            spot_window = spot_ret.iloc[window_start:window_end]
            swap_window = swap_ret.iloc[window_start:window_end]
            
            if len(spot_window) < min_window:
                continue
            
            # 计算跨小波相关性
            cross_corr_results = MODWTUtils.calculate_cross_wavelet_correlation(
                spot_window, swap_window,
                wavelet=config.MODWT_WAVELET,
                level=config.MODWT_LEVELS,
                lag_range=lag_range
            )
            
            if cross_corr_results is not None:
                total_lag = 0
                total_corr = 0
                n_scales = 0
                
                for level in range(1, config.MODWT_LEVELS + 1):
                    scale_name = f'D{level}'
                    if scale_name in cross_corr_results:
                        result = cross_corr_results[scale_name]
                        df.iloc[i, df.columns.get_loc(f'lead_lag_D{level}_optimal_lag')] = result['optimal_lag']
                        df.iloc[i, df.columns.get_loc(f'lead_lag_D{level}_correlation')] = result['correlation']
                        
                        # 加权聚合（高频尺度权重更高）
                        weight = 1 / level
                        total_lag += result['optimal_lag'] * weight
                        total_corr += abs(result['correlation']) * weight
                        n_scales += weight
                
                if n_scales > 0:
                    df.iloc[i, df.columns.get_loc('lead_lag_aggregate_lag')] = total_lag / n_scales
                    df.iloc[i, df.columns.get_loc('lead_lag_aggregate_corr')] = total_corr / n_scales
                    
                    # Lead-Lag 方向：正滞后表示合约领先现货
                    df.iloc[i, df.columns.get_loc('lead_lag_direction')] = np.sign(total_lag / n_scales)
        
        # 计算 Lead-Lag 稳定性（滚动标准差）
        for window in [20, 50]:
            df[f'lead_lag_lag_std_{window}'] = df['lead_lag_aggregate_lag'].rolling(window).std()
            df[f'lead_lag_corr_std_{window}'] = df['lead_lag_aggregate_corr'].rolling(window).std()
        
        # Lead-Lag 强度因子
        df['lead_lag_strength'] = (
            df['lead_lag_aggregate_corr'] * 
            (1 / (df['lead_lag_lag_std_20'] + 1e-10))
        )
        
        # 多尺度 Lead-Lag 一致性
        df['lead_lag_consistency'] = np.nan
        for i in range(window_size, len(df)):
            lags = []
            for level in range(1, config.MODWT_LEVELS + 1):
                lag_val = df.iloc[i, df.columns.get_loc(f'lead_lag_D{level}_optimal_lag')]
                if not np.isnan(lag_val):
                    lags.append(np.sign(lag_val))
            if len(lags) > 0:
                # 一致性 = 同号比例
                df.iloc[i, df.columns.get_loc('lead_lag_consistency')] = abs(sum(lags)) / len(lags)
        
        self.factors['lead_lag'] = [
            f'lead_lag_D{level}_optimal_lag' for level in range(1, config.MODWT_LEVELS + 1)
        ] + [
            f'lead_lag_D{level}_correlation' for level in range(1, config.MODWT_LEVELS + 1)
        ] + [
            'lead_lag_aggregate_lag', 'lead_lag_aggregate_corr', 'lead_lag_direction',
            'lead_lag_lag_std_20', 'lead_lag_lag_std_50',
            'lead_lag_corr_std_20', 'lead_lag_corr_std_50',
            'lead_lag_strength', 'lead_lag_consistency'
        ]
    
    def _compute_derived_factors(self):
        """衍生/交互因子"""
        df = self.df
        df['liq_adj_basis_signal'] = df['basis_zscore_10'] * df['liquidity_score']
        df['obi_basis_interaction'] = df['obi_zscore_5'] * df['basis_zscore_5']
        df['vol_adj_momentum'] = df['basis_momentum'] / (df['spot_vol_10'] + 1e-10)
        df['convergence_signal'] = -df['basis_zscore_10'] * np.sign(df['basis_momentum'])
        
        # 基于 Lead-Lag 的衍生因子
        if 'lead_lag_aggregate_lag' in df.columns:
            df['lead_lag_adj_basis'] = df['mid_basis'] * df['lead_lag_direction'].fillna(0)
            df['lead_lag_momentum'] = df['basis_momentum'] * df['lead_lag_strength'].fillna(0)
        
        df['composite_signal'] = (
            0.3 * df['basis_zscore_10'].fillna(0) +
            0.25 * df['obi_zscore_5'].fillna(0) +
            0.25 * df['convergence_signal'].fillna(0) +
            0.2 * df['liq_adj_basis_signal'].fillna(0)
        )
        
        # 包含 Lead-Lag 信息的复合信号
        if 'lead_lag_strength' in df.columns:
            df['composite_signal_ll'] = (
                0.25 * df['basis_zscore_10'].fillna(0) +
                0.2 * df['obi_zscore_5'].fillna(0) +
                0.2 * df['convergence_signal'].fillna(0) +
                0.15 * df['liq_adj_basis_signal'].fillna(0) +
                0.2 * df['lead_lag_strength'].fillna(0)
            )
        
        self.factors['derived'] = [
            'liq_adj_basis_signal', 'obi_basis_interaction',
            'vol_adj_momentum', 'convergence_signal', 'composite_signal'
        ]
        
        # Lead-Lag 衍生因子
        if 'lead_lag_adj_basis' in df.columns:
            self.factors['derived'].extend([
                'lead_lag_adj_basis', 'lead_lag_momentum', 'composite_signal_ll'
            ])

        
        # ✅ 新增：Kalman 衍生因子
        if 'kalman_basis_mr_signal' in df.columns:
            df['kalman_liq_adj_mr'] = df['kalman_basis_mr_signal'] * df['liquidity_score']
            df['kalman_vol_adj_mr'] = df['kalman_basis_mr_signal'] / (df['kalman_spot_vol'] + 1e-10)
            df['kalman_trend_mr_combo'] = (
                0.5 * df['kalman_basis_mr_signal'] + 
                0.5 * df['kalman_spot_velocity'].apply(np.sign)
            )
        
        # Kalman + 基差 Z-Score 组合
        if 'basis_zscore_10' in df.columns and 'kalman_basis_mr_signal' in df.columns:
            df['kalman_basis_combo'] = (
                0.6 * df['basis_zscore_10'] + 
                0.4 * df['kalman_basis_mr_signal']
            )
        
        # Kalman + Lead-Lag 组合
        if 'lead_lag_strength' in df.columns and 'kalman_trend_divergence' in df.columns:
            df['kalman_ll_combo'] = (
                0.5 * df['lead_lag_strength'] + 
                0.5 * np.sign(df['kalman_trend_divergence'])
            )
        
        # 更新衍生因子列表
        kalman_derived = [
            'kalman_liq_adj_mr', 'kalman_vol_adj_mr', 'kalman_trend_mr_combo',
            'kalman_basis_combo', 'kalman_ll_combo'
        ]
        existing_derived = [c for c in kalman_derived if c in df.columns]
        if 'derived' not in self.factors:
            self.factors['derived'] = []
        self.factors['derived'].extend(existing_derived)

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
# 因子分析引擎
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
        """计算因子 IC 值"""
        if factor_name not in self.df.columns:
            return {'ic': np.nan, 't_stat': np.nan, 'p_value': np.nan, 'n_samples': 0}
        
        valid_data = self.df[[factor_name, self.target_col]].dropna()
        
        if len(valid_data) < 100:
            return {'ic': np.nan, 't_stat': np.nan, 'p_value': np.nan, 'n_samples': len(valid_data)}
        
        try:
            x = valid_data[factor_name].values.flatten()
            y = valid_data[self.target_col].values.flatten()
            
            spearman_result = stats.spearmanr(x, y)
            ic = float(spearman_result.correlation)
            p_value = float(spearman_result.pvalue)
        except Exception as e:
            return {'ic': np.nan, 't_stat': np.nan, 'p_value': np.nan, 'n_samples': len(valid_data)}
        
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
        """分析所有因子"""
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
            
            results.append({
                'factor': col,
                'ic': ic_result.get('ic', np.nan),
                'ic_t_stat': ic_result.get('t_stat', np.nan),
                'ic_p_value': ic_result.get('p_value', np.nan),
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
# 可视化引擎
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
        """IC 值分布图"""
        if 'ic_table' not in self.analysis_results:
            print(f"    ⚠️ 无 IC 表数据，跳过 IC 分布图")
            return
        
        ic_table = self.analysis_results['ic_table']
        if ic_table is None or len(ic_table) == 0:
            print(f"    ⚠️ IC 表为空，跳过 IC 分布图")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 子图 1: IC 直方图
        ic_data = ic_table['ic'].dropna()
        if len(ic_data) > 0:
            ic_array = ic_data.to_numpy().flatten()
            axes[0, 0].hist(ic_array, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
            axes[0, 0].axvline(0.05, color='green', linestyle='--', linewidth=2)
            axes[0, 0].axvline(-0.05, color='green', linestyle='--', linewidth=2)
            axes[0, 0].set_xlabel('IC Value')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Factor IC Distribution')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 子图 2: IC vs T 统计量散点
        valid = ic_table.dropna(subset=['ic', 'ic_t_stat'])
        if len(valid) > 0:
            x_array = valid['ic'].to_numpy().flatten()
            y_array = valid['ic_t_stat'].to_numpy().flatten()
            axes[0, 1].scatter(x_array, y_array, alpha=0.6, s=30, color='navy')
            axes[0, 1].axhline(2, color='red', linestyle='--', linewidth=2)
            axes[0, 1].axhline(-2, color='red', linestyle='--', linewidth=2)
            axes[0, 1].set_xlabel('IC Value')
            axes[0, 1].set_ylabel('T-Statistic')
            axes[0, 1].set_title('IC vs T-Statistic')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 子图 3: Top 20 因子 IC 条形图
        top_20 = ic_table.head(20)
        if len(top_20) > 0:
            ic_vals = top_20['ic'].to_numpy().flatten()
            colors = ['green' if x > 0 else 'red' for x in ic_vals]
            axes[1, 0].barh(range(len(top_20)), ic_vals, color=colors)
            axes[1, 0].set_yticks(range(len(top_20)))
            axes[1, 0].set_yticklabels(top_20['factor'].str.slice(-20), fontsize=8)
            axes[1, 0].set_xlabel('IC Value')
            axes[1, 0].set_title('Top 20 Factors by IC')
            axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # 子图 4: 因子缺失率分布
        # missing_data = ic_table['missing_ratio'].dropna()
        # if len(missing_data) > 0:
        #     missing_array = missing_data.to_numpy().flatten()
        #     axes[1, 1].hist(missing_array, bins=30, edgecolor='black', alpha=0.7, color='orange')
        #     axes[1, 1].set_xlabel('Missing Ratio')
        #     axes[1, 1].set_ylabel('Frequency')
        #     axes[1, 1].set_title('Factor Missing Data Ratio')
        #     axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_ic_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    📈 保存 IC 分析图：{symbol}_ic_analysis.png")
    
    def plot_factor_timeseries(self, symbol: str, factor_names: list = None):
        """因子时间序列图"""
        if factor_names is None:
            factor_names = [
                'mid_basis', 'obi_zscore_5', 'basis_zscore_10', 
                'composite_signal', 'lead_lag_aggregate_lag', 'lead_lag_strength'
            ]
        
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
    
    def plot_lead_lag_analysis(self, symbol: str):
        """Lead-Lag 效应专项分析图"""
        lead_lag_factors = [
            'lead_lag_aggregate_lag', 'lead_lag_aggregate_corr',
            'lead_lag_direction', 'lead_lag_strength', 'lead_lag_consistency'
        ]
        
        available = [f for f in lead_lag_factors if f in self.df.columns]
        if len(available) < 2:
            print(f"    ⚠️ Lead-Lag 因子不足，跳过专项分析图")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
        # 子图 1: 最优滞后时间序列
        if 'lead_lag_aggregate_lag' in self.df.columns:
            data = self.df['lead_lag_aggregate_lag'].dropna()
            if len(data) > 0:
                axes[0, 0].plot(data.index, data.to_numpy(), linewidth=0.5, color='blue')
                axes[0, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)
                axes[0, 0].set_title('Aggregate Optimal Lag (Lead-Lag)')
                axes[0, 0].set_ylabel('Lag (ticks)')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].fill_between(data.index, -2, 2, alpha=0.2, color='gray', label='No Lead-Lag')
                axes[0, 0].legend()
        
        # 子图 2: Lead-Lag 相关性
        if 'lead_lag_aggregate_corr' in self.df.columns:
            data = self.df['lead_lag_aggregate_corr'].dropna()
            if len(data) > 0:
                axes[0, 1].plot(data.index, data.to_numpy(), linewidth=0.5, color='green')
                axes[0, 1].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Strong')
                axes[0, 1].axhline(0.3, color='orange', linestyle='--', alpha=0.5, label='Medium')
                axes[0, 1].set_title('Lead-Lag Correlation Strength')
                axes[0, 1].set_ylabel('Correlation')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].legend()
        
        # 子图 3: Lead-Lag 方向分布
        if 'lead_lag_direction' in self.df.columns:
            data = self.df['lead_lag_direction'].dropna()
            if len(data) > 0:
                directions = data.to_numpy()
                pos_ratio = np.sum(directions > 0) / len(directions)
                neg_ratio = np.sum(directions < 0) / len(directions)
                zero_ratio = np.sum(directions == 0) / len(directions)
                
                axes[1, 0].bar(['Swap Leads', 'No Lead', 'Spot Leads'], 
                              [pos_ratio, zero_ratio, neg_ratio],
                              color=['green', 'gray', 'red'])
                axes[1, 0].set_title('Lead-Lag Direction Distribution')
                axes[1, 0].set_ylabel('Proportion')
                for i, v in enumerate([pos_ratio, zero_ratio, neg_ratio]):
                    axes[1, 0].text(i, v + 0.02, f'{v:.2%}', ha='center')
        
        # 子图 4: Lead-Lag 强度
        if 'lead_lag_strength' in self.df.columns:
            data = self.df['lead_lag_strength'].dropna()
            if len(data) > 0:
                axes[1, 1].hist(data.to_numpy(), bins=50, edgecolor='black', alpha=0.7, color='purple')
                axes[1, 1].set_title('Lead-Lag Strength Distribution')
                axes[1, 1].set_xlabel('Strength')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].grid(True, alpha=0.3)
        
        # 子图 5: 多尺度滞后对比
        scale_lags = [f'lead_lag_D{i}_optimal_lag' for i in range(1, 5) if f'lead_lag_D{i}_optimal_lag' in self.df.columns]
        if len(scale_lags) >= 2:
            for lag_col in scale_lags[:4]:
                data = self.df[lag_col].dropna()
                if len(data) > 0:
                    axes[2, 0].plot(data.index[-500:], data.to_numpy()[-500:], linewidth=0.5, label=lag_col)
            axes[2, 0].set_title('Multi-Scale Optimal Lag Comparison')
            axes[2, 0].set_ylabel('Lag')
            axes[2, 0].legend(fontsize=6)
            axes[2, 0].grid(True, alpha=0.3)
        
        # 子图 6: Lead-Lag 一致性
        if 'lead_lag_consistency' in self.df.columns:
            data = self.df['lead_lag_consistency'].dropna()
            if len(data) > 0:
                axes[2, 1].plot(data.index, data.to_numpy(), linewidth=0.5, color='brown')
                axes[2, 1].axhline(0.8, color='green', linestyle='--', alpha=0.5, label='High')
                axes[2, 1].axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='Medium')
                axes[2, 1].set_title('Multi-Scale Lead-Lag Consistency')
                axes[2, 1].set_ylabel('Consistency')
                axes[2, 1].grid(True, alpha=0.3)
                axes[2, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_lead_lag_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    📈 保存 Lead-Lag 分析图：{symbol}_lead_lag_analysis.png")
    
    def plot_correlation_heatmap(self, symbol: str, top_n: int = 100):
        """因子相关性热力图"""
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
        self.plot_lead_lag_analysis(symbol)
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
    print(f"  📥 加载原始数据：{len(full_df)} 条记录")
    
    # ============================
    # ✅ 新增：重采样到 500ms 并 ffill 填充
    # ============================
    print(f"  🔄 重采样到 {config.RESAMPLE_FREQUENCY} 频率...")
    
    # 保存原始列用于 ffill
    price_cols = ['spot_bid1_px', 'spot_ask1_px', 'swap_bid1_px', 'swap_ask1_px']
    other_cols = ['funding_rate', 'index_price']
    
    # 1. 对价格列使用最近值重采样 (保持最新价格)
    resampled_df = full_df[price_cols].resample(config.RESAMPLE_FREQUENCY).last()
    
    # 2. 对其他列使用均值重采样
    for col in other_cols:
        if col in full_df.columns:
            resampled_df[col] = full_df[col].resample(config.RESAMPLE_FREQUENCY).mean()
    
    # 3. 使用前向填充避免 NaN
    resampled_df = resampled_df.ffill()
    
    # 4. 对开头的 NaN 使用后向填充
    resampled_df = resampled_df.bfill()
    
    # 5. 删除仍有 NaN 的行
    resampled_df = resampled_df.dropna()
    
    # 计算重采样统计
    original_duration = (full_df.index[-1] - full_df.index[0]).total_seconds()
    resampled_duration = (resampled_df.index[-1] - resampled_df.index[0]).total_seconds()
    
    if original_duration > 0:
        original_freq = len(full_df) / original_duration
        resampled_freq = len(resampled_df) / resampled_duration
    else:
        original_freq = 0
        resampled_freq = 0
    
    print(f"  ✅ 重采样后数据：{len(resampled_df)} 条记录 ({config.RESAMPLE_FREQUENCY})")
    print(f"     原始频率：{original_freq:.2f} Hz")
    print(f"     重采样频率：{resampled_freq:.2f} Hz")
    print(f"     数据压缩率：{1 - len(resampled_df)/len(full_df):.1%}")
    
    full_df = resampled_df
    
    # 计算高频因子
    factor_engine = HighFrequencyFactorEngine(full_df)
    factor_engine.compute_all_factors()
    factor_df = factor_engine.get_factor_dataframe()
    
    # 因子分析
    analyzer = FactorAnalyzer(factor_df, target_col='basis_ret_1')
    analyzer.compute_target(horizon=5)
    ic_table = analyzer.analyze_all_factors()
    top_factors = analyzer.get_top_factors(n=50, min_abs_ic=0.02)
    corr_matrix = analyzer.calculate_factor_correlation()
    
    # 保存因子数据
    symbol_output_dir = config.OUTPUT_DIR / symbol
    symbol_output_dir.mkdir(parents=True, exist_ok=True)
    
    factor_df_reset = factor_df.reset_index()
    factor_df_reset['year_month'] = factor_df_reset['timestamp'].dt.to_period('M')
    
    for period, group in factor_df_reset.groupby('year_month'):
        year = period.year
        month = str(period.month).zfill(2)
        filename = f"{symbol}_factors_{year}-{month}_{config.RESAMPLE_FREQUENCY}.csv.gz"
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
        'resample_freq': config.RESAMPLE_FREQUENCY,
        'status': 'success'
    }
    
    # Lead-Lag 因子统计
    ll_factors = [c for c in factor_df.columns if 'lead_lag' in c]
    if ll_factors:
        summary['n_lead_lag_factors'] = len(ll_factors)
        ll_ic_avg = ic_table[ic_table['factor'].isin(ll_factors)]['ic'].abs().mean() if not ic_table.empty else 0
        summary['lead_lag_avg_ic'] = ll_ic_avg
    
    print(f"\n📋 摘要:")
    print(f"     记录数：{summary['total_records']}")
    print(f"     因子数：{summary['n_factors']}")
    print(f"     平均|IC|: {summary['avg_ic']:.4f}")
    print(f"     最大|IC|: {summary['max_ic']:.4f}")
    
    if 'n_lead_lag_factors' in summary:
        print(f"     Lead-Lag 因子数：{summary['n_lead_lag_factors']}")
        print(f"     Lead-Lag 平均|IC|: {summary.get('lead_lag_avg_ic', 0):.4f}")
    
    if not top_factors.empty:
        print(f"     Top 因子：{top_factors['factor'].iloc[0]} (IC={top_factors['ic'].iloc[0]:.4f})")
    
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
        print("\n🏆 Top 5 交易对 (按平均 IC):")
        for _, row in top_symbols.iterrows():
            print(f"   {row['symbol']}: 平均 IC={row['avg_ic']:.4f}, 因子数={row['n_factors']}")
    
    # Lead-Lag 因子表现
    if 'lead_lag_avg_ic' in summary_df.columns:
        top_ll = summary_df.nlargest(5, 'lead_lag_avg_ic')
        print("\n🏆 Top 5 交易对 (按 Lead-Lag IC):")
        for _, row in top_ll.iterrows():
            print(f"   {row['symbol']}: Lead-Lag IC={row['lead_lag_avg_ic']:.4f}")
    
    print(f"\n💾 汇总报告保存至：{config.ANALYSIS_DIR / 'all_symbols_summary.csv'}")

# ============================
# 主程序入口
# ============================
if __name__ == "__main__":
    print("="*60)
    print("🚀 高频因子提取与分析系统 (含 MODWT Lead-Lag)")
    print("="*60)
    print(f"📁 输入目录：{config.INPUT_BASE}")
    print(f"📁 输出目录：{config.OUTPUT_DIR}")
    print(f"📁 分析目录：{config.ANALYSIS_DIR}")
    print(f"📅 日期范围：{config.START_DATE} ~ {config.END_DATE}")
    print(f"🌊 MODWT: {'✅ 可用' if PYWT_AVAILABLE else '❌ 不可用'}")
    print(f"🔄 重采样：{config.RESAMPLE_FREQUENCY}")
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
    # symbol = 'AVAXUSDT'
    # symbol = 'ADAUSDT'
    # symbol = 'ETHUSDT'
    # summary = process_symbol(symbol, config)
    # all_summaries.append(summary)

    generate_summary_report(all_summaries, config)
    
    print("\n" + "="*60)
    print("🎉 高频因子提取与分析完成!")
    print("="*60)