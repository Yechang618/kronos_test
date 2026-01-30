# backtest/signal_generator.py
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.stats.weightstats import DescrStatsW
from model.kronos import Kronos, KronosTokenizer, sample_from_logits
import torch

class KronosPredictor:
    """Kronos预测器封装"""
    def __init__(self, tokenizer_path, predictor_path, device=None, max_context=2048):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = KronosTokenizer.from_pretrained(tokenizer_path).to(device).eval()
        self.model = Kronos.from_pretrained(predictor_path).to(device).eval()
        self.device = device
        self.max_context = max_context

    def predict(self, x, x_stamp, y_stamp, pred_len=1, T=1.0, top_p=0.9, top_k=0):
        # ... (保留原predict逻辑)
        pass

class DynamicSignalGenerator:
    """
    Dynamic策略信号生成器
    负责基于Kronos预测生成动态交易阈值参数
    """
    def __init__(self, 
                 predictor: KronosPredictor,
                 lookback=144,      # 24小时10分钟K线
                 pred_length=48,    # 预测8小时
                 n_samples=100):
        self.predictor = predictor
        self.lookback = lookback
        self.pred_length = pred_length
        self.n_samples = n_samples
        self.current_params = None  # [c_tt_high, c_tt_low, c_mt_high, c_mt_low, c_tm_high, c_tm_low]
        self.pred_sequences = None
        self.pred_weights = None
        self.sigma = 1.0  # 观测噪声标准差

    def resample_to_10min(self, df_100ms: pd.DataFrame) -> pd.DataFrame:
        """将100ms数据重采样为10分钟K线"""
        # ... (保留原resample_to_10min逻辑，移除self引用)
        pass

    def generate_initial_signal(self, 
                               df_100ms: pd.DataFrame, 
                               current_time: pd.Timestamp,
                               feature_list: list,
                               time_features: list) -> list:
        """
        基于历史数据生成初始动态阈值信号
        返回: [c_tt_high, c_tt_low, c_mt_high, c_mt_low, c_tm_high, c_tm_low]
        """
        # 1. 重采样数据
        df_10min = self.resample_to_10min(df_100ms)
        if df_10min.empty or len(df_10min) < self.lookback + self.pred_length:
            # 数据不足时使用理论中点
            if self.current_params:
                mid = df_10min['close'].mean() if not df_10min.empty else 0
                c_tt_high, c_tt_low, c_mt_high, c_mt_low, c_tm_high, c_tm_low = self.current_params
                d = mid - (c_mt_high + c_mt_low) / 2
                return [c + d for c in self.current_params]
            return self.current_params or [0.01, -0.01, 0.008, -0.008, 0.009, -0.009]

        # 2. 准备预测输入
        x_df = df_10min[-self.lookback-self.pred_length:-self.pred_length].copy()
        y_df = df_10min[-self.pred_length:].copy()
        
        # 添加时间特征
        for col in time_features:
            if col == 'minute':
                x_df[col] = x_df.index.minute
                y_df[col] = y_df.index.minute
            # ... 其他时间特征
        
        # 特征归一化
        x = x_df[feature_list].values.astype(np.float32)
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x_norm = (x - x_mean) / (x_std + 1e-5)
        x_norm = np.clip(x_norm, -5.0, 5.0)
        x_stamp = x_df[time_features].values.astype(np.float32)
        y_stamp = y_df[time_features].values.astype(np.float32)

        # 3. 多样本预测
        preds = []
        pred_sequences = []
        for _ in range(self.n_samples):
            pred = self.predictor.predict(
                x=x_norm,
                x_stamp=x_stamp,
                y_stamp=y_stamp,
                pred_len=self.pred_length,
                T=0.6,
                top_p=0.9,
                top_k=0
            )
            # 反归一化
            for j in range(pred.shape[0]):
                pred[j, :] = pred[j, :] * (x_std + 1e-5) + x_mean
            preds.append(pred[-1])
            pred_sequences.append(pred)
        
        self.pred_sequences = np.array(pred_sequences)
        self.pred_weights = np.ones(self.n_samples) / self.n_samples

        # 4. 计算动态阈值
        preds = np.array(preds)
        high_mean = np.mean(preds[:, 1])
        high_std = np.std(preds[:, 1])
        low_mean = np.mean(preds[:, 2])
        low_std = np.std(preds[:, 2])
        
        high_estimate = high_mean + high_std
        low_estimate = low_mean - low_std
        
        # 调整阈值中点
        if self.current_params:
            c_tt_high, c_tt_low, c_mt_high, c_mt_low, c_tm_high, c_tm_low = self.current_params
            d = self._calculate_adjustment(
                high_estimate, low_estimate, 
                c_mt_high, c_mt_low,
                x[-self.lookback:, 1].mean(), 
                x[-self.lookback:, 2].mean()
            )
            self.current_params = [
                c_tt_high + d, c_tt_low + d,
                c_mt_high + d, c_mt_low + d,
                c_tm_high + d, c_tm_low + d
            ]
        else:
            # 初始化参数（基于预测范围）
            spread = high_estimate - low_estimate
            mid = (high_estimate + low_estimate) / 2
            self.current_params = [
                mid + spread * 0.3, mid - spread * 0.3,  # TT
                mid + spread * 0.2, mid - spread * 0.2,  # MT
                mid + spread * 0.25, mid - spread * 0.25 # TM
            ]
        
        return self.current_params.copy()

    def update_signal_with_observation(self, 
                                      observed_price: float,
                                      timestamp: pd.Timestamp) -> list:
        """
        基于最新观测价格对预测序列重加权，更新动态阈值
        """
        if self.pred_sequences is None or self.pred_sequences.shape[1] == 0:
            return self.current_params

        # 1. 重加权
        prior = self.pred_sequences[:, 0, 0]  # 预测的Close
        residuals = observed_price - prior
        likelihoods = np.exp(-0.5 * (residuals / self.sigma) ** 2) / (np.sqrt(2 * np.pi) * self.sigma)
        unnormalized = self.pred_weights * likelihoods
        weight_sum = np.sum(unnormalized)
        
        if weight_sum > 0:
            self.pred_weights = unnormalized / weight_sum
        else:
            self.pred_weights = np.ones_like(self.pred_weights) / len(self.pred_weights)
        
        # 2. 移除已观测时间步
        self.pred_sequences = self.pred_sequences[:, 1:, :]
        
        # 3. 重新计算阈值
        high = DescrStatsW(self.pred_sequences[:, 0, 1], weights=self.pred_weights)
        low = DescrStatsW(self.pred_sequences[:, 0, 2], weights=self.pred_weights)
        high_estimate = high.mean + high.std
        low_estimate = low.mean - low.std
        
        if self.current_params:
            c_tt_high, c_tt_low, c_mt_high, c_mt_low, c_tm_high, c_tm_low = self.current_params
            d = self._calculate_adjustment(
                high_estimate, low_estimate, 
                c_mt_high, c_mt_low,
                (high_estimate + low_estimate) / 2, 
                (high_estimate + low_estimate) / 2
            )
            self.current_params = [
                c_tt_high + d, c_tt_low + d,
                c_mt_high + d, c_mt_low + d,
                c_tm_high + d, c_tm_low + d
            ]
        
        return self.current_params.copy()

    def _calculate_adjustment(self, high_est, low_est, c_mt_high, c_mt_low, high_ref, low_ref):
        """计算阈值调整量"""
        if high_est - low_est > c_mt_high - c_mt_low:
            return (high_est + low_est - (c_mt_high + c_mt_low)) / 2
        elif high_est <= c_mt_high and low_est >= c_mt_low:
            return 0.1 * ((high_est + low_est) - (c_mt_high + c_mt_low)) / 2
        elif high_est > c_mt_high and low_est >= c_mt_low:
            return high_est - c_mt_high
        elif high_est <= c_mt_high and low_est < c_mt_low:
            return low_est - c_mt_low
        return 0

    def get_current_thresholds(self):
        """获取当前交易阈值"""
        return {
            'tt_open': self.current_params[0],
            'tt_close': self.current_params[1],
            'mt_open': self.current_params[2],
            'mt_close': self.current_params[3],
            'tm_open': self.current_params[4],
            'tm_close': self.current_params[5]
        }