# src/KronosPredictor.py
'''
Docstring for KronosPredictor
封装Kronos模型的预测功能，提供便捷的接口进行时间序列预测。
Inputs:
- tokenizer_path: 预训练的Kronos分词器路径
- predictor_path: 预训练的Kronos预测器路径
- device: 计算设备（CPU或GPU）
- max_context: 最大上下文长度
Outputs:
- KronosPredictor类实例，包含predict方法用于生成预测序列

Predictor类方法:
- predict: 基于输入序列和时间特征生成预测序列
'''

import os
# Fix OMP warning
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import torch

# Add project root
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from config import Config
from model.kronos import Kronos, KronosTokenizer
from model.kronos import sample_from_logits
from statsmodels.stats.weightstats import DescrStatsW

LOOKBACK = Config().lookback
PRED_LENGTH = Config().pred_length
N_SAMPLES = Config().n_samples
TOKENIZER_PATH_10min = Config().tokenizer_10min
PREDICTOR_PATH_10min = Config().predictor_10min

class KronosPredictor:
    """Kronos预测器封装"""
    def __init__(self, 
                 tokenizer_path, 
                 predictor_path, 
                 device=None, 
                 max_context=2048):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = KronosTokenizer.from_pretrained(tokenizer_path).to(device).eval()
        self.model = Kronos.from_pretrained(predictor_path).to(device).eval()
        self.device = device
        self.max_context = max_context
    
    def predict(self, x, x_stamp, y_stamp, pred_len=1, T=1.0, top_p=0.9, top_k=0):
        """生成预测序列
        Inputs:
        - x: 输入特征序列 (seq_len, feature_dim)
        - x_stamp: 输入时间特征序列 (seq_len, time_feature_dim)
        - y_stamp: 预测时间特征序列 (pred_len, time_feature_dim)
        - pred_len: 预测长度
        - T: 采样温度
        - top_p: top-p采样参数
        - top_k: top-k采样参数
        Outputs:
        - pred: 预测序列 (pred_len, feature_dim)
        """
        with torch.no_grad():
            x = torch.from_numpy(x).unsqueeze(0).to(self.device)
            x_stamp = torch.from_numpy(x_stamp).unsqueeze(0).to(self.device)
            y_stamp = torch.from_numpy(y_stamp).unsqueeze(0).to(self.device)
            x_token = self.tokenizer.encode(x, half=True)
            initial_seq_len = x.size(1)
            batch_size = x_token[0].size(0)
            total_seq_len = initial_seq_len + pred_len
            full_stamp = torch.cat([x_stamp, y_stamp], dim=1)
            generated_pre = x_token[0].new_empty(batch_size, pred_len)
            generated_post = x_token[1].new_empty(batch_size, pred_len)
            pre_buffer = x_token[0].new_zeros(batch_size, self.max_context)
            post_buffer = x_token[1].new_zeros(batch_size, self.max_context)
            buffer_len = min(initial_seq_len, self.max_context)
            if buffer_len > 0:
                start_idx = max(0, initial_seq_len - self.max_context)
                pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
                post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]
            for i in range(pred_len):
                current_seq_len = initial_seq_len + i
                window_len = min(current_seq_len, self.max_context)
                if current_seq_len <= self.max_context:
                    input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
                else:
                    input_tokens = [pre_buffer, post_buffer]
                context_end = current_seq_len
                context_start = max(0, context_end - self.max_context)
                current_stamp = full_stamp[:, context_start:context_end, :].contiguous()
                s1_logits, context = self.model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
                s1_logits = s1_logits[:, -1, :]
                sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)
                s2_logits = self.model.decode_s2(context, sample_pre)
                s2_logits = s2_logits[:, -1, :]
                sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)
                generated_pre[:, i] = sample_pre.squeeze(-1)
                generated_post[:, i] = sample_post.squeeze(-1)
                if current_seq_len < self.max_context:
                    pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                    post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
                else:
                    pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                    post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                    pre_buffer[:, -1] = sample_pre.squeeze(-1)
                    post_buffer[:, -1] = sample_post.squeeze(-1)
            full_pre = torch.cat([x_token[0], generated_pre], dim=1)
            full_post = torch.cat([x_token[1], generated_post], dim=1)
            context_start = max(0, total_seq_len - self.max_context)
            input_tokens = [full_pre[:, context_start:total_seq_len].contiguous(), full_post[:, context_start:total_seq_len].contiguous()]
            z = self.tokenizer.decode(input_tokens, half=True)
            return z[0, -pred_len:, :].cpu().numpy()

class DynamicSignalGenerator:
    """
    Dynamic策略信号生成器
    负责基于Kronos预测生成动态交易阈值参数
    支持动态lookback：当历史数据不足时自动调整
    """
    def __init__(self, 
                 predictor: KronosPredictor,
                 lookback=144,      # 24小时10分钟K线
                 pred_length=48,    # 预测8小时
                 n_samples=20,
                 temperature = 1.0):
        '''初始化DynamicSignalGenerator实例
        Inputs:
        - predictor: KronosPredictor实例
        - lookback: 期望的历史数据长度（最大值）
        - pred_length: 预测长度（固定不变）
        - n_samples: 预测样本数量
        Outputs:
        - DynamicSignalGenerator类实例
        '''
        self.predictor = predictor
        self.lookback = lookback
        self.pred_length = pred_length
        self.n_samples = n_samples
        self.current_params = None  # [c_tt_high, c_tt_low, c_mt_high, c_mt_low, c_tm_high, c_tm_low]
        self.pred_sequences = None
        self.estimates = []
        self.pred_weights = None
        self.sigma = 1e-4  # 观测噪声标准差
        self.temperature = temperature

    def resample_to_10min(self, df_100ms):
        """将100ms数据重采样为10分钟K线，使用正确的volume和amount定义"""
        if df_100ms.empty:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'amount'])
        
        # 确保所有需要的列都存在
        required_cols = []
        for prefix in ['spot', 'swap']:
            for level in range(3):  # 前3档
                required_cols.extend([f"{prefix}_bid{level}_amount", f"{prefix}_ask{level}_amount"])
            required_cols.extend([f"{prefix}_bid0_price", f"{prefix}_ask0_price"])
        required_cols.extend(['basis1_price', 'basis2_price', 'basis1_volume', 'basis2_volume'])
        required_cols.extend(['basis_mid_price', 'index_price', 'funding_rate'])

        for col in required_cols:
            if col not in df_100ms.columns:
                df_100ms[col] = np.nan
        
        # 移除全 NaN 的行
        df_clean = df_100ms.dropna(subset=required_cols, how='all')
        
        if df_clean.empty:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'amount'])
        
        # 重采样为10分钟
        resampled = df_clean.resample('10min')
        
        # 计算 Open/Close (使用 spot bid0)
        open_prices = resampled['basis_mid_price'].first()
        close_prices = resampled['basis_mid_price'].last()
        
        # 计算 High/Low (取 spot bid0, spot ask0, swap bid0, swap ask0 的极值)
        high_prices = resampled['basis1_price'].quantile(0.95, interpolation='nearest')
        low_prices = resampled['basis2_price'].quantile(0.05, interpolation='nearest')

        # 计算 Volume (使用 funding_rate 作为权重)
        volume_series = resampled['funding_rate'].last()
        # 计算 Amount
        spot_mid_price = (resampled['spot_bid0_price'].mean() + resampled['spot_ask0_price'].mean()) / 2
        amount_series = np.log(spot_mid_price) - np.log(resampled['index_price'].mean())
        
        # 合并结果
        df_10min = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume_series,
            'amount': amount_series
        })
        
        # 确保索引一致
        df_10min.index = open_prices.index
        
        # 强制转换为数值类型并移除 NaN
        for col in df_10min.columns:
            df_10min[col] = pd.to_numeric(df_10min[col], errors='coerce')
        
        df_10min = df_10min.dropna()
        return df_10min

    def generate_initial_signal(self, 
                               df_10min: pd.DataFrame, 
                               current_time: pd.Timestamp,
                               feature_list: list,
                               time_features: list) -> list:
        """
        基于历史数据生成初始动态阈值信号
        支持动态lookback：当历史K线不足self.lookback时，使用全部可用数据
        保持pred_length不变（预测未来固定长度）
        
        返回: [c_tt_high, c_tt_low, c_mt_high, c_mt_low, c_tm_high, c_tm_low]
        """
        # 1. 重采样数据
        # df_10min = self.resample_to_10min(df_100ms)
        # if df_10min.empty:
        #     # 无数据时返回默认参数
        #     return self.current_params or [0.01, -0.01, 0.008, -0.008, 0.009, -0.009]
        
        # 2. 动态确定可用的历史数据长度（最多self.lookback个）
        available_history = len(df_10min)
        actual_lookback = min(available_history, self.lookback)
        
        print(f"[INFO] 可用历史K线: {available_history}, 实际使用: {actual_lookback} (目标: {self.lookback}), "
              f"预测长度: {self.pred_length}")
        
        # 3. 准备预测输入
        # x_df: 使用全部可用历史数据（最多lookback个），不包含未来数据
        x_df = df_10min[-actual_lookback:].copy()
        
        # y_df: 生成未来pred_length个时间戳（仅用于提供时间特征，无实际值）
        last_timestamp = x_df.index[-1]
        future_timestamps = [last_timestamp + pd.Timedelta(minutes=10 * (i + 1)) 
                            for i in range(self.pred_length)]
        y_df = pd.DataFrame(index=future_timestamps)
        
        # 添加时间特征
        x_df = x_df.assign(
            minute=x_df.index.minute,
            hour=x_df.index.hour,
            weekday=x_df.index.weekday,
            day=x_df.index.day,
            month=x_df.index.month
        )
        y_df = y_df.assign(
            minute=y_df.index.minute,
            hour=y_df.index.hour,
            weekday=y_df.index.weekday,
            day=y_df.index.day,
            month=y_df.index.month
        )
        
        # 特征归一化（仅基于x_df的实际数据）
        x = x_df[feature_list].values.astype(np.float32)
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x_norm = (x - x_mean) / (x_std + 1e-5)
        x_norm = np.clip(x_norm, -5.0, 5.0)
        x_stamp = x_df[time_features].values.astype(np.float32)
        y_stamp = y_df[time_features].values.astype(np.float32)

        # 4. 多样本预测
        preds = []
        pred_sequences = []
        for n in range(self.n_samples):
            pred = self.predictor.predict(
                x=x_norm,
                x_stamp=x_stamp,
                y_stamp=y_stamp,
                pred_len=self.pred_length,
                T=self.temperature/(n+1)**0.5,
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

        # 5. 计算动态阈值
        preds = np.array(preds)
        high_mean = np.mean(preds[:, 1])
        high_std = np.std(preds[:, 1])
        low_mean = np.mean(preds[:, 2])
        low_std = np.std(preds[:, 2])
        
        high_estimate = high_mean + high_std
        low_estimate = low_mean - low_std
        self.estimates = [high_mean, high_std, low_mean, low_std]
        self.estimates_last = self.estimates.copy()
        
        # 调整阈值中点
        if self.current_params:
            c_tt_high, c_tt_low, c_mt_high, c_mt_low, c_tm_high, c_tm_low = self.current_params
            d = self._calculate_adjustment(
                high_estimate, low_estimate, 
                c_mt_high, c_mt_low,
                x[-actual_lookback:, 1].mean(),  # 使用实际可用数据
                x[-actual_lookback:, 2].mean()
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
        prior = self.pred_sequences[:, 0, 3]  # 预测的Close[3]; Open[0], High[1], Low[2], Close[3], Volume[4], Amount[5]
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

        self.estimates = [high.mean, high.std, low.mean, low.std]

        # 3.1. 重新计算长期阈值
        high_last = DescrStatsW(self.pred_sequences[:, -1, 1], weights=self.pred_weights)
        low_last = DescrStatsW(self.pred_sequences[:, -1, 2], weights=self.pred_weights)
        high_estimate = high_last.mean + high_last.std
        low_estimate = low_last.mean - low_last.std

        self.estimates_last = [high_last.mean, high_last.std, low_last.mean, low_last.std]

        
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
    
    def update_signal_with_full_observations(self, 
                                      observations: float,
                                      timestamp: pd.Timestamp) -> list:
        """
        基于最新观测价格对预测序列重加权，更新动态阈值
        self.pred_sequences: (N_SAMPLES, pred_length, feature_dim)
        observations: (pred_length,feature_dim)
        """
        if self.pred_sequences is None or self.pred_sequences.shape[1] == 0:
            return self.current_params
        feature_list = ['open', 'high', 'low', 'close', 'volume', 'amount']
        # 1. 重加权
        # prior = self.pred_sequences[:, 0, 3]  # 预测的Close[3]; Open[0], High[1], Low[2], Close[3], Volume[4], Amount[5]
        # residuals = observations - prior
        # likelihoods = np.exp(-0.5 * (residuals / self.sigma) ** 2) / (np.sqrt(2 * np.pi) * self.sigma)
        # unnormalized = self.pred_weights * likelihoods
        # weight_sum = np.sum(unnormalized)
        
        logweights = np.zeros((N_SAMPLES, len(feature_list))) 
        for f in range(len(feature_list)):
            vals = self.pred_sequences[:, 0, f]
            logweights[:, f] = -0.5 * ((observations[0, f] - vals) / self.sigma)**2
        # 综合所有特征的权重
        unnormalized = np.exp(logweights.sum(axis=1) - np.max(logweights.sum(axis=1)))
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

        self.estimates = [high.mean, high.std, low.mean, low.std]

        # 3.1. 重新计算长期阈值
        high_last = DescrStatsW(self.pred_sequences[:, -1, 1], weights=self.pred_weights)
        low_last = DescrStatsW(self.pred_sequences[:, -1, 2], weights=self.pred_weights)
        high_estimate = high_last.mean + high_last.std
        low_estimate = low_last.mean - low_last.std

        self.estimates_last = [high_last.mean, high_last.std, low_last.mean, low_last.std]

        
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