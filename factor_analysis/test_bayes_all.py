#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Model Factor Testing Script (Time-Series Split Version)
支持 Transformer、Linear Regression、Logistic Regression、LightGBM 测试
新增：Bayes推断结果、小波变换系数可视化
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import pickle
import json
import argparse
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, r2_score
from scipy import stats
# 可视化
import matplotlib.pyplot as plt
import seaborn as sns
# 小波变换
import pywt
# 尝试导入 LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ 警告：lightgbm 未安装，将跳过 LightGBM 模型")
# 本地模块
from transformer import FactorTransformer

# ============================
# 配置区域
# ============================
class TestConfig:
    FACTOR_DIR = Path("./datasets/factors/hf_factors")
    MODEL_DIR = Path("./datasets/model_training")
    OUTPUT_DIR = Path("./datasets/model_testing")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SEQ_LENGTH = 60
    PREDICTION_HORIZON = 1
    BATCH_SIZE = 64
    VB_NUM_SAMPLES = 50  # Bayes采样数
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = TestConfig()

# （TestFactorSequenceDataset 和 TestFactorFlatDataset 类保持不变，略）

# ============================
# 数据集类 (与 Train 脚本保持一致接口)
# ============================
class TestFactorSequenceDataset(Dataset):
    """测试因子序列数据集 (Transformer)"""
    def __init__(self, factor_df: pd.DataFrame, seq_length: int, prediction_horizon: int,
                 target_col: str, target_col_original: str,
                 scaler=None, target_mean=None, target_std=None,
                 train_feature_names=None):
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        self.target_col = target_col
        self.target_col_original = target_col_original
        self.scaler = scaler
        self.target_mean = target_mean
        self.target_std = target_std
        
        if train_feature_names is not None:
            self.factor_cols = train_feature_names
            print(f"  📋 使用训练配置中的 {len(self.factor_cols)} 个特征列")
        else:
            exclude_cols = [target_col, target_col_original, 'target', 'timestamp', 'year_month', 'timestampes']
            self.factor_cols = [c for c in factor_df.columns if c not in exclude_cols]
            print(f"  ⚠️ 未提供训练特征名，从数据中推断 {len(self.factor_cols)} 个特征列")
        
        # 确保测试数据包含所有训练时的特征列
        factor_data_raw = pd.DataFrame(index=factor_df.index)
        missing_cols = []
        for col in self.factor_cols:
            if col in factor_df.columns:
                factor_data_raw[col] = factor_df[col]
            else:
                missing_cols.append(col)
                factor_data_raw[col] = 0.0
        if missing_cols:
            print(f"  ⚠️ 警告：测试数据缺少 {len(missing_cols)} 个训练时的特征列，已用 0 填充")
        
        # 数据清洗
        factor_data_raw = factor_data_raw.replace([np.inf, -np.inf], np.nan)
        for col in self.factor_cols:
            if col in factor_data_raw.columns:
                lower = factor_data_raw[col].quantile(0.01)
                upper = factor_data_raw[col].quantile(0.99)
                factor_data_raw[col] = factor_data_raw[col].clip(lower, upper)
        factor_data_raw = factor_data_raw.fillna(0)
        factor_data_raw = factor_data_raw[self.factor_cols]
        
        # 标准化 (使用训练集的 Scaler)
        if scaler is not None:
            self.factor_data = scaler.transform(factor_data_raw.values)
        else:
            self.factor_data = factor_data_raw.values
        
        # 目标变量处理
        if target_col not in factor_df.columns:
            print(f"  ⚠️ 警告：测试数据缺少训练时的目标列 {target_col}，将使用 {target_col_original} 计算收益率作为目标")
            self.targets = factor_df[target_col_original].shift(-prediction_horizon).pct_change(prediction_horizon).values
        else:
            self.targets = factor_df[target_col].values
        
        self.targets = np.nan_to_num(self.targets, nan=0.0, posinf=0.0, neginf=0.0)
        if target_mean is not None and target_std is not None:
            self.targets = (self.targets - target_mean) / (target_std + 1e-10)
            self.targets = np.clip(self.targets, -5, 5)
        
        self.timestamps = factor_df.index.values if hasattr(factor_df, 'index') else None
        self.valid_indices = self._get_valid_indices()
        print(f"  📊 测试数据集：{len(self.valid_indices)} 个有效序列")

    def _get_valid_indices(self):
        valid = []
        for i in range(len(self.factor_data) - self.seq_length - self.prediction_horizon):
            seq = self.factor_data[i:i+self.seq_length]
            target_idx = i + self.seq_length + self.prediction_horizon - 1
            if np.isnan(seq).sum() / seq.size > 0.3:
                continue
            if target_idx >= len(self.targets):
                continue
            if np.isnan(self.targets[target_idx]) or np.isinf(self.targets[target_idx]):
                continue
            valid.append(i)
        return valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.seq_length
        target_idx = end_idx + self.prediction_horizon - 1
        seq = self.factor_data[start_idx:end_idx]
        target_reg = self.targets[target_idx]
        target_cls = 1 if target_reg > 0 else 0
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.FloatTensor(seq), torch.FloatTensor([target_reg])[0], torch.LongTensor([target_cls])[0], start_idx

    def get_factor_names(self):
        return self.factor_cols

    def get_timestamps(self, indices):
        if self.timestamps is None:
            return None
        return self.timestamps[indices]

class TestFactorFlatDataset(Dataset):
    """测试扁平化数据集 (Sklearn/LightGBM)"""
    def __init__(self, factor_df: pd.DataFrame, seq_length: int, prediction_horizon: int,
                 target_col: str, target_col_original: str,
                 scaler=None, target_mean=None, target_std=None,
                 train_feature_names=None):
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        self.target_col = target_col
        self.target_col_original = target_col_original
        self.scaler = scaler
        self.target_mean = target_mean
        self.target_std = target_std
        
        if train_feature_names is not None:
            self.factor_cols = train_feature_names
            print(f"  📋 使用训练配置中的 {len(self.factor_cols)} 个特征列")
        else:
            exclude_cols = [target_col, target_col_original, 'target', 'timestamp', 'year_month', 'timestampes']
            self.factor_cols = [c for c in factor_df.columns if c not in exclude_cols]
            print(f"  ⚠️ 未提供训练特征名，从数据中推断 {len(self.factor_cols)} 个特征列")
        
        factor_data_raw = pd.DataFrame(index=factor_df.index)
        missing_cols = []
        for col in self.factor_cols:
            if col in factor_df.columns:
                factor_data_raw[col] = factor_df[col]
            else:
                missing_cols.append(col)
                factor_data_raw[col] = 0.0
        if missing_cols:
            print(f"  ⚠️ 警告：测试数据缺少 {len(missing_cols)} 个训练时的特征列，已用 0 填充")
        
        factor_data_raw = factor_data_raw.replace([np.inf, -np.inf], np.nan)
        for col in self.factor_cols:
            if col in factor_data_raw.columns:
                lower = factor_data_raw[col].quantile(0.01)
                upper = factor_data_raw[col].quantile(0.99)
                factor_data_raw[col] = factor_data_raw[col].clip(lower, upper)
        factor_data_raw = factor_data_raw.fillna(0)
        factor_data_raw = factor_data_raw[self.factor_cols]
        
        if scaler is not None:
            self.factor_data = scaler.transform(factor_data_raw.values)
        else:
            self.factor_data = factor_data_raw.values
        
        if target_col not in factor_df.columns:
            print(f"  ⚠️ 警告：测试数据缺少训练时的目标列 {target_col}，将使用 {target_col_original} 计算收益率作为目标")
            self.targets = factor_df[target_col_original].shift(-prediction_horizon).pct_change(prediction_horizon).values
        else:
            self.targets = factor_df[target_col].values
        
        self.targets = np.nan_to_num(self.targets, nan=0.0, posinf=0.0, neginf=0.0)
        if target_mean is not None and target_std is not None:
            self.targets = (self.targets - target_mean) / (target_std + 1e-10)
            self.targets = np.clip(self.targets, -5, 5)
        
        self.timestamps = factor_df.index.values if hasattr(factor_df, 'index') else None
        self.valid_indices = self._get_valid_indices()
        print(f"  📊 测试数据集：{len(self.valid_indices)} 个有效样本")

    def _get_valid_indices(self):
        valid = []
        for i in range(len(self.factor_data) - self.prediction_horizon):
            target_idx = i + self.prediction_horizon
            if target_idx >= len(self.targets):
                continue
            if np.isnan(self.targets[target_idx]) or np.isinf(self.targets[target_idx]):
                continue
            valid.append(i)
        return valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        idx = self.valid_indices[idx]
        features = self.factor_data[idx]
        target_reg = self.targets[idx + self.prediction_horizon]
        target_cls = 1 if target_reg > 0 else 0
        return torch.FloatTensor(features), torch.FloatTensor([target_reg])[0], torch.LongTensor([target_cls])[0], idx

    def get_factor_names(self):
        return self.factor_cols

    def get_numpy_data(self):
        X, indices = [], []
        for idx in self.valid_indices:
            X.append(self.factor_data[idx])
            indices.append(idx)
        return np.array(X), np.array(indices)

    def get_timestamps(self, indices):
        if self.timestamps is None:
            return None
        return self.timestamps[indices]
# ============================
# Bayes 模型类（与训练脚本一致）
# ============================
class VariationalBayesianLayer(nn.Module):
    """变分贝叶斯层"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x, sample=False):
        if sample:
            weight = self.weight_mu + torch.exp(0.5 * self.weight_logvar) * torch.randn_like(self.weight_mu)
            bias = self.bias_mu + torch.exp(0.5 * self.bias_logvar) * torch.randn_like(self.bias_mu)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)
    
    def get_kl_divergence(self):
        kl_weight = -0.5 * torch.sum(1 + self.weight_logvar - self.weight_mu.pow(2) - self.weight_logvar.exp())
        kl_bias = -0.5 * torch.sum(1 + self.bias_logvar - self.bias_mu.pow(2) - self.bias_logvar.exp())
        return kl_weight + kl_bias

class BayesianFactorTransformer(nn.Module):
    """带Bayes推断的因子Transformer"""
    def __init__(self, n_factors, d_model=128, nhead=4, num_layers=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(n_factors, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.bayes_output = VariationalBayesianLayer(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sample=False):
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        output = self.bayes_output(x, sample=sample)
        return output
    
    def get_kl_divergence(self):
        return self.bayes_output.get_kl_divergence()

# ============================
# 测试可视化器（增强版）
# ============================
class TestingVisualizer:
    """测试可视化（增强版：支持Bayes和小波）"""
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_bayes_predictions(self, y_true: np.ndarray, y_pred_mean: np.ndarray, 
                               y_pred_std: np.ndarray, symbol: str, model_type: str, 
                               target_col: str, timestamps=None):
        """Bayes推断结果可视化（带不确定性区间）"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 预测值与真实值对比（带置信区间）
        ax1 = axes[0]
        ax1.plot(timestamps[:len(y_true)] if timestamps is not None else range(len(y_true)), 
                y_true, label='True Values', linewidth=1.5, color='blue', alpha=0.8)
        ax1.plot(timestamps[:len(y_pred_mean)] if timestamps is not None else range(len(y_pred_mean)), 
                y_pred_mean, label='Predicted Mean', linewidth=1.5, color='orange', alpha=0.8)
        
        # 95% 置信区间
        lower_bound = y_pred_mean - 1.96 * y_pred_std
        upper_bound = y_pred_mean + 1.96 * y_pred_std
        ax1.fill_between(
            timestamps[:len(y_pred_mean)] if timestamps is not None else range(len(y_pred_mean)),
            lower_bound, upper_bound, alpha=0.3, color='orange', label='95% Confidence Interval'
        )
        
        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_ylabel('Value (Increment)', fontsize=12)
        ax1.set_title(f'{symbol} ({model_type}): Bayes Predictions with Uncertainty', fontsize=14)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 不确定性随时间变化
        ax2 = axes[1]
        ax2.plot(timestamps[:len(y_pred_std)] if timestamps is not None else range(len(y_pred_std)), 
                y_pred_std, linewidth=1.5, color='red', alpha=0.8, label='Uncertainty (Std)')
        ax2.axhline(y=np.mean(y_pred_std), color='green', linestyle='--', 
                   label=f'Mean Uncertainty: {np.mean(y_pred_std):.6f}')
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Uncertainty', fontsize=12)
        ax2.set_title('Prediction Uncertainty Over Time', fontsize=14)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_{model_type}_bayes_predictions_{target_col}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存Bayes预测图：{symbol}_{model_type}_bayes_predictions_{target_col}.png")
    
    def plot_wavelet_coefficients(self, data: np.ndarray, symbol: str, model_type: str, 
                                target_col: str, wavelet='db4', level=4):
        """小波变换系数可视化"""
        # 执行小波分解
        coeffs = pywt.wavedec(data, wavelet, level=level)
        
        fig, axes = plt.subplots(level + 2, 1, figsize=(14, 12))
        
        # 原始信号
        axes[0].plot(data, linewidth=1, color='blue')
        axes[0].set_ylabel('Original Signal')
        axes[0].set_title(f'{symbol} ({model_type}): Original Signal and Wavelet Decomposition', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # 近似系数
        axes[1].plot(coeffs[0], linewidth=1, color='green')
        axes[1].set_ylabel(f'A{level}')
        axes[1].grid(True, alpha=0.3)
        
        # 细节系数
        for i in range(1, len(coeffs)):
            axes[i + 1].plot(coeffs[i], linewidth=1, color='orange', alpha=0.7)
            axes[i + 1].set_ylabel(f'D{level - i + 1}')
            axes[i + 1].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time Step', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_{model_type}_wavelet_coeffs_{target_col}.png', 
                dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存小波系数图：{symbol}_{model_type}_wavelet_coeffs_{target_col}.png")
        
        # 🔧 修复：分别保存每个系数，避免形状不一致问题
        save_dict = {
            'wavelet': wavelet,
            'level': level,
            'n_coeffs': len(coeffs)
        }
        for i, coeff in enumerate(coeffs):
            save_dict[f'coeff_{i}'] = coeff
            save_dict[f'coeff_{i}_shape'] = np.array([len(coeff)])
        
        np.savez(self.output_dir / f'{symbol}_{model_type}_wavelet_coeffs_{target_col}.npz',
                **save_dict)
        print(f"  💾 保存小波系数数据：{symbol}_{model_type}_wavelet_coeffs_{target_col}.npz")
    
    def plot_true_vs_pred_scatter(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  symbol: str, model_type: str, target_col: str):
        # （原有实现保持不变）
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        axes[0].scatter(y_true, y_pred, alpha=0.3, s=15, color='steelblue', edgecolors='none')
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        if len(y_true) > 2:
            coef = np.polyfit(y_true, y_pred, 1)
            poly = np.poly1d(coef)
            axes[0].plot([min_val, max_val], poly([min_val, max_val]), 'g-', linewidth=1.5,
                        label=f'Fit: y={coef[0]:.3f}x+{coef[1]:.3f}')
        
        ic, _ = stats.spearmanr(y_true, y_pred)
        try:
            pc, _ = stats.pearsonr(y_true[~np.isnan(y_pred)], y_pred[~np.isnan(y_true)])
        except:
            pc = np.nan
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        stats_text = f'IC(Spearman)={ic:.4f}\nR(Pearson)={pc:.4f}\nRMSE={rmse:.6f}\nR²={r2:.4f}'
        axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[0].set_xlabel('True Values (y_true)', fontsize=12)
        axes[0].set_ylabel('Predicted Values (y_pred)', fontsize=12)
        axes[0].set_title(f'{symbol} ({model_type}): True vs Predicted', fontsize=14)
        axes[0].legend(loc='lower right', fontsize=9)
        axes[0].grid(True, alpha=0.3)
        
        y_true_shifted = np.roll(y_true, 1)
        axes[1].scatter(y_true_shifted, y_true, alpha=0.3, s=15, color='blue', edgecolors='none')
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        axes[1].set_xlabel('True Values Shifted (y_true.shift(1))', fontsize=12)
        axes[1].set_ylabel('True Values (y_true)', fontsize=12)
        axes[1].set_title(f'{symbol} ({model_type}): True vs Previous True', fontsize=14)
        axes[1].legend(loc='lower right', fontsize=9)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_{model_type}_true_vs_pred_scatter_{target_col}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存散点图：{symbol}_{model_type}_true_vs_pred_scatter_{target_col}.png")
    
    def plot_true_pred_timeseries(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  target_col: str, timestamps=None, symbol: str = None,
                                  model_type: str = 'transformer', max_points: int = 200000):
        # （原有实现保持不变，略）
        pass
    def generate_all_test_plots(self, symbol: str, model_type: str, test_results: dict, 
                            timestamps=None, target_col: str = 'spot_mid', use_bayes: bool = False):
        """生成所有测试集可视化图表（增强版）"""
        print("  🎨 生成测试集可视化图表...")
        
        y_true = test_results['targets']
        y_pred = test_results['predictions']
        quantile_returns = test_results.get('quantile_returns', None)
        
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]
        
        if len(y_true_clean) < 10:
            print(f"  ⚠️ 有效样本不足，跳过可视化")
            return
        
        # 标准可视化
        self.plot_true_vs_pred_scatter(y_true_clean, y_pred_clean, symbol, model_type, target_col)
        self.plot_true_pred_timeseries(y_true_clean, y_pred_clean, target_col, timestamps, symbol, model_type)
        
        # Bayes 特定可视化
        if use_bayes and 'predictions_std' in test_results:
            y_pred_std = test_results['predictions_std'][valid_mask]
            self.plot_bayes_predictions(y_true_clean, y_pred_clean, y_pred_std, 
                                    symbol, model_type, target_col, 
                                    timestamps[valid_mask] if timestamps is not None else None)
            
            # 🔧 修复：确保残差是一维数组
            residuals = y_pred_clean - y_true_clean
            residuals = residuals[~np.isnan(residuals)]  # 移除 NaN
            
            if len(residuals) > 10:
                self.plot_wavelet_coefficients(residuals, symbol, model_type, target_col)
            else:
                print(f"  ⚠️ 残差样本不足，跳过小波分析")
        
        print("  ✅ 测试可视化完成")

# ============================
# Bayes 测试器
# ============================
class BayesianModelTester:
    def __init__(self, model, model_type: str, config: TestConfig):
        self.model = model
        self.model_type = model_type
        self.config = config
        if model_type == 'transformer':
            self.model = self.model.to(config.DEVICE)
            self.model.eval()
    
    def _flatten_predictions(self, pred_tensor):
        """🔧 确保预测张量是一维的"""
        if pred_tensor.dim() == 2:
            return pred_tensor.squeeze(-1)
        elif pred_tensor.dim() == 1:
            return pred_tensor
        else:
            return pred_tensor.view(-1)
    
    def _calculate_ic(self, preds, targets):
        preds = np.array(preds)
        targets = np.array(targets)
        valid_mask = ~(np.isnan(preds) | np.isnan(targets))
        preds, targets = preds[valid_mask], targets[valid_mask]
        if len(preds) < 10:
            return 0.0
        ic, _ = stats.spearmanr(preds, targets)
        return ic if not np.isnan(ic) else 0.0
    
    def _calculate_uncertainty(self, preds_samples):
        if len(preds_samples) < 2:
            return np.zeros_like(preds_samples[0])
        preds_stack = np.stack(preds_samples, axis=0)
        return np.std(preds_stack, axis=0)
    
    def evaluate_bayesian_transformer(self, test_loader, target_mean=None, target_std=None):
        all_preds_mean, all_preds_std, all_targets, all_indices = [], [], [], []
        
        with torch.no_grad():
            for seq, target_reg, target_cls, indices in test_loader:
                seq = seq.to(self.config.DEVICE)
                target_reg = target_reg.to(self.config.DEVICE)
                
                # 🔧 确保 target_reg 是一维的
                if target_reg.dim() == 2:
                    target_reg = target_reg.squeeze(-1)
                
                pred_samples = []
                for _ in range(self.config.VB_NUM_SAMPLES):
                    pred_reg = self.model(seq, sample=True)
                    pred_reg = self._flatten_predictions(pred_reg)
                    pred_samples.append(pred_reg.detach())
                
                pred_mean = torch.stack(pred_samples).mean(0)
                pred_mean = self._flatten_predictions(pred_mean)
                
                pred_samples_np = [p.cpu().numpy().flatten() for p in pred_samples]
                pred_std = self._calculate_uncertainty(pred_samples_np)
                
                target_np = target_reg.cpu().numpy().flatten()
                pred_np = pred_mean.cpu().numpy().flatten()
                
                valid_mask = ~(np.isnan(pred_np) | np.isnan(target_np))
                
                if valid_mask.sum() > 0:
                    all_preds_mean.extend(pred_np[valid_mask].tolist())
                    all_preds_std.extend(pred_std[valid_mask].tolist())
                    all_targets.extend(target_np[valid_mask].tolist())
                    all_indices.extend(indices.cpu().numpy()[valid_mask].tolist())
        
        if len(all_preds_mean) < 10:
            print(f"  ⚠️ 有效预测不足")
            return None
        
        all_preds_mean = np.array(all_preds_mean)
        all_preds_std = np.array(all_preds_std)
        all_targets = np.array(all_targets)
        
        if target_mean is not None and target_std is not None:
            all_preds_original = all_preds_mean * target_std + target_mean
            all_targets_original = all_targets * target_std + target_mean
            all_preds_std_original = all_preds_std * target_std
        else:
            all_preds_original = all_preds_mean
            all_targets_original = all_targets
            all_preds_std_original = all_preds_std
        
        ic = self._calculate_ic(all_preds_mean, all_targets)
        direction_acc = accuracy_score(np.sign(all_preds_mean), np.sign(all_targets))
        
        return {
            'ic': ic,
            'direction_accuracy': direction_acc,
            'predictions': all_preds_original,
            'predictions_std': all_preds_std_original,
            'targets': all_targets_original,
            'indices': all_indices
        }

# （原有的 ModelTester 类保持不变）
# ============================
# 测试器
# ============================
class ModelTester:
    def __init__(self, model, model_type: str, config: TestConfig):
        self.model = model
        self.model_type = model_type
        self.config = config
        if model_type == 'transformer':
            self.model = self.model.to(config.DEVICE)
            self.model.eval()

    def _calculate_ic(self, preds, targets):
        preds = np.array(preds)
        targets = np.array(targets)
        valid_mask = ~(np.isnan(preds) | np.isnan(targets))
        preds, targets = preds[valid_mask], targets[valid_mask]
        if len(preds) < 10:
            return 0.0
        ic, _ = stats.spearmanr(preds, targets)
        return ic if not np.isnan(ic) else 0.0

    def evaluate_transformer(self, test_loader, target_mean=None, target_std=None):
        all_preds, all_targets, all_indices = [], [], []
        with torch.no_grad():
            for seq, target_reg, target_cls, indices in test_loader:
                seq = seq.to(self.config.DEVICE)
                target_reg = target_reg.to(self.config.DEVICE)
                target_cls = target_cls.to(self.config.DEVICE)
                pred_reg, pred_cls = self.model(seq)
                if torch.isnan(pred_reg).any() or torch.isinf(pred_reg).any():
                    continue
                pred_np = pred_reg.detach().cpu().numpy()
                target_np = target_reg.cpu().numpy()
                valid_mask = ~(np.isnan(pred_np) | np.isnan(target_np))
                if valid_mask.sum() > 0:
                    all_preds.extend(pred_np[valid_mask].tolist())
                    all_targets.extend(target_np[valid_mask].tolist())
                    all_indices.extend(indices.cpu().numpy().tolist())
        
        if len(all_preds) < 10:
            print(f"  ⚠️ 有效预测不足")
            return None
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        if target_mean is not None and target_std is not None:
            all_preds_original = all_preds * target_std + target_mean
            all_targets_original = all_targets * target_std + target_mean
        else:
            all_preds_original = all_preds
            all_targets_original = all_targets
        
        ic = self._calculate_ic(all_preds, all_targets)
        direction_acc = accuracy_score(np.sign(all_preds), np.sign(all_targets))
        quantile_returns = self._quantile_backtest(all_preds_original, all_targets_original)
        
        return {
            'ic': ic,
            'direction_accuracy': direction_acc,
            'quantile_returns': quantile_returns,
            'predictions': all_preds_original,
            'targets': all_targets_original,
            'indices': all_indices
        }

    def evaluate_sklearn(self, test_dataset, target_mean=None, target_std=None):
        X_test, indices = test_dataset.get_numpy_data()
        if self.model_type == 'linear':
            all_preds = self.model['reg_model'].predict(X_test)
        elif self.model_type == 'logistic':
            all_preds = self.model['cls_model'].predict_proba(X_test)[:, 1]
            all_preds = (all_preds - 0.5) * 2 * 0.1
        elif self.model_type == 'lightgbm':
            if LIGHTGBM_AVAILABLE:
                all_preds = self.model['reg_model'].predict(X_test, num_threads=1)
            else:
                raise ImportError("LightGBM 未安装")
        
        all_targets = []
        for idx in indices:
            all_targets.append(test_dataset.targets[idx + test_dataset.prediction_horizon])
        all_targets = np.array(all_targets)
        
        if target_mean is not None and target_std is not None:
            all_preds_original = all_preds * target_std + target_mean
            all_targets_original = all_targets * target_std + target_mean
        else:
            all_preds_original = all_preds
            all_targets_original = all_targets
        
        ic = self._calculate_ic(all_preds, all_targets)
        direction_acc = accuracy_score(np.sign(all_preds), np.sign(all_targets))
        quantile_returns = self._quantile_backtest(all_preds_original, all_targets_original)
        
        return {
            'ic': ic,
            'direction_accuracy': direction_acc,
            'quantile_returns': quantile_returns,
            'predictions': all_preds_original,
            'targets': all_targets_original,
            'indices': indices.tolist()
        }

    def _quantile_backtest(self, preds, targets, n_quantiles=5):
        try:
            valid_mask = ~(np.isnan(preds) | np.isnan(targets))
            preds, targets = preds[valid_mask], targets[valid_mask]
            if len(preds) < n_quantiles:
                return {0: 0, 1: 0}
            quantiles = pd.qcut(preds, n_quantiles, labels=False, duplicates='drop')
            df = pd.DataFrame({'quantile': quantiles, 'return': targets})
            group_returns = df.groupby('quantile')['return'].mean()
            return group_returns.to_dict()
        except Exception as e:
            print(f"    ⚠️ 分层回测失败：{e}")
            return {0: 0, 1: 0}

# ============================
# 主测试流程
# ============================
def test_symbol(symbol: str, config: TestConfig, model_type: str = 'transformer', use_bayes: bool = False) -> dict:
    print(f"\n{'='*60}")
    print(f"🧪 测试交易对：{symbol} (模型：{model_type}, Bayes: {use_bayes})")
    print(f"{'='*60}")
    
    # 1. 加载训练配置
    model_dir = config.MODEL_DIR / symbol / model_type
    if not model_dir.exists():
        print(f"  ❌ 模型目录不存在：{model_dir}")
        return {'status': 'failed', 'reason': 'no_model'}
    
    config_file = model_dir / 'train_config.json'
    if not config_file.exists():
        print(f"  ❌ 配置文件不存在：{config_file}")
        return {'status': 'failed', 'reason': 'no_config'}
    
    with open(config_file, 'r') as f:
        train_config = json.load(f)
    
    print(f"  📋 加载训练配置：{train_config['symbol']} ({train_config['model_type']})")
    print(f"  📊 训练使用的目标列：{train_config.get('target_col', 'spot_mid')}")
    print(f"  🔮 使用Bayes推断：{train_config.get('use_bayes', False)}")
    
    use_bayes = train_config.get('use_bayes', use_bayes)
    target_col = train_config.get('target_col', 'spot_mid')
    target_col_original = train_config.get('target_col_original', 'spot_mid')
    train_feature_names = train_config.get('factor_names', None)
    
    train_end_timestamp_str = train_config.get('train_end_timestamp', None)
    if train_end_timestamp_str:
        print(f"  📅 训练截止于：{train_end_timestamp_str}")
        train_end_timestamp = pd.to_datetime(train_end_timestamp_str)
    else:
        print(f"  ⚠️ 配置中未找到 train_end_timestamp")
        train_end_timestamp = None
    
    # 2. 加载模型
    if model_type == 'transformer':
        model_file = model_dir / 'best_model.pth'
        if not model_file.exists():
            return {'status': 'failed', 'reason': 'no_model_file'}
        
        checkpoint = torch.load(model_file, map_location=config.DEVICE, weights_only=False)
        is_bayesian = checkpoint.get('is_bayesian', False)
        use_bayes = is_bayesian or use_bayes
        
        if use_bayes:
            model = BayesianFactorTransformer(
                n_factors=train_config['n_factors'],
                d_model=train_config['hidden_dim'],
                nhead=train_config['num_heads'],
                num_layers=train_config['num_layers'],
                dropout=train_config['dropout']
            )
            print(f"  ✅ 加载 Bayes Transformer 模型")
        else:
            model = FactorTransformer(
                n_factors=train_config['n_factors'],
                d_model=train_config['hidden_dim'],
                nhead=train_config['num_heads'],
                num_layers=train_config['num_layers'],
                dropout=train_config['dropout']
            )
            print(f"  ✅ 加载 Transformer 模型")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(config.DEVICE)
        model.eval()
        print(f"  ✅ 最佳验证 IC: {checkpoint['val_ic']:.4f}")
    else:
        # 非Transformer模型加载（略）
        pass
    
    # 3. 加载标准化器
    scaler_file = model_dir / 'scaler.pkl'
    if scaler_file.exists():
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
    else:
        scaler = None
    
    target_mean = train_config.get('target_mean', None)
    target_std = train_config.get('target_std', None)
    
    # 4. 加载测试数据
    symbol_factor_dir = config.FACTOR_DIR / symbol
    if not symbol_factor_dir.exists():
        return {'status': 'failed', 'reason': 'no_factor_data'}
    
    factor_files = list(symbol_factor_dir.glob("*.csv.gz"))
    if not factor_files:
        return {'status': 'failed', 'reason': 'no_factor_files'}
    
    print(f"  📥 加载 {len(factor_files)} 个因子文件...")
    all_dfs = []
    for f in factor_files:
        try:
            df = pd.read_csv(f, compression='gzip')
            time_cols = ['timestamp', 'timestampes', 'time']
            time_col_found = None
            for col in time_cols:
                if col in df.columns:
                    time_col_found = col
                    break
            if time_col_found:
                try:
                    df[time_col_found] = pd.to_datetime(df[time_col_found], format='ISO8601', utc=True)
                except:
                    df[time_col_found] = pd.to_datetime(df[time_col_found])
                df = df.set_index(time_col_found)
            all_dfs.append(df)
        except Exception as e:
            print(f"  ⚠️ 读取失败 {f}: {e}")
            continue
    
    if not all_dfs:
        return {'status': 'failed', 'reason': 'no_valid_data'}
    
    full_df = pd.concat(all_dfs, ignore_index=False)
    full_df = full_df.sort_index()
    print(f"  ✅ 加载 {len(full_df)} 条总记录")
    
    # 过滤测试数据
    if train_end_timestamp is not None:
        if not isinstance(full_df.index, pd.DatetimeIndex):
            full_df.index = pd.to_datetime(full_df.index)
        test_df = full_df[full_df.index > train_end_timestamp].copy()
        print(f"  ✅ 过滤后测试集记录：{len(test_df)} 条 (训练后数据)")
        if len(test_df) == 0:
            print(f"  ❌ 测试集为空")
            return {'status': 'failed', 'reason': 'empty_test_set'}
    else:
        test_df = full_df
        print(f"  ⚠️ 使用全部数据作为测试集")
    
    # 5. 创建测试数据集
    if model_type == 'transformer':
        test_dataset = TestFactorSequenceDataset(
            test_df, seq_length=config.SEQ_LENGTH, prediction_horizon=config.PREDICTION_HORIZON,
            target_col=train_config['target_col'],
            target_col_original=train_config.get('target_col_original', train_config['target_col']),
            scaler=scaler,
            target_mean=target_mean, target_std=target_std,
            train_feature_names=train_feature_names
        )
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    else:
        test_dataset = TestFactorFlatDataset(
            test_df, seq_length=config.SEQ_LENGTH, prediction_horizon=config.PREDICTION_HORIZON,
            target_col=train_config['target_col'],
            target_col_original=train_config.get('target_col_original', train_config['target_col']),
            scaler=scaler,
            target_mean=target_mean, target_std=target_std,
            train_feature_names=train_feature_names
        )
        test_loader = test_dataset
    
    if len(test_dataset) < 10:
        print(f"  ❌ 有效测试序列不足：{len(test_dataset)}")
        return {'status': 'failed', 'reason': 'insufficient_data'}
    
    test_timestamps = test_dataset.get_timestamps(test_dataset.valid_indices)
    
    # 6. 测试评估
    print(f"\n📊 测试集评估...")
    if use_bayes and model_type == 'transformer':
        tester = BayesianModelTester(model, model_type, config)
        test_results = tester.evaluate_bayesian_transformer(test_loader, target_mean, target_std)
    else:
        tester = ModelTester(model, model_type, config)
        if model_type == 'transformer':
            test_results = tester.evaluate_transformer(test_loader, target_mean, target_std)
        else:
            test_results = tester.evaluate_sklearn(test_dataset, target_mean, target_std)
    
    if test_results is None:
        return {'status': 'failed', 'reason': 'evaluation_failed'}
    
    # 7. 保存结果
    symbol_output_dir = config.OUTPUT_DIR / symbol / model_type
    symbol_output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame({
        'prediction': test_results['predictions'],
        'target': test_results['targets']
    })
    
    if use_bayes and 'predictions_std' in test_results:
        results_df['prediction_std'] = test_results['predictions_std']
    
    results_df.to_csv(symbol_output_dir / f'{symbol}_predictions.csv', index=False)
    
    # 8. 生成可视化
    visualizer = TestingVisualizer(symbol_output_dir)
    visualizer.generate_all_test_plots(
        symbol, model_type, test_results, test_timestamps, 
        target_col, use_bayes=use_bayes
    )
    
    # 9. 生成摘要
    summary = {
        'symbol': symbol,
        'model_type': model_type,
        'use_bayes': use_bayes,
        'n_samples': len(test_dataset),
        'n_factors': train_config['n_factors'],
        'test_ic': test_results['ic'],
        'direction_accuracy': test_results['direction_accuracy'],
        'train_best_val_ic': train_config['best_val_ic'],
        'status': 'success'
    }
    
    if use_bayes and 'predictions_std' in test_results:
        summary['mean_uncertainty'] = float(np.mean(test_results['predictions_std']))
    
    print(f"\n📋 测试摘要:")
    print(f"      测试 IC: {summary['test_ic']:.4f}")
    print(f"      方向准确率：{summary['direction_accuracy']:.4f}")
    print(f"      训练最佳 IC: {summary['train_best_val_ic']:.4f}")
    if use_bayes and 'mean_uncertainty' in summary:
        print(f"      平均不确定性：{summary['mean_uncertainty']:.6f}")
    
    return summary

# （discover_symbols, generate_summary_report 函数保持不变）

def discover_symbols(config: TestConfig) -> list:
    if not config.MODEL_DIR.exists():
        raise FileNotFoundError(f"模型目录不存在：{config.MODEL_DIR}")
    symbols = [d.name for d in config.MODEL_DIR.iterdir() if d.is_dir()]
    print(f"🔍 发现 {len(symbols)} 个交易对有训练模型")
    return symbols

def generate_summary_report(summaries: list, config: TestConfig):
    print(f"\n{'='*60}")
    print("📊 生成测试汇总报告")
    print(f"{'='*60}")
    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df[summary_df['status'] == 'success']
    if summary_df.empty:
        print("  ❌ 无成功测试的交易对")
        return
    summary_df.to_csv(config.OUTPUT_DIR / "all_symbols_testing.csv", index=False)
    for model_type in summary_df['model_type'].unique():
        model_df = summary_df[summary_df['model_type'] == model_type]
        top_by_ic = model_df.nlargest(5, 'test_ic')
        print(f"\n🏆 Top 5 交易对 ({model_type.upper()} - 按测试 IC):")
        for _, row in top_by_ic.iterrows():
            print(f"   {row['symbol']}: IC={row['test_ic']:.4f}, 准确率={row['direction_accuracy']:.4f}")
    print(f"\n💾 汇总报告：{config.OUTPUT_DIR / 'all_symbols_testing.csv'}")

# ============================
# 主程序入口
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='多模型因子测试脚本')
    parser.add_argument('--symbol', type=str, default='ADAUSDT', help='交易对名称')
    parser.add_argument('--model', type=str, default='transformer',
                       choices=['transformer', 'linear', 'logistic', 'lightgbm'],
                       help='模型类型')
    parser.add_argument('--all_models', action='store_true', help='测试所有模型')
    parser.add_argument('--bayes', action='store_true', help='启用Bayes推断')
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 多模型高频因子测试脚本 (时间序列分割版)")
    print("="*60)
    print(f"📁 因子目录：{config.FACTOR_DIR}")
    print(f"📁 模型目录：{config.MODEL_DIR}")
    print(f"📁 输出目录：{config.OUTPUT_DIR}")
    print(f"🧠 设备：{config.DEVICE}")
    print(f"📦 模型：{args.model}")
    print(f"🔮 Bayes推断：{args.bayes}")
    print("="*60)
    
    symbols = discover_symbols(config)
    if not symbols:
        print("❌ 未发现任何训练好的模型")
        exit(1)
    
    all_summaries = []
    if args.all_models:
        models = ['transformer', 'linear', 'logistic', 'lightgbm']
    else:
        models = [args.model]
    
    for model_type in models:
        if model_type == 'lightgbm' and not LIGHTGBM_AVAILABLE:
            print(f"⚠️ 跳过 {model_type} (未安装 lightgbm)")
            continue
        try:
            summary = test_symbol(args.symbol, config, model_type, use_bayes=True)
            all_summaries.append(summary)
        except Exception as e:
            print(f"❌ {args.symbol} ({model_type}) 测试失败：{e}")
            import traceback
            traceback.print_exc()
            all_summaries.append({'symbol': args.symbol, 'model_type': model_type, 
                                 'status': 'failed', 'error': str(e)})
    
    generate_summary_report(all_summaries, config)
    print("\n" + "="*60)
    print("🎉 多模型因子测试完成!")
    print("="*60)