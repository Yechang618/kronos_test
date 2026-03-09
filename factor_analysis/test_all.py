#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Model Factor Testing Script
支持 Transformer、Linear Regression、Logistic Regression、LightGBM 测试
修复版：修复 LightGBM 线程错误，添加 y_true vs y_pred 可视化
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
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = TestConfig()


# ============================
# 数据集类（修复版）
# ============================
# ============================
# 数据集类（修复版）
# ============================
class TestFactorSequenceDataset(Dataset):
    """测试因子序列数据集 (Transformer)"""
    def __init__(self, factor_df: pd.DataFrame, seq_length: int, prediction_horizon: int,
                 target_col: str, target_col_original: str, 
                 scaler=None, target_mean=None, target_std=None,
                 train_feature_names=None):  # ✅ 新增：训练时的特征名列表
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        self.target_col = target_col
        self.target_col_original = target_col_original
        self.scaler = scaler
        self.target_mean = target_mean
        self.target_std = target_std
        
        # ✅ 修复 1: 使用训练时的特征名列表，而不是从当前数据推断
        if train_feature_names is not None:
            self.factor_cols = train_feature_names
            print(f"  📋 使用训练配置中的 {len(self.factor_cols)} 个特征列")
        else:
            # 向后兼容：如果没有提供特征名，从数据中推断
            exclude_cols = [target_col, target_col_original, 'target', 'timestamp', 'year_month', 'timestampes']
            self.factor_cols = [c for c in factor_df.columns if c not in exclude_cols]
            print(f"  ⚠️ 未提供训练特征名，从数据中推断 {len(self.factor_cols)} 个特征列")
        
        # ✅ 修复 2: 确保测试数据包含所有训练时的特征列
        factor_data_raw = pd.DataFrame(index=factor_df.index)
        missing_cols = []
        for col in self.factor_cols:
            if col in factor_df.columns:
                factor_data_raw[col] = factor_df[col]
            else:
                # 如果测试数据缺少某列，用 0 填充（并记录）
                missing_cols.append(col)
                factor_data_raw[col] = 0.0
        
        if missing_cols:
            print(f"  ⚠️ 警告：测试数据缺少 {len(missing_cols)} 个训练时的特征列，已用 0 填充")
            print(f"     缺失列示例：{missing_cols[:5]}...")
        
        # ✅ 修复 3: 处理 Inf 值
        factor_data_raw = factor_data_raw.replace([np.inf, -np.inf], np.nan)
        
        # ✅ 修复 4: Winsorization 裁剪极端值（与训练时一致）
        for col in self.factor_cols:
            if col in factor_data_raw.columns:
                lower = factor_data_raw[col].quantile(0.01)
                upper = factor_data_raw[col].quantile(0.99)
                factor_data_raw[col] = factor_data_raw[col].clip(lower, upper)
        
        # ✅ 修复 5: 填充 NaN
        factor_data_raw = factor_data_raw.fillna(0)
        
        # ✅ 修复 6: 确保列顺序与训练时一致
        factor_data_raw = factor_data_raw[self.factor_cols]
        
        if scaler is not None:
            self.factor_data = scaler.transform(factor_data_raw.values)
        else:
            self.factor_data = factor_data_raw.values
        
        # 目标变量处理
        if target_col not in factor_df.columns:
            print(f"  {target_col} 列不存在")
            self.targets = factor_df[target_col_original].shift(-prediction_horizon).pct_change(prediction_horizon).values
        else:
            self.targets = factor_df[target_col].values
        
        self.targets = np.nan_to_num(self.targets, nan=0.0, posinf=0.0, neginf=0.0)
        
        if target_mean is not None and target_std is not None:
            self.targets = (self.targets - target_mean) / (target_std + 1e-10)
            self.targets = np.clip(self.targets, -5, 5)
        
        # 保存时间戳
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
                 train_feature_names=None):  # ✅ 新增：训练时的特征名列表
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        self.target_col = target_col
        self.target_col_original = target_col_original
        self.scaler = scaler
        self.target_mean = target_mean
        self.target_std = target_std
        
        # ✅ 修复 1: 使用训练时的特征名列表
        if train_feature_names is not None:
            self.factor_cols = train_feature_names
            print(f"  📋 使用训练配置中的 {len(self.factor_cols)} 个特征列")
        else:
            exclude_cols = [target_col, target_col_original, 'target', 'timestamp', 'year_month', 'timestampes']
            self.factor_cols = [c for c in factor_df.columns if c not in exclude_cols]
            print(f"  ⚠️ 未提供训练特征名，从数据中推断 {len(self.factor_cols)} 个特征列")
        
        # ✅ 修复 2: 确保测试数据包含所有训练时的特征列
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
        
        # ✅ 修复 3: 处理 Inf 值
        factor_data_raw = factor_data_raw.replace([np.inf, -np.inf], np.nan)
        
        # ✅ 修复 4: Winsorization 裁剪极端值
        for col in self.factor_cols:
            if col in factor_data_raw.columns:
                lower = factor_data_raw[col].quantile(0.01)
                upper = factor_data_raw[col].quantile(0.99)
                factor_data_raw[col] = factor_data_raw[col].clip(lower, upper)
        
        # ✅ 修复 5: 填充 NaN
        factor_data_raw = factor_data_raw.fillna(0)
        
        # ✅ 修复 6: 确保列顺序与训练时一致
        factor_data_raw = factor_data_raw[self.factor_cols]
        
        if scaler is not None:
            self.factor_data = scaler.transform(factor_data_raw.values)
        else:
            self.factor_data = factor_data_raw.values
        
        # 目标变量处理
        if target_col not in factor_df.columns:
            self.targets = factor_df[target_col_original].shift(-prediction_horizon).pct_change(prediction_horizon).values
        else:
            self.targets = factor_df[target_col].values
        
        self.targets = np.nan_to_num(self.targets, nan=0.0, posinf=0.0, neginf=0.0)
        
        if target_mean is not None and target_std is not None:
            self.targets = (self.targets - target_mean) / (target_std + 1e-10)
            self.targets = np.clip(self.targets, -5, 5)
        
        # 保存时间戳
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

# class TestFactorFlatDataset(Dataset):
#     """测试扁平化数据集 (Sklearn/LightGBM)"""
#     def __init__(self, factor_df: pd.DataFrame, seq_length: int, prediction_horizon: int,
#                  target_col: str, target_col_original:str, scaler=None, target_mean=None, target_std=None):
#         self.seq_length = seq_length
#         self.prediction_horizon = prediction_horizon
#         self.target_col = target_col
#         self.target_col_original = target_col_original
#         self.scaler = scaler
#         self.target_mean = target_mean
#         self.target_std = target_std
        
#         exclude_cols = [target_col, 'target', 'timestamp', 'year_month', 'timestampes']
#         self.factor_cols = [c for c in factor_df.columns if c not in exclude_cols]
        
#         factor_data_raw = factor_df[self.factor_cols].copy()
        
#         # ✅ 修复 1: 处理 Inf 值
#         factor_data_raw = factor_data_raw.replace([np.inf, -np.inf], np.nan)
        
#         # ✅ 修复 2: Winsorization 裁剪极端值
#         for col in self.factor_cols:
#             if col in factor_data_raw.columns:
#                 lower = factor_data_raw[col].quantile(0.01)
#                 upper = factor_data_raw[col].quantile(0.99)
#                 factor_data_raw[col] = factor_data_raw[col].clip(lower, upper)
        
#         # ✅ 修复 3: 填充 NaN
#         factor_data_raw = factor_data_raw.fillna(0)
        
#         if scaler is not None:
#             self.factor_data = scaler.transform(factor_data_raw)
#         else:
#             self.factor_data = factor_data_raw.values
        
#         # 目标变量处理
#         if target_col not in factor_df.columns:
#             self.targets = factor_df[target_col_original].shift(-prediction_horizon).pct_change(prediction_horizon).values
#         else:
#             self.targets = factor_df[target_col].values
        
#         self.targets = np.nan_to_num(self.targets, nan=0.0, posinf=0.0, neginf=0.0)
        
#         if target_mean is not None and target_std is not None:
#             self.targets = (self.targets - target_mean) / (target_std + 1e-10)
#             self.targets = np.clip(self.targets, -5, 5)
        
#         # 保存时间戳
#         self.timestamps = factor_df.index.values if hasattr(factor_df, 'index') else None
        
#         self.valid_indices = self._get_valid_indices()
#         print(f"  📊 测试数据集：{len(self.valid_indices)} 个有效样本")
    
#     def _get_valid_indices(self):
#         valid = []
#         for i in range(len(self.factor_data) - self.prediction_horizon):
#             target_idx = i + self.prediction_horizon
#             if target_idx >= len(self.targets):
#                 continue
#             if np.isnan(self.targets[target_idx]) or np.isinf(self.targets[target_idx]):
#                 continue
#             valid.append(i)
#         return valid
    
#     def __len__(self):
#         return len(self.valid_indices)
    
#     def __getitem__(self, idx):
#         idx = self.valid_indices[idx]
#         features = self.factor_data[idx]
#         target_reg = self.targets[idx + self.prediction_horizon]
#         target_cls = 1 if target_reg > 0 else 0
#         return torch.FloatTensor(features), torch.FloatTensor([target_reg])[0], torch.LongTensor([target_cls])[0], idx
    
#     def get_factor_names(self):
#         return self.factor_cols
    
#     def get_numpy_data(self):
#         X, indices = [], []
#         for idx in self.valid_indices:
#             X.append(self.factor_data[idx])
#             indices.append(idx)
#         return np.array(X), np.array(indices)
    
#     def get_timestamps(self, indices):
#         if self.timestamps is None:
#             return None
#         return self.timestamps[indices]


# ============================
# ✅ 新增：测试可视化器
# ============================
class TestingVisualizer:
    """测试可视化"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_true_vs_pred_scatter(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   symbol: str, model_type: str, target_col: str):
        """y_true vs y_pred 散点图 + 回归线"""
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Figure 1: y_true vs y_pred 散点图
        # 散点
        axes[0].scatter(y_true, y_pred, alpha=0.3, s=15, color='steelblue', edgecolors='none')
        
        # 对角线 (完美预测)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # 回归线
        if len(y_true) > 2:
            coef = np.polyfit(y_true, y_pred, 1)
            poly = np.poly1d(coef)
            axes[0].plot([min_val, max_val], poly([min_val, max_val]), 'g-', linewidth=1.5, 
                   label=f'Fit: y={coef[0]:.3f}x+{coef[1]:.3f}')
        
        # 统计信息
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
        
        # Figure 2: y_true vs y_true.shift(1) 散点图
        y_true_shifted = np.roll(y_true, 1)
        axes[1].scatter(y_true_shifted, y_true, alpha=0.3, s=15, color='blue', edgecolors='none')
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        axes[1].set_xlabel('True Values Shifted (y_true.shift(1))', fontsize=12)
        axes[1].set_ylabel('True Values (y_true)', fontsize=12)
        axes[1].set_title(f'{symbol} ({model_type}): True vs Previous True', fontsize=14)
        axes[1].legend(loc='lower right', fontsize=9)
        axes[1].grid(True, alpha=0.3)

        # 回归线
        if len(y_true_shifted) > 2:
            coef = np.polyfit(y_true[~np.isnan(y_true_shifted)], y_true_shifted[~np.isnan(y_true_shifted)], 1)
            poly = np.poly1d(coef)
            axes[1].plot([min_val, max_val], poly([min_val, max_val]), 'g-', linewidth=1.5, 
                   label=f'Fit: y={coef[0]:.3f}x+{coef[1]:.3f}')
        
        # 统计信息
        ic, _ = stats.spearmanr(y_true, y_true_shifted)
        try:
            pc, _ = stats.pearsonr(y_true[~np.isnan(y_true_shifted)], y_true_shifted[~np.isnan(y_true_shifted)])
        except:
            pc = np.nan
        mse = mean_squared_error(y_true, y_true_shifted)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_true_shifted)
        
        stats_text = f'IC(Spearman)={ic:.4f}\nR(Pearson)={pc:.4f}\nRMSE={rmse:.6f}\nR²={r2:.4f}'
        axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_{model_type}_true_vs_pred_scatter_{target_col}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存散点图：{symbol}_{model_type}_true_vs_pred_scatter_{target_col}.png")
    
    def plot_true_pred_timeseries(self, y_true: np.ndarray, y_pred: np.ndarray, target_col: str, timestamps = None, symbol: str = None,
                                model_type: str = 'transformer', max_points: int = 2000):
        """y_true 和 y_pred 时间序列对比图"""
        
        # ✅ 修复：验证时间戳长度是否匹配
        if timestamps is not None:
            try:
                timestamps = np.array(timestamps)
                if len(timestamps) != len(y_true):
                    print(f"  ⚠️ 时间戳长度 ({len(timestamps)}) 与数据长度 ({len(y_true)}) 不匹配，使用索引代替")
                    timestamps = None
            except Exception as e:
                print(f"  ⚠️ 时间戳转换失败：{e}，使用索引代替")
                timestamps = None
        
        # 如果数据点太多，采样显示
        if len(y_true) > max_points:
            step = len(y_true) // max_points
            y_true_plot = y_true[::step]
            y_pred_plot = y_pred[::step]
            if timestamps is not None:
                timestamps_plot = timestamps[::step]
            else:
                timestamps_plot = np.arange(len(y_true_plot))
        else:
            y_true_plot = y_true
            y_pred_plot = y_pred
            timestamps_plot = timestamps if timestamps is not None else np.arange(len(y_true))
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # 子图 1: 原始值对比
        axes[0].plot(timestamps_plot, y_true_plot, label='True', linewidth=1, alpha=0.8, color='blue')
        axes[0].plot(timestamps_plot, y_pred_plot, label='Predicted', linewidth=1, alpha=0.8, color='orange')
        axes[0].set_ylabel('Value')
        axes[0].set_title(f'{symbol} ({model_type}): True vs Predicted Time Series')
        axes[0].legend(loc='upper right', fontsize=9)
        axes[0].grid(True, alpha=0.3)
        
        # 子图 2: 误差 (残差)
        residuals = y_pred_plot - y_true_plot
        axes[1].plot(timestamps_plot, residuals, label='Residual (pred - true)', 
                    linewidth=0.5, color='gray', alpha=0.7)
        axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[1].fill_between(timestamps_plot, residuals, 0, 
                            where=(residuals > 0), color='green', alpha=0.2, label='Over-predicted')
        axes[1].fill_between(timestamps_plot, residuals, 0, 
                            where=(residuals < 0), color='red', alpha=0.2, label='Under-predicted')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Residual')
        axes[1].legend(loc='upper right', fontsize=9)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_{model_type}_true_pred_timeseries_{target_col}.png', 
                dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存时间序列对比图：{symbol}_{model_type}_true_pred_timeseries_{target_col}.png")
    
    def plot_residual_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, symbol: str, model_type: str, target_col: str):
        """残差分析图"""
        residuals = y_pred - y_true
        residuals = residuals[~np.isnan(residuals)]
        
        if len(residuals) < 10:
            print(f"  ⚠️ 残差样本不足，跳过残差分析图")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].hist(residuals, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        mu, std = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        from scipy.stats import norm
        axes[0, 0].plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2, label=f'N({mu:.4f}, {std:.4f}²)')
        axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=1)
        axes[0, 0].set_xlabel('Residual')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Residual Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].boxplot(residuals, vert=True, patch_artist=True, 
                       boxprops=dict(facecolor='lightblue', color='blue'))
        axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].set_title('Residual Boxplot')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        try:
            from statsmodels.tsa.stattools import acf
            max_lag = min(50, len(residuals) // 10)
            acf_vals = acf(residuals, nlags=max_lag, fft=True)
            axes[1, 0].stem(range(len(acf_vals)), acf_vals, linefmt='b-', markerfmt='bo', basefmt='gray')
            axes[1, 0].axhline(1.96 / np.sqrt(len(residuals)), color='red', linestyle='--', linewidth=1, label='95% CI')
            axes[1, 0].axhline(-1.96 / np.sqrt(len(residuals)), color='red', linestyle='--', linewidth=1)
            axes[1, 0].set_xlabel('Lag')
            axes[1, 0].set_ylabel('Autocorrelation')
            axes[1, 0].set_title('Residual Autocorrelation')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        except Exception as e:
            axes[1, 0].text(0.5, 0.5, f'ACF failed', ha='center', va='center',
                       transform=axes[1, 0].transAxes, fontsize=8)
            axes[1, 0].set_title('Residual Autocorrelation')
        
        axes[1, 1].scatter(y_pred, residuals, alpha=0.3, s=10, color='gray')
        axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[1, 1].set_xlabel('Predicted Value')
        axes[1, 1].set_ylabel('Residual')
        axes[1, 1].set_title('Residuals vs Predicted')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_{model_type}_residual_analysis_{target_col}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存残差分析图：{symbol}_{model_type}_residual_analysis_{target_col}.png")
    
    def plot_cumulative_returns(self, y_true: np.ndarray, y_pred: np.ndarray, symbol: str, model_type: str, target_col: str):
        """累积收益对比图"""
        pred_signal = np.sign(y_pred)
        true_return = y_true
        
        strategy_returns = pred_signal * true_return
        strategy_cum = np.cumsum(strategy_returns[~np.isnan(strategy_returns)])
        benchmark_cum = np.cumsum(true_return[~np.isnan(true_return)])
        
        time_axis = np.arange(len(strategy_cum))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_axis, strategy_cum, label='Strategy (Pred Signal)', linewidth=2, color='green')
        ax.plot(time_axis[:len(benchmark_cum)], benchmark_cum, label='Benchmark (Buy & Hold)', 
               linewidth=2, color='blue', alpha=0.7)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        
        if len(strategy_returns) > 1:
            valid_returns = strategy_returns[~np.isnan(strategy_returns)]
            if np.std(valid_returns) > 0:
                sharpe = np.mean(valid_returns) / np.std(valid_returns) * np.sqrt(252 * 24 * 3600)
                ax.text(0.02, 0.98, f'Sharpe (ann.): {sharpe:.4f}', 
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Cumulative Return', fontsize=12)
        ax.set_title(f'{symbol} ({model_type}): Cumulative Returns', fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_{model_type}_cumulative_returns_{target_col}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存累积收益图：{symbol}_{model_type}_cumulative_returns_{target_col}.png")
    
    def plot_quantile_returns(self, quantile_returns: dict, symbol: str, model_type: str, target_col: str = 'mid_basis'):
        """分层回测收益图"""
        if not quantile_returns:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        quantiles = list(quantile_returns.keys())
        returns = list(quantile_returns.values())
        colors = ['red' if r < 0 else 'green' for r in returns]
        
        ax.bar(range(len(quantiles)), returns, color=colors, edgecolor='black')
        ax.set_xticks(range(len(quantiles)))
        ax.set_xticklabels([f'Q{i+1}' for i in quantiles])
        ax.set_xlabel('Quantile')
        ax.set_ylabel('Average Return')
        ax.set_title(f'{symbol} ({model_type}): Quantile Returns')
        ax.grid(True, alpha=0.3, axis='y')
        
        if len(returns) >= 2:
            long_short = returns[-1] - returns[0]
            ax.axhline(long_short, color='blue', linestyle='--',
                      label=f'Long-Short: {long_short:.6f}')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_{model_type}_quantile_returns_{target_col}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存分层收益图：{symbol}_{model_type}_quantile_returns_{target_col}.png")
    
    def generate_all_test_plots(self, symbol: str, model_type: str, test_results: dict, timestamps=None, target_col: str = 'mid_basis'):
        """生成全部测试可视化"""
        print("  🎨 生成测试集可视化图表...")
        
        y_true = test_results['targets']
        y_pred = test_results['predictions']
        quantile_returns = test_results.get('quantile_returns', None)
        
        # 清理数据
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]
        
        if len(y_true_clean) < 10:
            print(f"  ⚠️ 有效样本不足，跳过可视化")
            return
        
        self.plot_true_vs_pred_scatter(y_true_clean, y_pred_clean, symbol, model_type, target_col)
        self.plot_true_pred_timeseries(y_true_clean, y_pred_clean, target_col, timestamps, symbol, model_type)
        self.plot_residual_analysis(y_true_clean, y_pred_clean, symbol, model_type, target_col)
        self.plot_cumulative_returns(y_true_clean, y_pred_clean, symbol, model_type, target_col)
        
        if quantile_returns:
            self.plot_quantile_returns(quantile_returns, symbol, model_type)
        
        print("  ✅ 测试可视化完成")


# ============================
# 测试器（修复版）
# ============================
class ModelTester:
    def __init__(self, model, model_type: str, config: TestConfig):
        self.model = model
        self.model_type = model_type
        self.config = config
        
        # ✅ 修复：确保模型在正确设备上
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
                # ✅ 修复：确保所有张量在同一设备
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
        
        # ✅ 修复：LightGBM 使用单线程避免 wmic 错误
        if self.model_type == 'linear':
            all_preds = self.model['reg_model'].predict(X_test)
        elif self.model_type == 'logistic':
            all_preds = self.model['cls_model'].predict_proba(X_test)[:, 1]
            all_preds = (all_preds - 0.5) * 2 * 0.1
        elif self.model_type == 'lightgbm':
            if LIGHTGBM_AVAILABLE:
                # ✅ 关键修复：设置 n_jobs=1 避免 Windows wmic 错误
                all_preds = self.model['reg_model'].predict(X_test, num_threads=1)
            else:
                raise ImportError("LightGBM 未安装")
        
        # 获取目标值
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
# 主测试流程（修复版）
# ============================
def test_symbol(symbol: str, config: TestConfig, model_type: str = 'transformer') -> dict:
    print(f"\n{'='*60}")
    print(f"🧪 测试交易对：{symbol} (模型：{model_type})")
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
    target_col = train_config.get('target_col', 'mid_basis_return')
    target_col_original = train_config.get('target_col_original', 'mid_basis_return')
    # 1. 从训练配置中加载特征名
    train_feature_names = train_config.get('factor_names', None)
    if train_feature_names:
        print(f"  📋 训练时使用了 {len(train_feature_names)} 个特征")
    else:
        print(f"  ⚠️ 训练配置中未找到特征名列表")
    
    # 2. 加载模型
    if model_type == 'transformer':
        model_file = model_dir / 'best_model.pth'
        if not model_file.exists():
            print(f"  ❌ 模型文件不存在：{model_file}")
            return {'status': 'failed', 'reason': 'no_model_file'}
        
        checkpoint = torch.load(model_file, map_location=config.DEVICE, weights_only=False)
        model = FactorTransformer(
            n_factors=train_config['n_factors'],
            d_model=train_config['hidden_dim'],
            nhead=train_config['num_heads'],
            num_layers=train_config['num_layers'],
            dropout=train_config['dropout']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # ✅ 修复：确保模型在正确设备上
        model = model.to(config.DEVICE)
        model.eval()
        
        print(f"  ✅ 加载 Transformer 模型，最佳验证 IC: {checkpoint['val_ic']:.4f}")
        print(f"  🖥️  模型设备：{next(model.parameters()).device}")
    else:
        model_file = model_dir / 'best_model.pkl'
        if not model_file.exists():
            print(f"  ❌ 模型文件不存在：{model_file}")
            return {'status': 'failed', 'reason': 'no_model_file'}
        
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        print(f"  ✅ 加载 {model_type.upper()} 模型，验证 IC: {model['val_ic']:.4f}")
    
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
        print(f"  ❌ 因子目录不存在：{symbol_factor_dir}")
        return {'status': 'failed', 'reason': 'no_factor_data'}
    
    factor_files = list(symbol_factor_dir.glob("*.csv.gz"))
    if not factor_files:
        print(f"  ❌ 无因子文件")
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
    print(f"  ✅ 加载 {len(full_df)} 条测试记录")
    
    # 5. 创建测试数据集
    if model_type == 'transformer':
        test_dataset = TestFactorSequenceDataset(
            full_df, seq_length=config.SEQ_LENGTH, prediction_horizon=config.PREDICTION_HORIZON,
            target_col=train_config['target_col'], 
            target_col_original=train_config.get('target_col_original', train_config['target_col']),
            scaler=scaler,
            target_mean=target_mean, target_std=target_std,
            train_feature_names=train_feature_names  # ✅ 传递训练时的特征名
        )
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    else:
        test_dataset = TestFactorFlatDataset(
            full_df, seq_length=config.SEQ_LENGTH, prediction_horizon=config.PREDICTION_HORIZON,
            target_col=train_config['target_col'],
            target_col_original=train_config.get('target_col_original', train_config['target_col']),
            scaler=scaler,
            target_mean=target_mean, target_std=target_std,
            train_feature_names=train_feature_names  # ✅ 传递训练时的特征名
        )
        test_loader = test_dataset
    
    if len(test_dataset) < 10:
        print(f"  ❌ 有效测试序列不足：{len(test_dataset)}")
        return {'status': 'failed', 'reason': 'insufficient_data'}
    
    # 获取时间戳
    test_timestamps = test_dataset.get_timestamps(test_dataset.valid_indices)
    
    # 6. 测试评估
    print(f"\n  📊 测试集评估...")
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
    
    pd.DataFrame({
        'prediction': test_results['predictions'],
        'target': test_results['targets']
    }).to_csv(symbol_output_dir / f'{symbol}_predictions.csv', index=False)
    
    # 8. ✅ 生成可视化
    visualizer = TestingVisualizer(symbol_output_dir)
    visualizer.generate_all_test_plots(symbol, model_type, test_results, test_timestamps, target_col)
    
    # 9. 生成摘要
    summary = {
        'symbol': symbol,
        'model_type': model_type,
        'n_samples': len(test_dataset),
        'n_factors': train_config['n_factors'],
        'test_ic': test_results['ic'],
        'direction_accuracy': test_results['direction_accuracy'],
        'long_short_return': (
            test_results['quantile_returns'].get(max(test_results['quantile_returns'].keys()), 0) -
            test_results['quantile_returns'].get(min(test_results['quantile_returns'].keys()), 0)
        ),
        'train_best_val_ic': train_config['best_val_ic'],
        'status': 'success'
    }
    
    print(f"\n  📋 测试摘要:")
    print(f"      测试 IC: {summary['test_ic']:.4f}")
    print(f"      方向准确率：{summary['direction_accuracy']:.4f}")
    print(f"      多空收益：{summary['long_short_return']:.6f}")
    print(f"      训练最佳 IC: {summary['train_best_val_ic']:.4f}")
    
    return summary


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
    parser.add_argument('--symbol', type=str, default='AAVEUSDT', help='交易对名称')
    parser.add_argument('--model', type=str, default='transformer',
                       choices=['transformer', 'linear', 'logistic', 'lightgbm'],
                       help='模型类型')
    parser.add_argument('--all_models', action='store_true', help='测试所有模型')
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 多模型高频因子测试脚本 (修复版)")
    print("="*60)
    print(f"📁 因子目录：{config.FACTOR_DIR}")
    print(f"📁 模型目录：{config.MODEL_DIR}")
    print(f"📁 输出目录：{config.OUTPUT_DIR}")
    print(f"🧠 设备：{config.DEVICE}")
    print(f"📦 模型：{args.model}")
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
            summary = test_symbol(args.symbol, config, model_type)
            all_summaries.append(summary)
        except Exception as e:
            print(f"❌ {args.symbol} ({model_type}) 测试失败：{e}")
            import traceback
            traceback.print_exc()
            all_summaries.append({'symbol': args.symbol, 'model_type': model_type, 'status': 'failed', 'error': str(e)})
    
    generate_summary_report(all_summaries, config)
    
    print("\n" + "="*60)
    print("🎉 多模型因子测试完成!")
    print("="*60)