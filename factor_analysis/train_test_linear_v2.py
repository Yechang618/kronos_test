#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear Regression Factor Training & Testing Script
纯 Linear Regression 版本，包含因子重要性分析和显著性检验
优化版：添加 Ridge 回归、FDR 校正、VIF 检测、完善可视化
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import pickle
import json
import argparse
from datetime import datetime
import time
warnings.filterwarnings('ignore')

# 机器学习
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats import t

# 多重检验校正
from statsmodels.stats.multitest import multipletests

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# 配置区域
# ============================
class Config:
    # 数据路径
    FACTOR_DIR = Path("./datasets/factors/hf_factors")
    OUTPUT_DIR = Path("./datasets/model_training/linear_regression")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 序列参数
    SEQ_LENGTH = 60
    PREDICTION_HORIZON = 1
    TARGET_COL = 'basis_ret_future_1'
    # DEFAULT_TARGET_COL = 'kalman_swap_filtered'
    # DEFAULT_TARGET_COL = 'basis_ask'
    DEFAULT_TARGET_COL = 'spot_mid'
    
    # 数据分割
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # 随机种子
    RANDOM_SEED = 42
    
    # 显著性检验
    SIGNIFICANCE_LEVEL = 0.05
    
    # ✅ 新增：使用 FDR 校正
    USE_FDR_CORRECTION = True
    FDR_ALPHA = 0.05
    
    # ✅ 新增：模型类型选择
    MODEL_TYPE = 'lasso'  # 'linear', 'ridge', 'lasso'
    RIDGE_ALPHA = 1.0
    
    # ✅ 新增：VIF 阈值（检测共线性）
    VIF_THRESHOLD = 10.0
    CHECK_VIF = True
    
    # 设备
    DEVICE = 'cpu'

config = Config()


# ============================
# 统计检验工具类（优化版）
# ============================
class StatisticalTests:
    """Linear Regression 统计显著性检验工具"""
    
    @staticmethod
    def linear_regression_summary(X, y, model, feature_names, use_fdr=False):
        n_samples, n_features = X.shape
        y_pred = model.predict(X)
        residuals = y - y_pred
        mse = np.sum(residuals**2) / (n_samples - n_features - 1)
        
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X.T @ X)
        
        coef_var = mse * np.diag(XtX_inv)
        coef_std = np.sqrt(coef_var)
        t_stats = model.coef_ / (coef_std + 1e-10)
        df = n_samples - n_features - 1
        p_values = 2 * (1 - t.cdf(np.abs(t_stats), df))
        
        # ✅ FDR 校正
        if use_fdr:
            reject, pvals_corrected, _, _ = multipletests(
                p_values, alpha=0.05, method='fdr_bh'
            )
            p_values_final = pvals_corrected
        else:
            p_values_final = p_values
        
        t_critical = t.ppf(1 - 0.025, df)
        ci_lower = model.coef_ - t_critical * coef_std
        ci_upper = model.coef_ + t_critical * coef_std
        
        ss_tot = np.sum((y - np.mean(y))**2)
        ss_res = np.sum(residuals**2)
        r_squared = 1 - ss_res / ss_tot
        adj_r_squared = 1 - (1 - r_squared) * (n_samples - 1) / (n_samples - n_features - 1)
        f_stat = (r_squared / n_features) / ((1 - r_squared) / (n_samples - n_features - 1))
        f_p_value = 1 - stats.f.cdf(f_stat, n_features, n_samples - n_features - 1)
        
        return {
            'feature_names': feature_names,
            'coefficients': model.coef_,
            'intercept': model.intercept_ if hasattr(model, 'intercept_') else 0,
            'coef_std': coef_std,
            't_statistics': t_stats,
            'p_values': p_values,
            'p_values_corrected': p_values_final,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'f_statistic': f_stat,
            'f_p_value': f_p_value,
            'mse': mse,
            'n_samples': n_samples,
            'n_features': n_features
        }
    
    @staticmethod
    def calculate_vif(X, feature_names, threshold=10.0):
        """计算方差膨胀因子 (VIF) 检测共线性"""
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        vif_df = pd.DataFrame()
        vif_df['factor'] = feature_names
        vif_df['VIF'] = [variance_inflation_factor(X.values, i) 
                         for i in range(X.shape[1])]
        vif_df = vif_df.sort_values('VIF', ascending=False)
        
        high_vif = vif_df[vif_df['VIF'] > threshold]
        
        return {
            'vif_table': vif_df,
            'high_vif_factors': high_vif,
            'n_high_vif': len(high_vif),
            'max_vif': vif_df['VIF'].max(),
            'mean_vif': vif_df['VIF'].mean()
        }
    
    @staticmethod
    def get_significant_features(summary, alpha=0.05, use_fdr=False):
        p_values = summary['p_values_corrected'] if use_fdr else summary['p_values']
        feature_names = summary['feature_names']
        
        significant = []
        for i, (name, p) in enumerate(zip(feature_names, p_values)):
            significant.append({
                'feature': name,
                'coefficient': summary['coefficients'][i],
                'p_value': p,
                'p_value_raw': summary['p_values'][i],
                'significant': p < alpha,
                'significance_level': '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
            })
        
        return pd.DataFrame(significant).sort_values('p_value')


# ============================
# 数据集类（优化版）
# ============================
class FactorFlatDataset:
    """扁平化因子数据集 (用于 Linear Regression)"""
    
    def __init__(self, factor_df: pd.DataFrame, seq_length: int,
                 prediction_horizon: int, target_col: str = 'target', default_target_col: str = 'mid_basis'):
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        self.target_col = target_col
        self.default_target_col = default_target_col
        
        exclude_cols = [target_col, 'target', 'timestamp', 'year_month', 'timestampes']
        self.factor_cols = [c for c in factor_df.columns if c not in exclude_cols]
        
        factor_data_raw = factor_df[self.factor_cols].copy()
        
        # ✅ 修复：先处理 Inf 值
        factor_data_raw = factor_data_raw.replace([np.inf, -np.inf], np.nan)
        
        # Winsorization
        self.winsorization_limits = {}
        for col in self.factor_cols:
            if col in factor_data_raw.columns:
                lower = factor_data_raw[col].quantile(0.01)
                upper = factor_data_raw[col].quantile(0.99)
                factor_data_raw[col] = factor_data_raw[col].clip(lower, upper)
                self.winsorization_limits[col] = {'lower': lower, 'upper': upper}
        
        # 填充 NaN
        factor_data_raw = factor_data_raw.fillna(0)
        
        # 标准化
        self.scaler = RobustScaler()
        self.factor_data = self.scaler.fit_transform(factor_data_raw)
        
        # 目标变量处理
        if target_col not in factor_df.columns:
            print(f"  {target_col} 列不存在，使用 '{default_target_col}_{prediction_horizon}' 计算未来收益作为目标")
            self.targets = factor_df[default_target_col].shift(-prediction_horizon).pct_change(
                prediction_horizon).values
        else:
            self.targets = factor_df[target_col].values
        
        self.targets = np.nan_to_num(self.targets, nan=0.0, posinf=0.0, neginf=0.0)
        self.target_mean = np.mean(self.targets)
        self.target_std = np.std(self.targets) + 1e-10
        self.targets_normalized = (self.targets - self.target_mean) / self.target_std
        self.targets_normalized = np.clip(self.targets_normalized, -5, 5)
        
        # 保存时间戳
        self.timestamps = factor_df.index.values if hasattr(factor_df, 'index') else None
        
        self.valid_indices = self._get_valid_indices()
        print(f"  📊 数据集：{len(self.valid_indices)} 个有效样本，{len(self.factor_cols)} 个因子")
    
    def _get_valid_indices(self):
        valid = []
        for i in range(len(self.factor_data) - self.prediction_horizon):
            target_idx = i + self.prediction_horizon
            if target_idx >= len(self.targets):
                continue
            if np.isnan(self.targets[target_idx]) or np.isinf(self.targets[target_idx]):
                continue
            if np.isnan(self.factor_data[i]).sum() / len(self.factor_data[i]) > 0.3:
                continue
            valid.append(i)
        return valid
    
    def __len__(self):
        return len(self.valid_indices)
    
    def get_data(self):
        X, y_reg, indices = [], [], []
        for idx in self.valid_indices:
            X.append(self.factor_data[idx])
            y_reg.append(self.targets_normalized[idx + self.prediction_horizon])
            indices.append(idx)
        return np.array(X), np.array(y_reg), np.array(indices)
    
    def get_factor_names(self):
        return self.factor_cols
    
    def get_scaler(self):
        return self.scaler
    
    def get_target_stats(self):
        return {'mean': self.target_mean, 'std': self.target_std}
    
    def get_timestamps(self, indices):
        if self.timestamps is None:
            return None
        return self.timestamps[indices]
    
    def get_winsorization_limits(self):
        return self.winsorization_limits


# ============================
# 模型训练器（优化版）
# ============================
class LinearTrainer:
    """Linear Regression 训练器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.summary = None
        self.vif_results = None
    
    def train(self, X_train, y_train, X_val, y_val, feature_names):
        print("\n  📈 训练 Linear Regression...")
        
        # ✅ 检查 VIF
        if self.config.CHECK_VIF:
            print("    🔍 检查因子共线性 (VIF)...")
            self.vif_results = StatisticalTests.calculate_vif(
                pd.DataFrame(X_train, columns=feature_names),
                feature_names,
                threshold=self.config.VIF_THRESHOLD
            )
            print(f"    最大 VIF: {self.vif_results['max_vif']:.2f}")
            print(f"    平均 VIF: {self.vif_results['mean_vif']:.2f}")
            print(f"    高 VIF 因子数：{self.vif_results['n_high_vif']}/{len(feature_names)}")
            
            if self.vif_results['n_high_vif'] > 0:
                print(f"    ⚠️ 警告：{self.vif_results['n_high_vif']} 个因子 VIF > {self.config.VIF_THRESHOLD}")
        
        # ✅ 根据配置选择模型
        if self.config.MODEL_TYPE == 'ridge':
            print(f"    使用 Ridge 回归 (alpha={self.config.RIDGE_ALPHA})")
            self.model = Ridge(alpha=self.config.RIDGE_ALPHA)
        elif self.config.MODEL_TYPE == 'lasso':
            print(f"    使用 Lasso 回归")
            self.model = Lasso(alpha=0.01, max_iter=10000)
        else:
            print(f"    使用普通 Linear Regression")
            self.model = LinearRegression()
        
        self.model.fit(X_train, y_train)
        
        # 训练集评估
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        # 统计检验
        self.summary = StatisticalTests.linear_regression_summary(
            X_train, y_train, self.model, feature_names,
            use_fdr=self.config.USE_FDR_CORRECTION
        )
        
        print(f"    训练 MSE: {train_mse:.6f}, R²: {train_r2:.4f}")
        print(f"    验证 MSE: {val_mse:.6f}, R²: {val_r2:.4f}")
        print(f"    调整 R²: {self.summary['adj_r_squared']:.4f}")
        print(f"    F 统计量：{self.summary['f_statistic']:.4f} (p={self.summary['f_p_value']:.4e})")
        
        # 显著性因子数量
        significant_df = StatisticalTests.get_significant_features(
            self.summary, alpha=self.config.SIGNIFICANCE_LEVEL,
            use_fdr=self.config.USE_FDR_CORRECTION
        )
        n_significant = significant_df['significant'].sum()
        n_significant_raw = (significant_df['p_value_raw'] < self.config.SIGNIFICANCE_LEVEL).sum()
        
        print(f"    显著性因子数 (原始 p<{self.config.SIGNIFICANCE_LEVEL}): {n_significant_raw}/{len(feature_names)}")
        if self.config.USE_FDR_CORRECTION:
            print(f"    显著性因子数 (FDR 校正后): {n_significant}/{len(feature_names)}")
        
        return {
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'n_significant': n_significant,
            'n_significant_raw': n_significant_raw,
            'status': 'success'
        }
    
    def get_importance(self, top_n=20):
        if self.summary is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.summary['feature_names'],
            'coefficient': self.summary['coefficients'],
            'abs_coefficient': np.abs(self.summary['coefficients']),
            'std_error': self.summary['coef_std'],
            't_statistic': self.summary['t_statistics'],
            'p_value': self.summary['p_values_corrected'] if self.config.USE_FDR_CORRECTION else self.summary['p_values'],
            'p_value_raw': self.summary['p_values'],
            'ci_lower': self.summary['ci_lower'],
            'ci_upper': self.summary['ci_upper']
        })
        
        importance_df['significant'] = importance_df['p_value'] < self.config.SIGNIFICANCE_LEVEL
        importance_df['significance_level'] = importance_df['p_value'].apply(
            lambda p: '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
        )
        
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        return importance_df.head(top_n)


# ============================
# 模型测试器（优化版）
# ============================
class LinearTester:
    """Linear Regression 测试器"""
    
    def __init__(self, model, config: Config):
        self.model = model
        self.config = config
    
    def test(self, X_test, y_test, target_mean, target_std, timestamps=None):
        print("\n  📈 测试 Linear Regression...")
        
        pred_normalized = self.model.predict(X_test)
        # 保持标准化格式用于 IC 计算
        pred_original = pred_normalized
        target_original = y_test
        
        # 回归指标
        mse = mean_squared_error(y_test, pred_normalized)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, pred_normalized)
        
        # 方向准确率
        direction_acc = accuracy_score(np.sign(pred_normalized), np.sign(y_test))
        
        # IC (Spearman 相关系数)
        ic, ic_pvalue = stats.spearmanr(pred_normalized, y_test)
        
        # 分层回测
        quantile_returns = self._quantile_backtest(pred_original, target_original)
        
        print(f"    测试 MSE: {mse:.6f}, RMSE: {rmse:.6f}")
        print(f"    测试 R²: {r2:.4f}")
        print(f"    方向准确率：{direction_acc:.4f}")
        print(f"    IC: {ic:.4f} (p={ic_pvalue:.4e})")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'direction_accuracy': direction_acc,
            'ic': ic,
            'ic_pvalue': ic_pvalue,
            'quantile_returns': quantile_returns,
            'predictions': pred_original,
            'targets': target_original,
            'timestamps': timestamps
        }
    
    def _quantile_backtest(self, preds, targets, n_quantiles=5):
        try:
            valid_mask = ~(np.isnan(preds) | np.isnan(targets))
            preds = preds[valid_mask]
            targets = targets[valid_mask]
            
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
# 可视化器（完整修复版）
# ============================
class ImportanceVisualizer:
    """因子重要性和测试可视化"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_name = f"{config.DEFAULT_TARGET_COL}_{config.PREDICTION_HORIZON}"
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_top_factors(self, importance_df, symbol):
        if importance_df is None or len(importance_df) == 0:
            return
        
        top_factors = importance_df.head(15)
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = ['green' if c > 0 else 'red' for c in top_factors['coefficient']]
        ax.barh(range(len(top_factors)), top_factors['coefficient'], color=colors)
        ax.set_yticks(range(len(top_factors)))
        ax.set_yticklabels(top_factors['feature'].str.slice(-30), fontsize=9)
        ax.set_xlabel('Coefficient')
        ax.set_title(f'Linear Regression - Top 15 Factors ({symbol})')
        ax.axvline(0, color='black', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_top_factors_{config.DEFAULT_TARGET_COL}_{config.PREDICTION_HORIZON}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存 Top 因子图：{symbol}_top_factors_{config.DEFAULT_TARGET_COL}_{config.PREDICTION_HORIZON}.png")
    
    def plot_significance_table(self, importance_df, symbol):
        if importance_df is None or len(importance_df) == 0:
            return
        
        top_sig = importance_df[importance_df['significant']].head(10)
        if len(top_sig) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        for _, row in top_sig.iterrows():
            table_data.append([
                row['feature'][:35],
                f"{row['coefficient']:.6f}",
                f"{row['t_statistic']:.2f}",
                f"{row['p_value']:.4e}",
                row['significance_level']
            ])
        
        table = ax.table(
            cellText=table_data,
            colLabels=['Factor', 'Coefficient', 't-stat', 'p-value', 'Signif'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        ax.set_title(f'Linear Regression - Top 10 Significant Factors ({symbol})', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_significance_table_{config.DEFAULT_TARGET_COL}_{config.PREDICTION_HORIZON}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存显著性表格：{symbol}_significance_table_{config.DEFAULT_TARGET_COL}_{config.PREDICTION_HORIZON}.png")
    
    def plot_prediction_quality(self, test_results, symbol):
        if test_results is None:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].scatter(test_results['targets'], test_results['predictions'], alpha=0.3, s=10)
        axes[0].plot([test_results['targets'].min(), test_results['targets'].max()],
                    [test_results['targets'].min(), test_results['targets'].max()],
                    'r--', linewidth=2)
        axes[0].set_xlabel('Actual Return')
        axes[0].set_ylabel('Predicted Return')
        axes[0].set_title(f'Prediction vs Actual (IC={test_results["ic"]:.4f})')
        axes[0].grid(True, alpha=0.3)
        
        quantile_returns = test_results['quantile_returns']
        if quantile_returns:
            quantiles = list(quantile_returns.keys())
            returns = list(quantile_returns.values())
            colors = ['red' if r < 0 else 'green' for r in returns]
            axes[1].bar(range(len(quantiles)), returns, color=colors, edgecolor='black')
            axes[1].set_xticks(range(len(quantiles)))
            axes[1].set_xticklabels([f'Q{i+1}' for i in quantiles])
            axes[1].set_xlabel('Quantile')
            axes[1].set_ylabel('Average Return')
            axes[1].set_title('Quantile Backtest Returns')
            axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_prediction_quality_{config.DEFAULT_TARGET_COL}_{config.PREDICTION_HORIZON}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存预测质量图：{symbol}_prediction_quality_{config.DEFAULT_TARGET_COL}_{config.PREDICTION_HORIZON}.png")
    
    def plot_coefficient_distribution(self, importance_df, symbol):
        if importance_df is None or len(importance_df) == 0:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(importance_df['coefficient'], bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Coefficient')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Coefficient Distribution')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(importance_df['t_statistic'], bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1].axvline(2, color='green', linestyle='--', linewidth=2, label='t=2')
        axes[1].axvline(-2, color='green', linestyle='--', linewidth=2)
        axes[1].set_xlabel('t-Statistic')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('t-Statistic Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_coefficient_distribution_{config.DEFAULT_TARGET_COL}_{config.PREDICTION_HORIZON}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存系数分布图：{symbol}_coefficient_distribution_{config.DEFAULT_TARGET_COL}_{config.PREDICTION_HORIZON}.png")
    
    def plot_true_vs_pred_scatter(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   symbol: str, title: str = None):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(y_true, y_pred, alpha=0.3, s=15, color='steelblue', edgecolors='none')
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        if len(y_true) > 2:
            coef = np.polyfit(y_true, y_pred, 1)
            poly = np.poly1d(coef)
            ax.plot([min_val, max_val], poly([min_val, max_val]), 'g-', linewidth=1.5, 
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
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('True Values (y_true)', fontsize=12)
        ax.set_ylabel('Predicted Values (y_pred)', fontsize=12)
        ax.set_title(title or f'{symbol}: True vs Predicted', fontsize=14)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_true_vs_pred_scatter_{config.DEFAULT_TARGET_COL}_{config.PREDICTION_HORIZON}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存散点图：{symbol}_true_vs_pred_scatter_{config.DEFAULT_TARGET_COL}_{config.PREDICTION_HORIZON}.png")
    
    def plot_true_pred_timeseries(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   timestamps: np.ndarray = None, symbol: str = None,
                                   max_points: int = 2000):
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
        
        axes[0].plot(timestamps_plot, y_true_plot, label='True', linewidth=1, alpha=0.8, color='blue')
        axes[0].plot(timestamps_plot, y_pred_plot, label='Predicted', linewidth=1, alpha=0.8, color='orange')
        axes[0].set_ylabel('Value')
        axes[0].set_title(f'{symbol}: True vs Predicted Time Series ({self.target_name})' if symbol else 'True vs Predicted Time Series')
        axes[0].legend(loc='upper right', fontsize=9)
        axes[0].grid(True, alpha=0.3)
        
        residuals = y_pred_plot - y_true_plot
        axes[1].plot(timestamps_plot, residuals, label='Residual (pred - true)',
                    linewidth=0.5, color='gray', alpha=0.7)
        axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[1].fill_between(timestamps_plot, residuals, 0,
                            where=(residuals > 0), color='green', alpha=0.2, label='Over-predicted')
        axes[1].fill_between(timestamps_plot, residuals, 0,
                            where=(residuals < 0), color='red', alpha=0.2, label='Under-predicted')
        axes[1].set_xlabel('Time Step' if timestamps is None else 'Timestamp')
        axes[1].set_ylabel('Residual')
        axes[1].legend(loc='upper right', fontsize=9)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_true_pred_timeseries_{config.DEFAULT_TARGET_COL}_{config.PREDICTION_HORIZON}.png' if symbol else 'true_pred_timeseries_{config.DEFAULT_TARGET_COL}_{config.PREDICTION_HORIZON}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存时间序列对比图：{symbol}_true_pred_timeseries_{config.DEFAULT_TARGET_COL}_{config.PREDICTION_HORIZON}.png")
    
    def plot_residual_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, symbol: str):
        residuals = y_pred - y_true
        residuals = residuals[~np.isnan(residuals)]
        
        if len(residuals) < 10:
            print(f"  ⚠️ 残差样本不足，跳过残差分析图")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # ✅ 修复：使用正确的 2D 数组索引 [row, col]
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
            axes[1, 0].text(0.5, 0.5, f'ACF failed: {str(e)}', ha='center', va='center',
                       transform=axes[1, 0].transAxes, fontsize=8)
            axes[1, 0].set_title('Residual Autocorrelation')
        
        axes[1, 1].scatter(y_pred, residuals, alpha=0.3, s=10, color='gray')
        axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[1, 1].set_xlabel('Predicted Value')
        axes[1, 1].set_ylabel('Residual')
        axes[1, 1].set_title('Residuals vs Predicted')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_residual_analysis_{config.DEFAULT_TARGET_COL}_{config.PREDICTION_HORIZON}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存残差分析图：{symbol}_residual_analysis_{config.DEFAULT_TARGET_COL}_{config.PREDICTION_HORIZON}.png")
    
    def plot_cumulative_returns(self, y_true: np.ndarray, y_pred: np.ndarray, symbol: str):
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
        ax.set_title(f'{symbol}: Cumulative Returns Comparison', fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_cumulative_returns_{config.DEFAULT_TARGET_COL}_{config.PREDICTION_HORIZON}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存累积收益图：{symbol}_cumulative_returns_{config.DEFAULT_TARGET_COL}_{config.PREDICTION_HORIZON}.png")
    
    def plot_quantile_returns(self, quantile_returns: dict, symbol: str):
        if not quantile_returns:
            print(f"  ⚠️ 无分层收益数据，跳过")
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
        ax.set_title(f'{symbol}: Quantile Backtest Returns')
        ax.grid(True, alpha=0.3, axis='y')
        
        if len(returns) >= 2:
            long_short = returns[-1] - returns[0]
            ax.axhline(long_short, color='blue', linestyle='--',
                      label=f'Long-Short: {long_short:.6f}')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_quantile_returns_{config.DEFAULT_TARGET_COL}_{config.PREDICTION_HORIZON}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存分层收益图：{symbol}_quantile_returns_{config.DEFAULT_TARGET_COL}_{config.PREDICTION_HORIZON}.png")
    
    def plot_prediction_distribution(self, y_true: np.ndarray, y_pred: np.ndarray, symbol: str):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        bins = 50
        axes[0].hist(y_true, bins=bins, alpha=0.5, label='True', color='blue', density=True, edgecolor='black')
        axes[0].hist(y_pred, bins=bins, alpha=0.5, label='Predicted', color='orange', density=True, edgecolor='black')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Distribution Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        from scipy.stats import gaussian_kde
        try:
            y_true_clean = y_true[~np.isnan(y_true)]
            y_pred_clean = y_pred[~np.isnan(y_pred)]
            
            if len(y_true_clean) > 10 and len(y_pred_clean) > 10:
                kde_true = gaussian_kde(y_true_clean)
                kde_pred = gaussian_kde(y_pred_clean)
                
                x_grid = np.linspace(min(y_true_clean.min(), y_pred_clean.min()),
                                    max(y_true_clean.max(), y_pred_clean.max()), 200)
                axes[1].plot(x_grid, kde_true(x_grid), label='True', linewidth=2, color='blue')
                axes[1].plot(x_grid, kde_pred(x_grid), label='Predicted', linewidth=2, color='orange')
                axes[1].set_xlabel('Value')
                axes[1].set_ylabel('Density')
                axes[1].set_title('KDE Comparison')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
        except:
            axes[1].text(0.5, 0.5, 'KDE computation failed', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('KDE Comparison')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_prediction_distribution_{config.DEFAULT_TARGET_COL}_{config.PREDICTION_HORIZON}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存分布对比图：{symbol}_prediction_distribution_{config.DEFAULT_TARGET_COL}_{config.PREDICTION_HORIZON}.png")
    
    def plot_vif_analysis(self, vif_results, symbol):
        """VIF 共线性分析图"""
        if vif_results is None:
            return
        
        vif_df = vif_results['vif_table'].head(20)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['red' if v > 10 else ('orange' if v > 5 else 'green') for v in vif_df['VIF']]
        ax.barh(range(len(vif_df)), vif_df['VIF'], color=colors)
        ax.set_yticks(range(len(vif_df)))
        ax.set_yticklabels(vif_df['factor'].str.slice(-30), fontsize=8)
        ax.set_xlabel('VIF')
        ax.set_title(f'{symbol}: Variance Inflation Factor (Top 20)')
        ax.axvline(10, color='red', linestyle='--', linewidth=2, label='VIF=10 (High)')
        ax.axvline(5, color='orange', linestyle='--', linewidth=1, label='VIF=5 (Moderate)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_vif_analysis_{config.DEFAULT_TARGET_COL}_{config.PREDICTION_HORIZON}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存 VIF 分析图：{symbol}_vif_analysis_{config.DEFAULT_TARGET_COL}_{config.PREDICTION_HORIZON}.png")
    
    def generate_all_test_plots(self, symbol: str, test_results: dict, vif_results=None):
        print("  🎨 生成测试集可视化图表...")
        
        y_true = test_results['targets']
        y_pred = test_results['predictions']
        timestamps = test_results.get('timestamps', None)
        quantile_returns = test_results.get('quantile_returns', None)
        
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]
        
        if len(y_true_clean) < 10:
            print(f"  ⚠️ 有效样本不足，跳过可视化")
            return
        
        self.plot_true_vs_pred_scatter(y_true_clean, y_pred_clean, symbol)
        self.plot_true_pred_timeseries(y_true_clean, y_pred_clean, timestamps, symbol)
        self.plot_residual_analysis(y_true_clean, y_pred_clean, symbol)
        self.plot_cumulative_returns(y_true_clean, y_pred_clean, symbol)
        self.plot_prediction_distribution(y_true_clean, y_pred_clean, symbol)
        
        if quantile_returns:
            self.plot_quantile_returns(quantile_returns, symbol)
        
        if vif_results:
            self.plot_vif_analysis(vif_results, symbol)
        
        print("  ✅ 测试可视化完成")


# ============================
# 主流程
# ============================
def train_test_symbol(symbol: str, config: Config) -> dict:
    print(f"\n{'='*60}")
    print(f"🧠 交易对：{symbol}")
    print(f"{'='*60}")
    
    # 1. 加载因子数据
    symbol_dir = config.FACTOR_DIR / symbol
    if not symbol_dir.exists():
        print(f"  ❌ 因子目录不存在：{symbol_dir}")
        return {'status': 'failed', 'reason': 'no_factor_data'}
    
    factor_files = list(symbol_dir.glob("*.csv.gz"))
    if not factor_files:
        print(f"  ❌ 无因子文件：{symbol_dir}")
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
    print(f"  ✅ 加载 {len(full_df)} 条记录")
    
    # 2. 创建数据集
    dataset = FactorFlatDataset(
        full_df,
        seq_length=config.SEQ_LENGTH,
        prediction_horizon=config.PREDICTION_HORIZON,
        target_col=config.TARGET_COL,
        default_target_col= config.DEFAULT_TARGET_COL
    )
    
    if len(dataset) < 100:
        print(f"  ❌ 有效样本不足：{len(dataset)}")
        return {'status': 'failed', 'reason': 'insufficient_data'}
    
    # 3. 获取数据
    X, y_reg, indices = dataset.get_data()
    feature_names = dataset.get_factor_names()
    timestamps = dataset.get_timestamps(indices)
    print(f"  📊 数据形状：X={X.shape}, y_reg={y_reg.shape}")
    
    # 4. ✅ 时间序列分割（避免前视偏差）
    n_samples = len(X)
    train_end = int(n_samples * config.TRAIN_RATIO)
    val_end = int(n_samples * (config.TRAIN_RATIO + config.VAL_RATIO))
    
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y_reg[:train_end], y_reg[train_end:val_end], y_reg[val_end:]
    indices_train, indices_val, indices_test = indices[:train_end], indices[train_end:val_end], indices[val_end:]
    
    test_timestamps = dataset.get_timestamps(indices_test)
    
    print(f"  📊 数据分割：训练={len(X_train)}, 验证={len(X_val)}, 测试={len(X_test)}")
    
    # 5. 训练模型
    trainer = LinearTrainer(config)
    metrics = trainer.train(X_train, y_train, X_val, y_val, feature_names)
    
    # 6. 测试模型
    tester = LinearTester(trainer.model, config)
    target_stats = dataset.get_target_stats()
    test_results = tester.test(X_test, y_test, target_stats['mean'], target_stats['std'],
                               timestamps=test_timestamps)
    
    # 7. 获取因子重要性
    importance = trainer.get_importance(top_n=20)
    
    # 8. 保存结果
    symbol_output_dir = config.OUTPUT_DIR / symbol
    symbol_output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(symbol_output_dir / 'linear_model.pkl', 'wb') as f:
        pickle.dump(trainer.model, f)
    
    with open(symbol_output_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(dataset.get_scaler(), f)
    
    train_config = {
        'symbol': symbol,
        'model_type': config.MODEL_TYPE,
        'ridge_alpha': config.RIDGE_ALPHA if config.MODEL_TYPE == 'ridge' else None,
        'seq_length': config.SEQ_LENGTH,
        'prediction_horizon': config.PREDICTION_HORIZON,
        'target_col': config.TARGET_COL,
        'n_factors': len(feature_names),
        'factor_names': feature_names,
        'target_mean': target_stats['mean'],
        'target_std': target_stats['std'],
        'use_fdr_correction': config.USE_FDR_CORRECTION,
        'vif_threshold': config.VIF_THRESHOLD,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    with open(symbol_output_dir / 'train_config.json', 'w') as f:
        json.dump(train_config, f, indent=2, default=str)
    
    if importance is not None:
        importance.to_csv(symbol_output_dir / 'factor_importance.csv', index=False)
    
    if trainer.summary is not None:
        summary_df = pd.DataFrame({
            'feature': trainer.summary['feature_names'],
            'coefficient': trainer.summary['coefficients'],
            'std_error': trainer.summary['coef_std'],
            't_statistic': trainer.summary['t_statistics'],
            'p_value': trainer.summary['p_values_corrected'] if config.USE_FDR_CORRECTION else trainer.summary['p_values'],
            'p_value_raw': trainer.summary['p_values'],
            'ci_lower': trainer.summary['ci_lower'],
            'ci_upper': trainer.summary['ci_upper'],
            'significant': (trainer.summary['p_values_corrected'] if config.USE_FDR_CORRECTION else trainer.summary['p_values']) < config.SIGNIFICANCE_LEVEL
        })
        summary_df.to_csv(symbol_output_dir / 'statistical_summary.csv', index=False)
    
    if trainer.vif_results is not None:
        trainer.vif_results['vif_table'].to_csv(symbol_output_dir / 'vif_analysis.csv', index=False)
    
    # 9. ✅ 增强版可视化
    visualizer = ImportanceVisualizer(symbol_output_dir)
    visualizer.plot_top_factors(importance, symbol)
    visualizer.plot_significance_table(importance, symbol)
    visualizer.plot_prediction_quality(test_results, symbol)
    visualizer.plot_coefficient_distribution(importance, symbol)
    visualizer.generate_all_test_plots(symbol, test_results, trainer.vif_results)
    
    # 10. 生成摘要
    summary = {
        'symbol': symbol,
        'n_samples': len(dataset),
        'n_factors': len(feature_names),
        'model_type': config.MODEL_TYPE,
        'train_r2': metrics['train_r2'],
        'val_r2': metrics['val_r2'],
        'test_ic': test_results['ic'],
        'test_direction_acc': test_results['direction_accuracy'],
        'n_significant': metrics['n_significant'],
        'n_significant_raw': metrics['n_significant_raw'],
        'max_vif': trainer.vif_results['max_vif'] if trainer.vif_results else None,
        'status': 'success'
    }
    
    print(f"\n  📋 训练测试摘要:")
    print(f"     模型类型：{config.MODEL_TYPE}")
    print(f"     验证 R²: {summary['val_r2']:.4f}")
    print(f"     测试 IC: {summary['test_ic']:.4f}")
    print(f"     方向准确率：{summary['test_direction_acc']:.4f}")
    print(f"     显著因子数 (FDR 校正): {summary['n_significant']}/{len(feature_names)}")
    print(f"     显著因子数 (原始): {summary['n_significant_raw']}/{len(feature_names)}")
    if trainer.vif_results:
        print(f"     最大 VIF: {trainer.vif_results['max_vif']:.2f}")
    print(f"     模型保存至：{symbol_output_dir}")
    
    return summary


def discover_symbols(config: Config) -> list:
    if not config.FACTOR_DIR.exists():
        raise FileNotFoundError(f"因子目录不存在：{config.FACTOR_DIR}")
    symbols = [d.name for d in config.FACTOR_DIR.iterdir() if d.is_dir()]
    print(f"🔍 发现 {len(symbols)} 个交易对有因子数据")
    return symbols


def generate_summary_report(summaries: list, config: Config):
    print(f"\n{'='*60}")
    print("📊 生成汇总报告")
    print(f"{'='*60}")
    
    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df[summary_df['status'] == 'success']
    
    if summary_df.empty:
        print("  ❌ 无成功训练的交易对")
        return
    
    summary_df.to_csv(config.OUTPUT_DIR / "all_symbols_summary.csv", index=False)
    
    if 'test_ic' in summary_df.columns:
        top_ic = summary_df.nlargest(5, 'test_ic')
        print("\n🏆 Top 5 交易对 (按测试 IC):")
        for _, row in top_ic.iterrows():
            print(f"   {row['symbol']}: IC={row['test_ic']:.4f}, R²={row['val_r2']:.4f}, "
                  f"显著因子={row['n_significant']}/{row['n_factors']}")
    
    if 'val_r2' in summary_df.columns:
        top_r2 = summary_df.nlargest(5, 'val_r2')
        print("\n🏆 Top 5 交易对 (按验证 R²):")
        for _, row in top_r2.iterrows():
            print(f"   {row['symbol']}: R²={row['val_r2']:.4f}, IC={row['test_ic']:.4f}, "
                  f"显著因子={row['n_significant']}/{row['n_factors']}")
    
    print(f"\n💾 汇总报告：{config.OUTPUT_DIR / 'all_symbols_summary.csv'}")


# ============================
# 主程序入口
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Linear Regression 因子训练测试脚本 (优化版)')
    parser.add_argument('--symbol', type=str, default='AVAXUSDT', help='交易对名称')
    parser.add_argument('--all_symbols', action='store_true', help='处理所有交易对')
    parser.add_argument('--model', type=str, default='linear',
                       choices=['linear', 'ridge', 'lasso'],
                       help='模型类型')
    parser.add_argument('--ridge_alpha', type=float, default=1.0, help='Ridge 正则化参数')
    parser.add_argument('--use_fdr', action='store_true', default=True, help='使用 FDR 校正')
    parser.add_argument('--check_vif', action='store_true', default=True, help='检查 VIF 共线性')
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 Linear Regression 因子训练测试脚本 (优化版)")
    print("="*60)
    print(f"📁 因子目录：{config.FACTOR_DIR}")
    print(f"📁 输出目录：{config.OUTPUT_DIR}")
    print(f"📐 序列长度：{config.SEQ_LENGTH} (30 秒 @ 500ms)")
    print(f"🎯 预测期：{config.PREDICTION_HORIZON} (下一秒)")
    print(f"🔧 模型类型：{args.model}")
    if args.model == 'ridge':
        print(f"🔧 Ridge Alpha: {args.ridge_alpha}")
    print(f"📊 FDR 校正：{'✅ 启用' if args.use_fdr else '❌ 禁用'}")
    print(f"📊 VIF 检测：{'✅ 启用' if args.check_vif else '❌ 禁用'}")
    print("="*60)
    
    # 应用配置
    config.MODEL_TYPE = args.model
    config.RIDGE_ALPHA = args.ridge_alpha
    config.USE_FDR_CORRECTION = args.use_fdr
    config.CHECK_VIF = args.check_vif
    
    symbols = discover_symbols(config)
    if not symbols:
        print("❌ 未发现任何交易对因子数据")
        exit(1)
    
    all_summaries = []
    
    if args.all_symbols:
        symbols_to_process = symbols
    else:
        symbols_to_process = [args.symbol]
    
    for i, symbol in enumerate(symbols_to_process, 1):
        print(f"\n[{i}/{len(symbols_to_process)}] 处理进度")
        try:
            summary = train_test_symbol(symbol, config)
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
    print("🎉 Linear Regression 训练测试完成!")
    print("="*60)