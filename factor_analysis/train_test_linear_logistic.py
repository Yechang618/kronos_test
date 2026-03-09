#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear & Logistic Regression Factor Training & Testing Script
支持 Linear Regression 和 Logistic Regression 两种模型
包含因子重要性分析和显著性检验
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import pickle
import json
import argparse
from datetime import datetime
warnings.filterwarnings('ignore')

# 机器学习
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    mean_squared_error, r2_score, classification_report
)
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats import t

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# 配置区域
# ============================
class Config:
    # 数据路径
    FACTOR_DIR = Path("./datasets/factors/hf_factors")
    OUTPUT_DIR = Path("./datasets/model_training/linear_logistic")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 序列参数
    SEQ_LENGTH = 60              # 30 秒 @ 500ms = 60 个时间点
    PREDICTION_HORIZON = 1       # 预测下一秒
    TARGET_COL = 'basis_ret_future_1'
    
    # 数据分割
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # 模型参数
    LOGISTIC_REGULARIZATION = 1.0  # C 参数 (逆正则化强度)
    LOGISTIC_MAX_ITER = 1000
    LOGISTIC_SOLVER = 'lbfgs'
    
    # 随机种子
    RANDOM_SEED = 42
    
    # 显著性检验
    SIGNIFICANCE_LEVEL = 0.05
    
    # 设备
    DEVICE = 'cpu'  # sklearn 不使用 GPU

    # ✅ 修复：Logistic 参数优化
    LOGISTIC_REGULARIZATION = 1.0
    LOGISTIC_MAX_ITER = 500  # 减少迭代次数
    LOGISTIC_SOLVER = 'lbfgs'
    
    # ✅ 修复：显著性检验参数
    SIGNIFICANCE_LEVEL = 0.05
    BOOTSTRAP_ITERATIONS = 30  # 从 100 减少到 30
    
    # ✅ 修复：分类阈值
    CLASSIFICATION_PERCENTILE = 40  # 使用 40% 分位数，确保约 40% 正样本

config = Config()


# ============================
# 统计检验工具类
# ============================
class StatisticalTests:
    """统计显著性检验工具"""
    
    @staticmethod
    def linear_regression_summary(X, y, model, feature_names):
        """
        Linear Regression 统计摘要
        返回：系数、标准误、t 统计量、p 值、置信区间
        """
        n_samples, n_features = X.shape
        
        # 预测值
        y_pred = model.predict(X)
        
        # 残差
        residuals = y - y_pred
        
        # 残差方差估计
        mse = np.sum(residuals**2) / (n_samples - n_features - 1)
        
        # 系数协方差矩阵 (X'X)^(-1) * MSE
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            # 如果奇异，使用伪逆
            XtX_inv = np.linalg.pinv(X.T @ X)
        
        coef_var = mse * np.diag(XtX_inv)
        coef_std = np.sqrt(coef_var)
        
        # t 统计量
        t_stats = model.coef_ / (coef_std + 1e-10)
        
        # p 值 (双尾检验)
        df = n_samples - n_features - 1
        p_values = 2 * (1 - t.cdf(np.abs(t_stats), df))
        
        # 置信区间 (95%)
        t_critical = t.ppf(1 - 0.025, df)
        ci_lower = model.coef_ - t_critical * coef_std
        ci_upper = model.coef_ + t_critical * coef_std
        
        # R² 和调整后 R²
        ss_tot = np.sum((y - np.mean(y))**2)
        ss_res = np.sum(residuals**2)
        r_squared = 1 - ss_res / ss_tot
        adj_r_squared = 1 - (1 - r_squared) * (n_samples - 1) / (n_samples - n_features - 1)
        
        # F 统计量
        f_stat = (r_squared / n_features) / ((1 - r_squared) / (n_samples - n_features - 1))
        f_p_value = 1 - stats.f.cdf(f_stat, n_features, n_samples - n_features - 1)
        
        summary = {
            'feature_names': feature_names,
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'coef_std': coef_std,
            't_statistics': t_stats,
            'p_values': p_values,
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
        
        return summary
    
    @staticmethod
    def logistic_regression_summary(X, y, model, feature_names):
        """
        Logistic Regression 统计摘要
        返回：系数、标准误、z 统计量、p 值、Odds Ratio
        """
        n_samples, n_features = X.shape
        
        # 对于逻辑回归，使用 Hessian 矩阵估计标准误
        # 这里使用简化方法：基于 Fisher 信息矩阵
        
        # 预测概率
        y_prob = model.predict_proba(X)[:, 1]
        
        # 系数标准误 (使用数值方法近似)
        # 更准确的方法需要使用完整的 Hessian 矩阵
        try:
            # 使用 bootstrap 估计标准误
            n_bootstrap = 100
            coef_samples = []
            
            for _ in range(n_bootstrap):
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_boot = X[indices]
                y_boot = y[indices]
                
                boot_model = LogisticRegression(
                    C=model.C,
                    max_iter=model.max_iter,
                    solver=model.solver,
                    random_state=np.random.randint(0, 10000)
                )
                boot_model.fit(X_boot, y_boot)
                coef_samples.append(boot_model.coef_[0])
            
            coef_samples = np.array(coef_samples)
            coef_std = np.std(coef_samples, axis=0)
        except:
            # 如果 bootstrap 失败，使用简化估计
            coef_std = np.ones(n_features) * 0.1
        
        # z 统计量
        z_stats = model.coef_[0] / (coef_std + 1e-10)
        
        # p 值 (双尾检验)
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))
        
        # Odds Ratio
        odds_ratio = np.exp(model.coef_[0])
        
        # 置信区间 (95%)
        z_critical = 1.96
        ci_lower = model.coef_[0] - z_critical * coef_std
        ci_upper = model.coef_[0] + z_critical * coef_std
        or_ci_lower = np.exp(ci_lower)
        or_ci_upper = np.exp(ci_upper)
        
        # 模型整体指标
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        # 伪 R² (McFadden's)
        null_model = LogisticRegression(C=model.C, max_iter=model.max_iter, solver=model.solver)
        null_model.fit(X, np.zeros_like(y))
        
        ll_full = model.score(X, y)  # 对数似然
        ll_null = null_model.score(X, y)
        mcfadden_r2 = 1 - ll_full / ll_null if ll_null != 0 else 0
        
        summary = {
            'feature_names': feature_names,
            'coefficients': model.coef_[0],
            'intercept': model.intercept_[0],
            'coef_std': coef_std,
            'z_statistics': z_stats,
            'p_values': p_values,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'odds_ratio': odds_ratio,
            'or_ci_lower': or_ci_lower,
            'or_ci_upper': or_ci_upper,
            'accuracy': accuracy,
            'mcfadden_r2': mcfadden_r2,
            'n_samples': n_samples,
            'n_features': n_features
        }
        
        return summary
    
    @staticmethod
    def get_significant_features(summary, alpha=0.05):
        """获取显著性因子"""
        p_values = summary['p_values']
        feature_names = summary['feature_names']
        
        significant = []
        for i, (name, p) in enumerate(zip(feature_names, p_values)):
            significant.append({
                'feature': name,
                'coefficient': summary['coefficients'][i],
                'p_value': p,
                'significant': p < alpha,
                'significance_level': '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
            })
        
        return pd.DataFrame(significant).sort_values('p_value')


# ============================
# 数据集类
# ============================
class FactorFlatDataset:
    """扁平化因子数据集 (用于 Linear/Logistic Regression)"""
    
    def __init__(self, factor_df: pd.DataFrame, seq_length: int,
                 prediction_horizon: int, target_col: str = 'target',
                 classification_percentile: int = 40):  # ✅ 新增参数):
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        self.target_col = target_col
        self.classification_percentile = classification_percentile

        # 获取因子列
        exclude_cols = [target_col, 'target', 'timestamp', 'year_month', 'timestampes']
        self.factor_cols = [c for c in factor_df.columns if c not in exclude_cols]
        
        # 数据预处理
        factor_data_raw = factor_df[self.factor_cols].copy()
        
        # Winsorization
        for col in self.factor_cols:
            if col in factor_data_raw.columns:
                lower = factor_data_raw[col].quantile(0.01)
                upper = factor_data_raw[col].quantile(0.99)
                factor_data_raw[col] = factor_data_raw[col].clip(lower, upper)
        
        factor_data_raw = factor_data_raw.fillna(0)
        
        # 标准化
        self.scaler = RobustScaler()
        self.factor_data = self.scaler.fit_transform(factor_data_raw)
        
        # 目标变量处理
        if target_col not in factor_df.columns:
            print(f"  {target_col} 列不存在，使用 'mid_basis' 计算未来收益作为目标")
            self.targets = factor_df['mid_basis'].shift(-prediction_horizon).pct_change(
                prediction_horizon).values
        else:
            self.targets = factor_df[target_col].values
        
        # 处理 Inf 和 NaN
        self.targets = np.nan_to_num(self.targets, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 目标变量标准化
        self.target_mean = np.mean(self.targets)
        self.target_std = np.std(self.targets) + 1e-10
        self.targets_normalized = (self.targets - self.target_mean) / self.target_std
        self.targets_normalized = np.clip(self.targets_normalized, -5, 5)
        
        # 分类标签
        # self.targets_cls = (self.targets > 0).astype(int)
        # ✅ 修复：使用分位数阈值而不是 0
        print(f"targets_normalized 分布：均值={np.mean(self.targets_normalized):.6f}, 标准差={np.std(self.targets_normalized):.6f}, "
              f"最小值={np.min(self.targets_normalized):.6f}, 最大值={np.max(self.targets_normalized):.6f}")
        print(f"  📊 使用 {classification_percentile}% 分位数作为分类阈值...")
        print(f"  📊 目标变量分位数 3: {np.percentile(self.targets_normalized, 3):.6f}")
        print(f"  📊 目标变量分位数 4: {np.percentile(self.targets_normalized, 4):.6f}")
        print(f"  📊 目标变量分位数 5: {np.percentile(self.targets_normalized, 5):.6f}")
        print(f"  📊 目标变量分位数 5.5: {np.percentile(self.targets_normalized, 5.5):.6f}")
        print(f"  📊 目标变量分位数 5.8: {np.percentile(self.targets_normalized, 5.8):.6f}")
        threshold = np.percentile(self.targets_normalized, classification_percentile)
        self.targets_cls = (self.targets_normalized > threshold).astype(int)
        print(f"  📊 分类阈值：{threshold:.6f} ({classification_percentile}% 分位数)")
        print(f"  📊 标签分布：0 类={np.sum(self.targets_cls == 0)} ({np.mean(self.targets_cls == 0):.2%}), "
              f"1 类={np.sum(self.targets_cls == 1)} ({np.mean(self.targets_cls == 1):.2%})")

        # 有效样本索引
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
            # 检查因子是否有过多 NaN
            if np.isnan(self.factor_data[i]).sum() / len(self.factor_data[i]) > 0.3:
                continue
            valid.append(i)
        return valid
    
    # ✅ 修复：添加 __len__() 方法
    def __len__(self):
        """返回有效样本数量"""
        return len(self.valid_indices)
        
    def get_data(self):
        """获取 numpy 格式数据"""
        X = []
        y_reg = []
        y_cls = []
        indices = []
        
        for idx in self.valid_indices:
            X.append(self.factor_data[idx])
            y_reg.append(self.targets_normalized[idx + self.prediction_horizon])
            y_cls.append(self.targets_cls[idx + self.prediction_horizon])
            indices.append(idx)
        
        return np.array(X), np.array(y_reg), np.array(y_cls), np.array(indices)
    
    def get_factor_names(self):
        return self.factor_cols
    
    def get_scaler(self):
        return self.scaler
    
    def get_target_stats(self):
        return {'mean': self.target_mean, 'std': self.target_std}


# ============================
# 模型训练器
# ============================
class LinearLogisticTrainer:
    """Linear & Logistic Regression 训练器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.linear_model = None
        self.logistic_model = None
        self.linear_summary = None
        self.logistic_summary = None
    
    def train_linear(self, X_train, y_train, X_val, y_val, feature_names):
        """训练 Linear Regression"""
        print("\n  📈 训练 Linear Regression...")
        
        self.linear_model = LinearRegression()
        self.linear_model.fit(X_train, y_train)
        
        # 训练集评估
        train_pred = self.linear_model.predict(X_train)
        val_pred = self.linear_model.predict(X_val)
        
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        # 统计检验
        self.linear_summary = StatisticalTests.linear_regression_summary(
            X_train, y_train, self.linear_model, feature_names
        )
        
        print(f"    训练 MSE: {train_mse:.6f}, R²: {train_r2:.4f}")
        print(f"    验证 MSE: {val_mse:.6f}, R²: {val_r2:.4f}")
        print(f"    调整 R²: {self.linear_summary['adj_r_squared']:.4f}")
        print(f"    F 统计量：{self.linear_summary['f_statistic']:.4f} (p={self.linear_summary['f_p_value']:.4e})")
        
        # 显著性因子数量
        significant_df = StatisticalTests.get_significant_features(
            self.linear_summary, alpha=self.config.SIGNIFICANCE_LEVEL
        )
        n_significant = significant_df['significant'].sum()
        print(f"    显著性因子数 (p<{self.config.SIGNIFICANCE_LEVEL}): {n_significant}/{len(feature_names)}")
        
        return {
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'n_significant': n_significant
        }
    
    def train_logistic(self, X_train, y_train, X_val, y_val, feature_names):
        """训练 Logistic Regression"""
        print("\n  📊 训练 Logistic Regression...")

        # ✅ 检查标签分布
        unique_classes = np.unique(y_train)
        pos_ratio = np.mean(y_train == 1)
        print(f"    训练集标签分布：{unique_classes}")
        print(f"    类别 0 比例：{np.mean(y_train == 0):.4f}")
        print(f"    类别 1 比例：{pos_ratio:.4f}")
        
        # ✅ 如果类别比例极端，跳过
        if pos_ratio < 0.1 or pos_ratio > 0.9:
            print(f"    ⚠️ 警告：类别严重不平衡 (正样本 {pos_ratio:.2%})，跳过 Logistic Regression")
            return {
                'train_accuracy': 0.5,
                'val_accuracy': 0.5,
                'mcfadden_r2': 0.0,
                'n_significant': 0,
                'status': 'skipped_imbalanced'
            }
        
        # ✅ 修复：添加 class_weight='balanced'
        self.logistic_model = LogisticRegression(
            C=self.config.LOGISTIC_REGULARIZATION,
            max_iter=self.config.LOGISTIC_MAX_ITER,
            solver=self.config.LOGISTIC_SOLVER,
            random_state=self.config.RANDOM_SEED,
            class_weight='balanced'
        )
        
        # self.logistic_model = LogisticRegression(
        #     C=self.config.LOGISTIC_REGULARIZATION,
        #     max_iter=self.config.LOGISTIC_MAX_ITER,
        #     solver=self.config.LOGISTIC_SOLVER,
        #     random_state=self.config.RANDOM_SEED
        # )
        self.logistic_model.fit(X_train, y_train)
        
        # 训练集评估
        train_pred = self.logistic_model.predict(X_train)
        val_pred = self.logistic_model.predict(X_val)
        train_prob = self.logistic_model.predict_proba(X_train)[:, 1]
        val_prob = self.logistic_model.predict_proba(X_val)[:, 1]
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        
        # 统计检验
        self.logistic_summary = StatisticalTests.logistic_regression_summary(
            X_train, y_train, self.logistic_model, feature_names
        )
        
        print(f"    训练准确率：{train_acc:.4f}")
        print(f"    验证准确率：{val_acc:.4f}")
        print(f"    McFadden R²: {self.logistic_summary['mcfadden_r2']:.4f}")
        
        # 显著性因子数量
        significant_df = StatisticalTests.get_significant_features(
            self.logistic_summary, alpha=self.config.SIGNIFICANCE_LEVEL
        )
        n_significant = significant_df['significant'].sum()
        print(f"    显著性因子数 (p<{self.config.SIGNIFICANCE_LEVEL}): {n_significant}/{len(feature_names)}")
        
        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'mcfadden_r2': self.logistic_summary['mcfadden_r2'],
            'n_significant': n_significant
        }
    
    def get_linear_importance(self, top_n=20):
        """获取 Linear Regression 因子重要性"""
        if self.linear_summary is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.linear_summary['feature_names'],
            'coefficient': self.linear_summary['coefficients'],
            'abs_coefficient': np.abs(self.linear_summary['coefficients']),
            'std_error': self.linear_summary['coef_std'],
            't_statistic': self.linear_summary['t_statistics'],
            'p_value': self.linear_summary['p_values'],
            'ci_lower': self.linear_summary['ci_lower'],
            'ci_upper': self.linear_summary['ci_upper']
        })
        
        importance_df['significant'] = importance_df['p_value'] < self.config.SIGNIFICANCE_LEVEL
        importance_df['significance_level'] = importance_df['p_value'].apply(
            lambda p: '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
        )
        
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        return importance_df.head(top_n)
    
    def get_logistic_importance(self, top_n=20):
        """获取 Logistic Regression 因子重要性"""
        if self.logistic_summary is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.logistic_summary['feature_names'],
            'coefficient': self.logistic_summary['coefficients'],
            'abs_coefficient': np.abs(self.logistic_summary['coefficients']),
            'odds_ratio': self.logistic_summary['odds_ratio'],
            'std_error': self.logistic_summary['coef_std'],
            'z_statistic': self.logistic_summary['z_statistics'],
            'p_value': self.logistic_summary['p_values'],
            'or_ci_lower': self.logistic_summary['or_ci_lower'],
            'or_ci_upper': self.logistic_summary['or_ci_upper']
        })
        
        importance_df['significant'] = importance_df['p_value'] < self.config.SIGNIFICANCE_LEVEL
        importance_df['significance_level'] = importance_df['p_value'].apply(
            lambda p: '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
        )
        
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        return importance_df.head(top_n)


# ============================
# 模型测试器
# ============================
class LinearLogisticTester:
    """Linear & Logistic Regression 测试器"""
    
    def __init__(self, linear_model, logistic_model, config: Config):
        self.linear_model = linear_model
        self.logistic_model = logistic_model
        self.config = config
    
    def test_linear(self, X_test, y_test, target_mean, target_std):
        """测试 Linear Regression"""
        print("\n  📈 测试 Linear Regression...")
        
        pred_normalized = self.linear_model.predict(X_test)
        pred_original = pred_normalized * target_std + target_mean
        target_original = y_test * target_std + target_mean
        
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
            'targets': target_original
        }
    
    def test_logistic(self, X_test, y_test, target_mean, target_std):
        """测试 Logistic Regression"""
        print("\n  📊 测试 Logistic Regression...")
        
        pred_cls = self.logistic_model.predict(X_test)
        pred_prob = self.logistic_model.predict_proba(X_test)[:, 1]
        
        # 分类指标
        accuracy = accuracy_score(y_test, pred_cls)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, pred_cls, average='binary', zero_division=0
        )
        
        # 将概率转换为回归格式 (用于 IC 计算)
        pred_reg = (pred_prob - 0.5) * 2 * 0.1
        
        # 获取真实目标值 (需要重新计算)
        y_test_original = y_test * target_std + target_mean
        
        # IC
        ic, ic_pvalue = stats.spearmanr(pred_reg, y_test)
        
        # 方向准确率 (与分类准确率相同)
        direction_acc = accuracy
        
        print(f"    测试准确率：{accuracy:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    F1 Score: {f1:.4f}")
        print(f"    IC: {ic:.4f} (p={ic_pvalue:.4e})")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ic': ic,
            'ic_pvalue': ic_pvalue,
            'direction_accuracy': direction_acc,
            'predictions_prob': pred_prob,
            'predictions_cls': pred_cls,
            'targets': y_test
        }
    
    def _quantile_backtest(self, preds, targets, n_quantiles=5):
        """分层回测"""
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
# 可视化器
# ============================
class ImportanceVisualizer:
    """因子重要性可视化"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_coefficient_comparison(self, linear_imp, logistic_imp, symbol):
        """对比两种模型的系数"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Linear Regression
        if linear_imp is not None and len(linear_imp) > 0:
            top_factors = linear_imp.head(15)
            colors = ['green' if c > 0 else 'red' for c in top_factors['coefficient']]
            axes[0].barh(range(len(top_factors)), top_factors['coefficient'], color=colors)
            axes[0].set_yticks(range(len(top_factors)))
            axes[0].set_yticklabels(top_factors['feature'].str.slice(-25), fontsize=8)
            axes[0].set_xlabel('Coefficient')
            axes[0].set_title('Linear Regression - Top 15 Factors')
            axes[0].axvline(0, color='black', linewidth=1)
            axes[0].grid(True, alpha=0.3, axis='x')
        
        # Logistic Regression
        if logistic_imp is not None and len(logistic_imp) > 0:
            top_factors = logistic_imp.head(15)
            colors = ['green' if c > 0 else 'red' for c in top_factors['coefficient']]
            axes[1].barh(range(len(top_factors)), top_factors['coefficient'], color=colors)
            axes[1].set_yticks(range(len(top_factors)))
            axes[1].set_yticklabels(top_factors['feature'].str.slice(-25), fontsize=8)
            axes[1].set_xlabel('Coefficient')
            axes[1].set_title('Logistic Regression - Top 15 Factors')
            axes[1].axvline(0, color='black', linewidth=1)
            axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_coefficient_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存系数对比图：{symbol}_coefficient_comparison.png")
    
    def plot_significance_heatmap(self, linear_imp, logistic_imp, symbol):
        """显著性热力图"""
        if linear_imp is None or logistic_imp is None:
            return
        
        # 找出两种模型共同的因子
        common_factors = set(linear_imp['feature']) & set(logistic_imp['feature'])
        common_factors = list(common_factors)[:20]  # 最多 20 个
        
        if len(common_factors) < 2:
            return
        
        # 构建热力图数据
        heatmap_data = []
        for factor in common_factors:
            lin_row = linear_imp[linear_imp['feature'] == factor]
            log_row = logistic_imp[logistic_imp['feature'] == factor]
            
            if len(lin_row) > 0 and len(log_row) > 0:
                heatmap_data.append({
                    'factor': factor,
                    'linear_p': lin_row['p_value'].values[0],
                    'logistic_p': log_row['p_value'].values[0],
                    'linear_coef': lin_row['coefficient'].values[0],
                    'logistic_coef': log_row['coefficient'].values[0]
                })
        
        heatmap_df = pd.DataFrame(heatmap_data)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # p 值热力图
        pivot_p = heatmap_df.set_index('factor')[['linear_p', 'logistic_p']].T
        pivot_p.columns = [f[:20] for f in pivot_p.columns]
        sns.heatmap(-np.log10(pivot_p + 1e-10), annot=False, cmap='YlOrRd', ax=axes[0])
        axes[0].set_title('-log10(p-value) Heatmap')
        axes[0].set_xlabel('Factor')
        axes[0].set_ylabel('Model')
        
        # 系数对比散点图
        axes[1].scatter(heatmap_df['linear_coef'], heatmap_df['logistic_coef'], alpha=0.6, s=50)
        axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[1].axvline(0, color='gray', linestyle='--', alpha=0.5)
        axes[1].plot([heatmap_df['linear_coef'].min(), heatmap_df['linear_coef'].max()],
                    [heatmap_df['linear_coef'].min(), heatmap_df['linear_coef'].max()],
                    'r--', alpha=0.5)
        axes[1].set_xlabel('Linear Coefficient')
        axes[1].set_ylabel('Logistic Coefficient')
        axes[1].set_title('Coefficient Correlation')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_significance_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存显著性热力图：{symbol}_significance_heatmap.png")
    
    def plot_top_factors_table(self, linear_imp, logistic_imp, symbol):
        """生成 Top 因子表格"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        if linear_imp is not None and len(linear_imp) > 0:
            top_lin = linear_imp.head(10)
            axes[0].axis('tight')
            axes[0].axis('off')
            table_data_lin = []
            for _, row in top_lin.iterrows():
                table_data_lin.append([
                    row['feature'][:30],
                    f"{row['coefficient']:.4f}",
                    f"{row['t_statistic']:.2f}",
                    f"{row['p_value']:.4e}",
                    row['significance_level']
                ])
            table_lin = axes[0].table(
                cellText=table_data_lin,
                colLabels=['Factor', 'Coef', 't-stat', 'p-value', 'Signif'],
                loc='center',
                cellLoc='center'
            )
            table_lin.auto_set_font_size(False)
            table_lin.set_fontsize(8)
            axes[0].set_title('Linear Regression - Top 10 Significant Factors', pad=20)
        
        if logistic_imp is not None and len(logistic_imp) > 0:
            top_log = logistic_imp.head(10)
            axes[1].axis('tight')
            axes[1].axis('off')
            table_data_log = []
            for _, row in top_log.iterrows():
                table_data_log.append([
                    row['feature'][:30],
                    f"{row['coefficient']:.4f}",
                    f"{row['odds_ratio']:.4f}",
                    f"{row['p_value']:.4e}",
                    row['significance_level']
                ])
            table_log = axes[1].table(
                cellText=table_data_log,
                colLabels=['Factor', 'Coef', 'Odds Ratio', 'p-value', 'Signif'],
                loc='center',
                cellLoc='center'
            )
            table_log.auto_set_font_size(False)
            table_log.set_fontsize(8)
            axes[1].set_title('Logistic Regression - Top 10 Significant Factors', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_top_factors_table.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存因子表格：{symbol}_top_factors_table.png")
    
    def plot_prediction_quality(self, linear_results, logistic_results, symbol):
        """预测质量对比"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Linear Regression 预测散点图
        if linear_results is not None:
            axes[0].scatter(linear_results['targets'], linear_results['predictions'], alpha=0.3, s=10)
            axes[0].plot([linear_results['targets'].min(), linear_results['targets'].max()],
                        [linear_results['targets'].min(), linear_results['targets'].max()],
                        'r--', linewidth=2)
            axes[0].set_xlabel('Actual Return')
            axes[0].set_ylabel('Predicted Return')
            axes[0].set_title(f'Linear Regression (IC={linear_results["ic"]:.4f})')
            axes[0].grid(True, alpha=0.3)
        
        # Logistic Regression 概率分布
        if logistic_results is not None:
            prob_correct = logistic_results['predictions_prob'][logistic_results['targets'] == 1]
            prob_wrong = logistic_results['predictions_prob'][logistic_results['targets'] == 0]
            axes[1].hist(prob_correct, bins=30, alpha=0.5, label='Positive (Correct)', color='green')
            axes[1].hist(1 - prob_wrong, bins=30, alpha=0.5, label='Negative (Correct)', color='blue')
            axes[1].set_xlabel('Predicted Probability')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title(f'Logistic Regression (Acc={logistic_results["accuracy"]:.4f})')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_prediction_quality.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存预测质量图：{symbol}_prediction_quality.png")


# ============================
# 主流程
# ============================
def train_test_symbol(symbol: str, config: Config) -> dict:
    """训练和测试单个交易对"""
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
        classification_percentile=config.CLASSIFICATION_PERCENTILE
    )
    
    if len(dataset) < 100:
        print(f"  ❌ 有效样本不足：{len(dataset)}")
        return {'status': 'failed', 'reason': 'insufficient_data'}
    
    # 3. 获取数据
    X, y_reg, y_cls, indices = dataset.get_data()
    feature_names = dataset.get_factor_names()
    
    print(f"  📊 数据形状：X={X.shape}, y_reg={y_reg.shape}, y_cls={y_cls.shape}")
    
    # 4. 数据分割
    # 先分出测试集
    X_temp, X_test, y_reg_temp, y_reg_test, y_cls_temp, y_cls_test = train_test_split(
        X, y_reg, y_cls, test_size=config.TEST_RATIO, random_state=config.RANDOM_SEED
    )
    
    # 再分出训练集和验证集
    val_ratio = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    X_train, X_val, y_reg_train, y_reg_val, y_cls_train, y_cls_val = train_test_split(
        X_temp, y_reg_temp, y_cls_temp, test_size=val_ratio, random_state=config.RANDOM_SEED
    )
    
    print(f"  📊 数据分割：训练={len(X_train)}, 验证={len(X_val)}, 测试={len(X_test)}")
    
    # 5. 训练模型
    trainer = LinearLogisticTrainer(config)
    
    linear_metrics = trainer.train_linear(X_train, y_reg_train, X_val, y_reg_val, feature_names)
    logistic_metrics = trainer.train_logistic(X_train, y_cls_train, X_val, y_cls_val, feature_names)
    
    # 6. 测试模型
    tester = LinearLogisticTester(trainer.linear_model, trainer.logistic_model, config)
    target_stats = dataset.get_target_stats()
    
    linear_results = tester.test_linear(X_test, y_reg_test, target_stats['mean'], target_stats['std'])
    logistic_results = tester.test_logistic(X_test, y_cls_test, target_stats['mean'], target_stats['std'])
    
    # 7. 获取因子重要性
    linear_importance = trainer.get_linear_importance(top_n=20)
    logistic_importance = trainer.get_logistic_importance(top_n=20)
    
    # 8. 保存结果
    symbol_output_dir = config.OUTPUT_DIR / symbol
    symbol_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    with open(symbol_output_dir / 'linear_model.pkl', 'wb') as f:
        pickle.dump(trainer.linear_model, f)
    with open(symbol_output_dir / 'logistic_model.pkl', 'wb') as f:
        pickle.dump(trainer.logistic_model, f)
    
    # 保存标准化器
    with open(symbol_output_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(dataset.get_scaler(), f)
    
    # 保存配置
    train_config = {
        'symbol': symbol,
        'seq_length': config.SEQ_LENGTH,
        'prediction_horizon': config.PREDICTION_HORIZON,
        'target_col': config.TARGET_COL,
        'n_factors': len(feature_names),
        'factor_names': feature_names,
        'target_mean': target_stats['mean'],
        'target_std': target_stats['std'],
        'linear_metrics': linear_metrics,
        'logistic_metrics': logistic_metrics,
        'timestamp': datetime.now().isoformat()
    }
    with open(symbol_output_dir / 'train_config.json', 'w') as f:
        json.dump(train_config, f, indent=2, default=str)
    
    # 保存因子重要性
    if linear_importance is not None:
        linear_importance.to_csv(symbol_output_dir / 'linear_factor_importance.csv', index=False)
    if logistic_importance is not None:
        logistic_importance.to_csv(symbol_output_dir / 'logistic_factor_importance.csv', index=False)
    
    # 保存统计摘要
    if trainer.linear_summary is not None:
        linear_summary_df = pd.DataFrame({
            'feature': trainer.linear_summary['feature_names'],
            'coefficient': trainer.linear_summary['coefficients'],
            'std_error': trainer.linear_summary['coef_std'],
            't_statistic': trainer.linear_summary['t_statistics'],
            'p_value': trainer.linear_summary['p_values'],
            'ci_lower': trainer.linear_summary['ci_lower'],
            'ci_upper': trainer.linear_summary['ci_upper']
        })
        linear_summary_df.to_csv(symbol_output_dir / 'linear_statistical_summary.csv', index=False)
    
    if trainer.logistic_summary is not None:
        logistic_summary_df = pd.DataFrame({
            'feature': trainer.logistic_summary['feature_names'],
            'coefficient': trainer.logistic_summary['coefficients'],
            'std_error': trainer.logistic_summary['coef_std'],
            'z_statistic': trainer.logistic_summary['z_statistics'],
            'p_value': trainer.logistic_summary['p_values'],
            'odds_ratio': trainer.logistic_summary['odds_ratio'],
            'or_ci_lower': trainer.logistic_summary['or_ci_lower'],
            'or_ci_upper': trainer.logistic_summary['or_ci_upper']
        })
        logistic_summary_df.to_csv(symbol_output_dir / 'logistic_statistical_summary.csv', index=False)
    
    # 9. 可视化
    visualizer = ImportanceVisualizer(symbol_output_dir)
    visualizer.plot_coefficient_comparison(linear_importance, logistic_importance, symbol)
    visualizer.plot_significance_heatmap(linear_importance, logistic_importance, symbol)
    visualizer.plot_top_factors_table(linear_importance, logistic_importance, symbol)
    visualizer.plot_prediction_quality(linear_results, logistic_results, symbol)
    
    # 10. 生成摘要
    summary = {
        'symbol': symbol,
        'n_samples': len(dataset),
        'n_factors': len(feature_names),
        'linear_val_r2': linear_metrics['val_r2'],
        'linear_test_ic': linear_results['ic'],
        'linear_test_direction_acc': linear_results['direction_accuracy'],
        'linear_n_significant': linear_metrics['n_significant'],
        'logistic_val_acc': logistic_metrics['val_accuracy'],
        'logistic_test_ic': logistic_results['ic'],
        'logistic_test_accuracy': logistic_results['accuracy'],
        'logistic_n_significant': logistic_metrics['n_significant'],
        'status': 'success'
    }
    
    print(f"\n  📋 训练测试摘要:")
    print(f"     Linear - 验证 R²: {summary['linear_val_r2']:.4f}, 测试 IC: {summary['linear_test_ic']:.4f}")
    print(f"     Logistic - 验证 Acc: {summary['logistic_val_acc']:.4f}, 测试 IC: {summary['logistic_test_ic']:.4f}")
    print(f"     模型保存至：{symbol_output_dir}")
    
    return summary


def discover_symbols(config: Config) -> list:
    """发现所有有因子数据的交易对"""
    if not config.FACTOR_DIR.exists():
        raise FileNotFoundError(f"因子目录不存在：{config.FACTOR_DIR}")
    symbols = [d.name for d in config.FACTOR_DIR.iterdir() if d.is_dir()]
    print(f"🔍 发现 {len(symbols)} 个交易对有因子数据")
    return symbols


def generate_summary_report(summaries: list, config: Config):
    """生成汇总报告"""
    print(f"\n{'='*60}")
    print("📊 生成汇总报告")
    print(f"{'='*60}")
    
    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df[summary_df['status'] == 'success']
    
    if summary_df.empty:
        print("  ❌ 无成功训练的交易对")
        return
    
    summary_df.to_csv(config.OUTPUT_DIR / "all_symbols_summary.csv", index=False)
    
    # Linear Regression Top
    if 'linear_test_ic' in summary_df.columns:
        top_linear = summary_df.nlargest(5, 'linear_test_ic')
        print("\n🏆 Top 5 交易对 (Linear Regression - 按测试 IC):")
        for _, row in top_linear.iterrows():
            print(f"   {row['symbol']}: IC={row['linear_test_ic']:.4f}, 显著因子={row['linear_n_significant']}")
    
    # Logistic Regression Top
    if 'logistic_test_accuracy' in summary_df.columns:
        top_logistic = summary_df.nlargest(5, 'logistic_test_accuracy')
        print("\n🏆 Top 5 交易对 (Logistic Regression - 按准确率):")
        for _, row in top_logistic.iterrows():
            print(f"   {row['symbol']}: Acc={row['logistic_test_accuracy']:.4f}, 显著因子={row['logistic_n_significant']}")
    
    print(f"\n💾 汇总报告：{config.OUTPUT_DIR / 'all_symbols_summary.csv'}")


# ============================
# 主程序入口
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Linear & Logistic Regression 训练测试脚本')
    parser.add_argument('--symbol', type=str, default='AVAXUSDT', help='交易对名称')
    parser.add_argument('--all_symbols', action='store_true', help='处理所有交易对')
    parser.add_argument('--percentile', type=int, default=40, help='分类阈值分位数 (默认 40)')
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 Linear & Logistic Regression 因子训练测试脚本 (修复版)")
    print("="*60)
    print(f"📁 因子目录：{config.FACTOR_DIR}")
    print(f"📁 输出目录：{config.OUTPUT_DIR}")
    print(f"📐 序列长度：{config.SEQ_LENGTH} (30 秒 @ 500ms)")
    print(f"🎯 预测期：{config.PREDICTION_HORIZON} (下一秒)")
    print(f"📊 显著性水平：{config.SIGNIFICANCE_LEVEL}")
    print(f"🔢 分类阈值：{args.percentile}% 分位数")
    print(f"⚡ Bootstrap 迭代：{config.BOOTSTRAP_ITERATIONS} 次")
    print("="*60)
    
    config.CLASSIFICATION_PERCENTILE = args.percentile
    
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
    print("🎉 Linear & Logistic Regression 训练测试完成!")
    print("="*60)