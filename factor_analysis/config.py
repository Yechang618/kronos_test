#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子分析配置文件
"""

# 因子类别开关
FACTOR_CONFIG = {
    'momentum': True,          # 动量类因子
    'volatility': True,        # 波动率类因子
    'mean_reversion': True,    # 均值回归类因子
    'volume_based': True,      # 成交量/资金类因子
    'basis_specific': True,    # 基差特异性因子
    'technical_patterns': True # 技术形态因子
}

# 分析配置
ANALYSIS_CONFIG = {
    'ic_analysis': True,           # IC 分析
    'quantile_backtest': True,     # 分层回测
    'correlation_matrix': True,    # 相关性矩阵
    'factor_decay': True,          # 因子衰减分析
    'top_n_factors': 10,           # 分析 Top N 因子
    'quantiles': 5                 # 分层数量
}

# 数据配置
DATA_CONFIG = {
    'min_rows': 1000,              # 最小数据行数
    'max_nan_ratio': 0.3,          # 最大 NaN 比例
    'outlier_std': 5               # 异常值标准差阈值
}

# 因子有效性阈值
FACTOR_THRESHOLD = {
    'min_ic': 0.02,                # 最小 IC 绝对值
    'min_icir': 0.3,               # 最小 ICIR
    'min_t_stat': 2.0,             # 最小 t 统计量
    'max_correlation': 0.9         # 最大因子相关性 (去冗余)
}