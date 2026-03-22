import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import warnings
import os
import multiprocessing
from pathlib import Path
import glob
import time
from scipy import stats
from sklearn.utils import resample

# ================= 环境变量设置 =================
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings('ignore')

# ================= PyTorch 导入 =================
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    print("✅ PyTorch 可用")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch 不可用，CNN 显著性检验将禁用")
# ===============================================

# ================= 配置区域 =================
DATASET_DIR = "./dataset"
DATASET_PATTERN = "./dataset/samples_*.csv"

# OOS 分割方式：'time' 或 'symbol'
OOS_SPLIT_MODE = 'symbol'  # 🔹 按 symbol 分割（跨资产泛化检验）
TRAIN_SYMBOL_RATIO = 0.7   # 70% symbol 用于训练
SIGNIFICANCE_LEVEL = 0.05  # 显著性水平 α
N_BOOTSTRAP = 50           # Bootstrap 重采样次数（可减小加速）
N_JOBS = 1
RANDOM_STATE = 42

# CNN 配置
CNN_SEQUENCE_LENGTH = 10
CNN_N_SAMPLES = 50  # 梯度检验采样数
# ===========================================


# ================= CNN 模型定义 =================
class SlippageCNN(nn.Module):
    def __init__(self, num_features, sequence_length, hidden_dim=64, 
                 num_filters=32, kernel_size=3, dropout=0.3):
        super(SlippageCNN, self).__init__()
        self.conv1 = nn.Conv1d(num_features, num_filters, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(num_filters*2)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = None
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, 1)
        self.hidden_dim = hidden_dim
        self.num_filters = num_filters
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.dropout(x)
        x = self.conv2(x); x = self.bn2(x); x = self.relu(x); x = self.pool(x); x = self.dropout(x)
        x = x.flatten(start_dim=1)
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], self.hidden_dim).to(x.device)
        x = self.fc1(x); x = self.relu(x); x = self.dropout(x)
        x = self.fc2(x); x = self.relu(x); x = self.fc3(x)
        return x.squeeze()
# ===============================================


def load_all_symbol_data(dataset_dir, dataset_pattern):
    """加载多 symbol 数据"""
    csv_files = glob.glob(dataset_pattern)
    if not csv_files:
        default_path = os.path.join(dataset_dir, "processed_training_set.csv")
        if os.path.exists(default_path):
            csv_files = [default_path]
        else:
            raise FileNotFoundError(f"No dataset found at {dataset_pattern}")
    
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if 'symbol' in df.columns:
                dfs.append(df)
        except:
            continue
    
    if not dfs:
        raise FileNotFoundError("No valid data files loaded.")
    
    full_df = pd.concat(dfs, ignore_index=True)
    return full_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)


def prepare_features(df):
    """准备特征和标签"""
    feature_cols = [c for c in df.columns if c.startswith('factor_')]
    # 修复因子名称中的空格
    cleaned_cols = []
    for col in feature_cols:
        clean_col = col.replace(' ', '')
        if clean_col in df.columns:
            cleaned_cols.append(clean_col)
        elif col in df.columns:
            cleaned_cols.append(col)
    
    X = df[cleaned_cols].fillna(0)
    y = df['label_slippage']
    return X, y, cleaned_cols


def split_data_oos(df, mode='symbol', train_ratio=0.7, random_state=42):
    """
    Out-of-Sample 数据分割
    mode: 'time' - 按时间分割; 'symbol' - 按交易对分割
    """
    np.random.seed(random_state)
    
    if mode == 'symbol':
        # 按 symbol 分割（跨资产泛化检验）
        all_symbols = df['symbol'].unique()
        n_train = max(1, int(len(all_symbols) * train_ratio))
        train_symbols = np.random.choice(all_symbols, n_train, replace=False)
        test_symbols = np.array([s for s in all_symbols if s not in train_symbols])
        
        train_df = df[df['symbol'].isin(train_symbols)].copy()
        test_df = df[df['symbol'].isin(test_symbols)].copy()
        
        print(f"📐 Symbol OOS Split:")
        print(f"  Train symbols ({len(train_symbols)}): {list(train_symbols)[:5]}{'...' if len(train_symbols)>5 else ''}")
        print(f"  Test symbols ({len(test_symbols)}): {list(test_symbols)[:5]}{'...' if len(test_symbols)>5 else ''}")
        
    elif mode == 'time':
        # 按时间分割（时间序列泛化检验）
        df = df.sort_values('timestamp')
        n = len(df)
        train_end = int(n * train_ratio)
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[train_end:].copy()
        
        print(f"📐 Time OOS Split:")
        print(f"  Train period: {train_df['timestamp'].min()} - {train_df['timestamp'].max()}")
        print(f"  Test period: {test_df['timestamp'].min()} - {test_df['timestamp'].max()}")
    else:
        raise ValueError(f"Unknown split mode: {mode}")
    
    print(f"  Train samples: {len(train_df):,}, Test samples: {len(test_df):,}")
    return train_df, test_df


def create_sequence_data(X, y, df, sequence_length=10):
    """为 CNN 创建序列数据 - 修复维度问题"""
    print(f"  创建序列数据 (length={sequence_length})...")
    
    # 处理不同类型输入
    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = X
    
    if isinstance(y, pd.Series):
        y_values = y.values
    else:
        y_values = y
    
    X_seq, y_seq, valid_idx = [], [], []
    df_copy = df.copy()
    df_copy['original_idx'] = range(len(df))
    
    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        sym_df = df_copy[mask].sort_values('timestamp').reset_index(drop=True)
        sym_X = X_values[mask]
        sym_y = y_values[mask]
        
        for i in range(sequence_length, len(sym_df)):
            X_seq.append(sym_X[i-sequence_length:i])
            y_seq.append(sym_y[i])
            valid_idx.append(sym_df.iloc[i]['original_idx'])
    
    return np.array(X_seq), np.array(y_seq), np.array(valid_idx)


# ================= LR OOS 显著性检验 =================
def test_lr_oos_significance(X_train, y_train, X_test, y_test, feature_cols, alpha=0.05):
    """
    Linear Regression OOS 显著性检验
    1. 训练集拟合模型，获取系数
    2. 测试集计算系数标准误和 t 统计量
    3. 检验训练集显著因子在测试集是否保持显著
    """
    print(f"\n🔍 LR OOS 显著性检验 (α={alpha})")
    
    # 1. 训练集拟合
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 2. 训练集显著性（基准）
    n_train = len(X_train)
    n_features = len(feature_cols)
    y_train_pred = model.predict(X_train)
    residuals_train = y_train - y_train_pred
    mse_train = np.sum(residuals_train**2) / (n_train - n_features - 1)
    
    X_train_const = np.column_stack([np.ones(n_train), X_train.values])
    XtX_inv = np.linalg.pinv(X_train_const.T @ X_train_const)
    
    train_results = []
    for i, col in enumerate(feature_cols):
        coef = model.coef_[i]
        se_train = np.sqrt(mse_train * XtX_inv[i+1, i+1])
        t_train = coef / se_train if se_train > 0 else 0
        p_train = 2 * (1 - stats.t.cdf(abs(t_train), df=n_train-n_features-1))
        train_results.append({
            'feature': col, 'coef': coef, 'se_train': se_train, 
            't_train': t_train, 'p_train': p_train, 'sig_train': p_train < alpha
        })
    
    # 3. 测试集验证显著性（使用训练集系数，测试集残差估计标准误）
    n_test = len(X_test)
    y_test_pred = model.predict(X_test)
    residuals_test = y_test - y_test_pred
    mse_test = np.sum(residuals_test**2) / (n_test - n_features - 1) if n_test > n_features + 1 else mse_train
    
    # 使用训练集的 (X'X)^-1 和测试集 MSE 计算测试集标准误
    oos_results = []
    for i, col in enumerate(feature_cols):
        coef = model.coef_[i]
        se_oos = np.sqrt(mse_test * XtX_inv[i+1, i+1])  # OOS 标准误
        t_oos = coef / se_oos if se_oos > 0 else 0
        p_oos = 2 * (1 - stats.t.cdf(abs(t_oos), df=n_test-n_features-1)) if n_test > n_features + 1 else 1.0
        
        oos_results.append({
            'feature': col, 'coef': coef, 'se_oos': se_oos,
            't_oos': t_oos, 'p_oos': p_oos, 'sig_oos': p_oos < alpha
        })
    
    # 合并结果
    results_df = pd.DataFrame(train_results)
    results_df['p_oos'] = [r['p_oos'] for r in oos_results]
    results_df['sig_oos'] = [r['sig_oos'] for r in oos_results]
    results_df['oos_persistence'] = results_df['sig_train'] & results_df['sig_oos']
    
    # 打印结果
    train_sig = results_df['sig_train'].sum()
    oos_sig = results_df['sig_oos'].sum()
    persistent = results_df['oos_persistence'].sum()
    
    print(f"  训练集显著因子：{train_sig}/{len(feature_cols)} ({train_sig/len(feature_cols):.1%})")
    print(f"  测试集显著因子：{oos_sig}/{len(feature_cols)} ({oos_sig/len(feature_cols):.1%})")
    print(f"  🔹 OOS 保持显著：{persistent}/{train_sig} ({persistent/max(1,train_sig):.1%} of train-significant)")
    
    if persistent > 0:
        print(f"\n  📊 OOS 持续显著因子 (Top 10):")
        for _, row in results_df[results_df['oos_persistence']].sort_values('p_oos').head(10).iterrows():
            print(f"    • {row['feature']}: coef={row['coef']:+.4f}, p_train={row['p_train']:.4f}, p_oos={row['p_oos']:.4f}")
    
    return model, results_df


# ================= LGBM OOS 显著性检验 =================
def test_lgbm_oos_significance(X_train, y_train, X_test, y_test, feature_cols, 
                              model=None, n_bootstrap=50, alpha=0.05, random_state=42):
    """
    LightGBM OOS 显著性检验
    1. 训练集计算特征重要性（Permutation）
    2. 测试集重新计算重要性，检验是否保持
    """
    print(f"\n🔍 LGBM OOS 显著性检验 (Bootstrap={n_bootstrap}, α={alpha})")
    
    np.random.seed(random_state)
    
    # 训练模型
    if model is None:
        model = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=-1,
            subsample=0.8, colsample_bytree=0.8, random_state=random_state,
            verbosity=-1, n_jobs=N_JOBS, force_col_wise=True
        )
        model.fit(X_train, y_train)
    
    # 训练集基线分数
    baseline_train = -mean_squared_error(y_train, model.predict(X_train))
    
    # 训练集 Permutation Importance
    train_importance = {}
    for col in feature_cols:
        X_perm = X_train.copy()
        X_perm[col] = np.random.permutation(X_perm[col].values)
        perm_score = -mean_squared_error(y_train, model.predict(X_perm))
        train_importance[col] = baseline_train - perm_score
    
    # 测试集基线分数
    baseline_test = -mean_squared_error(y_test, model.predict(X_test))
    
    # 测试集 Permutation Importance + Bootstrap CI
    oos_importance_samples = {col: [] for col in feature_cols}
    
    for b in range(n_bootstrap):
        # 测试集重采样
        X_boot, y_boot = resample(X_test, y_test, random_state=random_state+b)
        
        for col in feature_cols:
            X_perm = X_boot.copy()
            X_perm[col] = np.random.permutation(X_perm[col].values)
            perm_score = -mean_squared_error(y_boot, model.predict(X_perm))
            importance = baseline_test - perm_score
            oos_importance_samples[col].append(importance)
    
    # 计算统计量
    results = []
    for col in feature_cols:
        train_imp = train_importance[col]
        oos_imp_values = np.array(oos_importance_samples[col])
        mean_oos_imp = np.mean(oos_imp_values)
        std_oos_imp = np.std(oos_imp_values)
        
        # p-value: 检验测试集重要性是否显著 > 0
        p_oos = np.mean(oos_imp_values <= 0)
        
        # 持久性：训练集和测试集重要性同号且测试集显著
        persistent = (train_imp * mean_oos_imp > 0) and (p_oos < alpha)
        
        results.append({
            'feature': col,
            'train_importance': train_imp,
            'oos_mean_importance': mean_oos_imp,
            'oos_std_importance': std_oos_imp,
            'oos_p_value': p_oos,
            'oos_significant': p_oos < alpha,
            'oos_persistent': persistent
        })
    
    results_df = pd.DataFrame(results).sort_values('train_importance', ascending=False)
    
    # 打印结果
    train_sig = (results_df['train_importance'] > 0).sum()
    oos_sig = results_df['oos_significant'].sum()
    persistent = results_df['oos_persistent'].sum()
    
    print(f"  训练集重要因子 (imp>0)：{train_sig}/{len(feature_cols)}")
    print(f"  测试集显著因子 (p<{alpha})：{oos_sig}/{len(feature_cols)}")
    print(f"  🔹 OOS 持久因子：{persistent}/{train_sig} ({persistent/max(1,train_sig):.1%} of train-important)")
    
    if persistent > 0:
        print(f"\n  📊 OOS 持久因子 (Top 10):")
        for _, row in results_df[results_df['oos_persistent']].head(10).iterrows():
            print(f"    • {row['feature']}: train_imp={row['train_importance']:+.4f}, oos_imp={row['oos_mean_importance']:+.4f}±{row['oos_std_importance']:.4f}, p_oos={row['oos_p_value']:.4f}")
    
    return model, results_df


# ================= CNN OOS 显著性检验 =================
def test_cnn_oos_significance(X_train_seq, y_train_seq, X_test_seq, y_test_seq, 
                              feature_cols, num_features, sequence_length, 
                              device='cpu', n_samples=50, alpha=0.05):
    """
    CNN OOS 显著性检验
    1. 训练集计算梯度重要性
    2. 测试集验证梯度重要性是否保持
    """
    if not TORCH_AVAILABLE:
        print("⚠️  PyTorch 不可用，跳过 CNN OOS 检验")
        return None, pd.DataFrame()
    
    print(f"\n🔍 CNN OOS 显著性检验 (Gradient-based, n_samples={n_samples})")
    
    # 修复：确保特征数量匹配
    actual_num_features = X_train_seq.shape[2]
    if actual_num_features != len(feature_cols):
        print(f"  ⚠️  特征数不匹配：actual={actual_num_features}, expected={len(feature_cols)}")
        if actual_num_features < len(feature_cols):
            feature_cols = feature_cols[:actual_num_features]
        else:
            feature_cols = feature_cols + [f'extra_{i}' for i in range(actual_num_features - len(feature_cols))]
    
    # 训练模型
    model = SlippageCNN(actual_num_features, sequence_length).to(device)
    model.eval()
    
    # 训练集梯度重要性
    train_gradients = {col: [] for col in feature_cols}
    n_train_use = min(n_samples, len(X_train_seq))
    train_indices = np.random.choice(len(X_train_seq), n_train_use, replace=False)
    
    for i in train_indices:
        x = torch.FloatTensor(X_train_seq[i:i+1]).to(device).requires_grad_(True)
        y = torch.FloatTensor([y_train_seq[i]]).to(device)
        output = model(x)
        loss = ((output - y)**2).sum()
        
        model.zero_grad()
        loss.backward()
        
        if x.grad is not None:
            grad = x.grad.cpu().detach().numpy()[0]
            for f_idx, col in enumerate(feature_cols):
                if f_idx < grad.shape[0]:
                    train_gradients[col].append(np.mean(np.abs(grad[f_idx, :])))
    
    # 测试集梯度重要性
    oos_gradients = {col: [] for col in feature_cols}
    n_test_use = min(n_samples, len(X_test_seq))
    test_indices = np.random.choice(len(X_test_seq), n_test_use, replace=False)
    
    for i in test_indices:
        x = torch.FloatTensor(X_test_seq[i:i+1]).to(device).requires_grad_(True)
        y = torch.FloatTensor([y_test_seq[i]]).to(device)
        output = model(x)
        loss = ((output - y)**2).sum()
        
        model.zero_grad()
        loss.backward()
        
        if x.grad is not None:
            grad = x.grad.cpu().detach().numpy()[0]
            for f_idx, col in enumerate(feature_cols):
                if f_idx < grad.shape[0]:
                    oos_gradients[col].append(np.mean(np.abs(grad[f_idx, :])))
    
    # 计算统计量
    results = []
    for col in feature_cols:
        train_grad = np.mean(train_gradients[col]) if train_gradients[col] else 0
        oos_grad_values = np.array(oos_gradients[col])
        mean_oos_grad = np.mean(oos_grad_values) if len(oos_grad_values) > 0 else 0
        std_oos_grad = np.std(oos_grad_values) if len(oos_grad_values) > 1 else 0
        
        # p-value
        if len(oos_grad_values) > 1 and std_oos_grad > 0:
            _, p_oos = stats.ttest_1samp(oos_grad_values, 0, alternative='greater')
        else:
            p_oos = 1.0
        
        # 持久性
        persistent = (train_grad > 0) and (mean_oos_grad > 0) and (p_oos < alpha)
        
        results.append({
            'feature': col,
            'train_gradient': train_grad,
            'oos_mean_gradient': mean_oos_grad,
            'oos_std_gradient': std_oos_grad,
            'oos_p_value': p_oos,
            'oos_significant': p_oos < alpha,
            'oos_persistent': persistent
        })
    
    results_df = pd.DataFrame(results).sort_values('train_gradient', ascending=False)
    
    # 打印结果
    train_sig = (results_df['train_gradient'] > 0).sum()
    oos_sig = results_df['oos_significant'].sum()
    persistent = results_df['oos_persistent'].sum()
    
    print(f"  训练集重要因子 (grad>0)：{train_sig}/{len(feature_cols)}")
    print(f"  测试集显著因子 (p<{alpha})：{oos_sig}/{len(feature_cols)}")
    print(f"  🔹 OOS 持久因子：{persistent}/{train_sig} ({persistent/max(1,train_sig):.1%} of train-important)")
    
    if persistent > 0:
        print(f"\n  📊 OOS 持久因子 (Top 10):")
        for _, row in results_df[results_df['oos_persistent']].head(10).iterrows():
            print(f"    • {row['feature']}: train_grad={row['train_gradient']:+.4f}, oos_grad={row['oos_mean_gradient']:+.4f}±{row['oos_std_gradient']:.4f}, p_oos={row['oos_p_value']:.4f}")
    
    return model, results_df


# ================= 可视化 =================
def plot_oos_significance(lr_results, lgbm_results, cnn_results, feature_cols, 
                          split_mode, output_dir='./dataset'):
    """可视化 OOS 显著性结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 图 1: OOS 持久因子数量对比
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    models = ['LR', 'LGBM', 'CNN']
    persistent_counts = []
    
    if lr_results is not None and not lr_results.empty:
        persistent_counts.append(lr_results['oos_persistence'].sum())
    else:
        persistent_counts.append(0)
    
    if lgbm_results is not None and not lgbm_results.empty:
        persistent_counts.append(lgbm_results['oos_persistent'].sum())
    else:
        persistent_counts.append(0)
    
    if cnn_results is not None and not cnn_results.empty:
        persistent_counts.append(cnn_results['oos_persistent'].sum())
    else:
        persistent_counts.append(0)
    
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    bars = ax1.bar(models, persistent_counts, color=colors, edgecolor='black')
    ax1.set_ylabel('Number of OOS-Persistent Factors')
    ax1.set_title(f'OOS-Persistent Factors by Model (Split: {split_mode}, α={SIGNIFICANCE_LEVEL})')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, cnt in zip(bars, persistent_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{int(cnt)}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/oos_persistent_count.png', dpi=150)
    plt.close()
    print(f"  ✓ 保存：{output_dir}/oos_persistent_count.png")
    
    # 图 2: 训练集 vs 测试集 重要性/系数对比（LR）
    if lr_results is not None and not lr_results.empty:
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        top_features = lr_results.nlargest(15, 'oos_persistence')['feature']
        
        x = np.arange(len(top_features))
        width = 0.35
        
        coefs_train = lr_results.set_index('feature').loc[top_features, 'coef'].values
        # 用系数符号 * -log10(p) 表示显著性强度
        sig_train = -np.log10(lr_results.set_index('feature').loc[top_features, 'p_train'] + 1e-10) * np.sign(coefs_train)
        sig_oos = -np.log10(lr_results.set_index('feature').loc[top_features, 'p_oos'] + 1e-10) * np.sign(coefs_train)
        
        ax2.barh(x - width/2, sig_train, width, label='Train Significance', alpha=0.7, color='steelblue')
        ax2.barh(x + width/2, sig_oos, width, label='OOS Significance', alpha=0.7, color='coral')
        
        ax2.set_yticks(x)
        ax2.set_yticklabels(top_features)
        ax2.set_xlabel('-log10(p-value) × sign(coef)')
        ax2.set_title('LR Factor Significance: Train vs OOS\n(Positive=Positive Coef, Negative=Negative Coef)')
        ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1)
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/oos_lr_significance_compare.png', dpi=150)
        plt.close()
        print(f"  ✓ 保存：{output_dir}/oos_lr_significance_compare.png")
    
    # 图 3: 多模型共同持久的因子
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    persistent_sets = {}
    if lr_results is not None and not lr_results.empty:
        persistent_sets['LR'] = set(lr_results[lr_results['oos_persistence']]['feature'])
    if lgbm_results is not None and not lgbm_results.empty:
        persistent_sets['LGBM'] = set(lgbm_results[lgbm_results['oos_persistent']]['feature'])
    if cnn_results is not None and not cnn_results.empty:
        persistent_sets['CNN'] = set(cnn_results[cnn_results['oos_persistent']]['feature'])
    
    if len(persistent_sets) >= 2:
        # 计算交集
        all_persistent = set.intersection(*persistent_sets.values()) if len(persistent_sets) > 1 else set()
        
        categories = []
        counts = []
        for model in persistent_sets:
            categories.append(f'{model} only')
            counts.append(len(persistent_sets[model] - set.union(*(s for m,s in persistent_sets.items() if m!=model))))
        
        if len(persistent_sets) == 2:
            categories.append('Both')
            counts.append(len(all_persistent))
        elif len(persistent_sets) == 3:
            lr_lgbm = persistent_sets['LR'] & persistent_sets['LGBM'] - persistent_sets.get('CNN', set())
            lr_cnn = persistent_sets['LR'] & persistent_sets['CNN'] - persistent_sets.get('LGBM', set())
            lgbm_cnn = persistent_sets['LGBM'] & persistent_sets['CNN'] - persistent_sets.get('LR', set())
            categories.extend(['LR+LGBM', 'LR+CNN', 'LGBM+CNN', 'All 3'])
            counts.extend([len(lr_lgbm), len(lr_cnn), len(lgbm_cnn), len(all_persistent)])
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        ax3.barh(categories, counts, color=colors, edgecolor='black')
        ax3.set_xlabel('Number of Factors')
        ax3.set_title('Overlap of OOS-Persistent Factors Across Models')
        ax3.grid(axis='x', alpha=0.3)
        
        for i, (cat, cnt) in enumerate(zip(categories, counts)):
            if cnt > 0:
                ax3.text(cnt + 0.1, i, f'{cnt}', va='center')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/oos_persistent_overlap.png', dpi=150)
    plt.close()
    print(f"  ✓ 保存：{output_dir}/oos_persistent_overlap.png")
    
    # 保存详细结果
    if lr_results is not None:
        lr_results.to_csv(f'{output_dir}/oos_lr_significance.csv', index=False)
    if lgbm_results is not None:
        lgbm_results.to_csv(f'{output_dir}/oos_lgbm_significance.csv', index=False)
    if cnn_results is not None:
        cnn_results.to_csv(f'{output_dir}/oos_cnn_significance.csv', index=False)
    print(f"  ✓ 保存：{output_dir}/oos_*_significance.csv")
    
    return persistent_sets


# ================= 主函数 =================
def main():
    print("=" * 70)
    print("🔬 Out-of-Sample Factor Significance Testing")
    print("=" * 70)
    print(f"📊 Split Mode: {OOS_SPLIT_MODE}")
    print(f"📈 Train Ratio: {TRAIN_SYMBOL_RATIO}")
    print(f"🔍 Significance Level (α): {SIGNIFICANCE_LEVEL}")
    print(f"🔄 Bootstrap: {N_BOOTSTRAP}")
    print(f"💻 CPU Cores: {multiprocessing.cpu_count()}")
    print(f"🔥 PyTorch: {TORCH_AVAILABLE}")
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)
    
    device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
    
    # 1. 加载数据
    print("\n📂 Loading data...")
    try:
        df = load_all_symbol_data(DATASET_DIR, DATASET_PATTERN)
        print(f"  ✓ Loaded {len(df):,} samples, {df['symbol'].nunique()} symbols")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # 2. OOS 分割
    print(f"\n📐 Splitting data ({OOS_SPLIT_MODE} mode)...")
    train_df, test_df = split_data_oos(df, mode=OOS_SPLIT_MODE, train_ratio=TRAIN_SYMBOL_RATIO)
    
    # 3. 准备特征
    print("\n🔧 Preparing features...")
    X_train, y_train, feature_cols = prepare_features(train_df)
    X_test, y_test, _ = prepare_features(test_df)
    print(f"  ✓ {len(feature_cols)} factors")
    
    # 特征标准化（fit 在训练集，transform 在测试集）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. LR OOS 显著性检验
    print("\n" + "-" * 70)
    lr_model, lr_results = test_lr_oos_significance(
        pd.DataFrame(X_train_scaled, columns=feature_cols), y_train,
        pd.DataFrame(X_test_scaled, columns=feature_cols), y_test,
        feature_cols, alpha=SIGNIFICANCE_LEVEL
    )
    
    # 5. LGBM OOS 显著性检验
    print("\n" + "-" * 70)
    lgbm_model, lgbm_results = test_lgbm_oos_significance(
        pd.DataFrame(X_train_scaled, columns=feature_cols), y_train,
        pd.DataFrame(X_test_scaled, columns=feature_cols), y_test,
        feature_cols, n_bootstrap=N_BOOTSTRAP, alpha=SIGNIFICANCE_LEVEL
    )
    
    # 6. CNN OOS 显著性检验
    cnn_model = None
    cnn_results = pd.DataFrame()
    if TORCH_AVAILABLE and len(X_train) >= 500:
        print("\n" + "-" * 70)
        SEQ_LEN = CNN_SEQUENCE_LENGTH
        
        # 创建序列数据
        X_train_seq, y_train_seq, _ = create_sequence_data(
            pd.DataFrame(X_train_scaled, columns=feature_cols), y_train, train_df, SEQ_LEN)
        X_test_seq, y_test_seq, _ = create_sequence_data(
            pd.DataFrame(X_test_scaled, columns=feature_cols), y_test, test_df, SEQ_LEN)
        
        actual_num_features = X_train_seq.shape[2]
        
        cnn_model, cnn_results = test_cnn_oos_significance(
            X_train_seq, y_train_seq, X_test_seq, y_test_seq,
            feature_cols, actual_num_features, SEQ_LEN,
            device=device, n_samples=CNN_N_SAMPLES
        )
    else:
        print("\n⚠️  Skipping CNN OOS test (insufficient data or PyTorch unavailable)")
    
    # 7. 可视化
    print("\n📊 Generating visualizations...")
    persistent_sets = plot_oos_significance(
        lr_results, lgbm_results, cnn_results, feature_cols, OOS_SPLIT_MODE
    )
    
    # 8. 结论总结
    print("\n" + "=" * 70)
    print("--- 📋 OOS Significance Summary ---")
    
    summary = []
    if lr_results is not None and not lr_results.empty:
        persistent_lr = lr_results['oos_persistence'].sum()
        summary.append(('LR', persistent_lr, len(feature_cols)))
        print(f"✅ LR: {persistent_lr}/{len(feature_cols)} OOS-persistent factors")
    
    if lgbm_results is not None and not lgbm_results.empty:
        persistent_lgbm = lgbm_results['oos_persistent'].sum()
        summary.append(('LGBM', persistent_lgbm, len(feature_cols)))
        print(f"✅ LGBM: {persistent_lgbm}/{len(feature_cols)} OOS-persistent factors")
    
    if cnn_results is not None and not cnn_results.empty:
        persistent_cnn = cnn_results['oos_persistent'].sum()
        summary.append(('CNN', persistent_cnn, len(feature_cols)))
        print(f"✅ CNN: {persistent_cnn}/{len(feature_cols)} OOS-persistent factors")
    
    # 多模型共同持久因子
    if len(persistent_sets) >= 2:
        common_persistent = set.intersection(*persistent_sets.values())
        if common_persistent:
            print(f"\n🎯 Common OOS-persistent factors (all models): {len(common_persistent)}")
            for f in list(common_persistent)[:10]:
                print(f"   • {f}")
    
    print("\n💡 Recommendations:")
    print("   • 优先使用多模型共同持久的因子构建策略")
    print("   • 对 OOS 不持久的因子：可能是过拟合或市场机制变化")
    print("   • 建议定期重新检验因子显著性（滚动窗口）")
    print("=" * 70)
    
    return lr_results, lgbm_results, cnn_results


if __name__ == "__main__":
    main()