import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
import os
import multiprocessing
from pathlib import Path
import glob
import time

# ================= 关键修复：设置环境变量 =================
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ========================================================
warnings.filterwarnings('ignore')

# ================= PyTorch 导入 =================
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    print("✅ PyTorch 可用")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch 不可用，CNN 功能将禁用")
    print("💡 安装：pip install torch")
# ===============================================

# ================= 配置区域 =================
DATASET_DIR = "./dataset"
DATASET_PATTERN = "./dataset/samples_*.csv"
TEST_SYMBOL_RATIO = 0.3
VAL_SYMBOL_RATIO = 0.2
N_JOBS = 1
RANDOM_STATE = 42

# CNN 配置
CNN_SEQUENCE_LENGTH = 10
CNN_HIDDEN_DIM = 64
CNN_NUM_FILTERS = 32
CNN_KERNEL_SIZE = 3
CNN_BATCH_SIZE = 64
CNN_EPOCHS = 50
CNN_LEARNING_RATE = 0.001
CNN_DROPOUT = 0.3
# ===========================================


# ================= CNN 模型定义 =================
class SlippageCNN(nn.Module):
    """
    1D CNN 模型用于滑点预测
    修复：动态计算 FC 层输入维度
    """
    def __init__(self, num_features, sequence_length, hidden_dim=64, 
                 num_filters=32, kernel_size=3, dropout=0.3):
        super(SlippageCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(num_filters)
        
        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters * 2,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(num_filters * 2)
        
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # 修复：不预先定义 fc1，在 forward 中动态创建
        self.fc1 = None
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.hidden_dim = hidden_dim
        self.num_filters = num_filters
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        x = x.flatten(start_dim=1)
        
        # 修复：动态初始化 fc1
        if self.fc1 is None:
            flattened_size = x.shape[1]
            self.fc1 = nn.Linear(flattened_size, self.hidden_dim).to(x.device)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        return x.squeeze()


class SlippageDataset(TensorDataset):
    """PyTorch 数据集"""
    pass
# ===============================================


def load_all_symbol_data(dataset_dir, dataset_pattern):
    """加载目标路径下所有 symbol 的数据"""
    print("📂 扫描数据文件...")
    
    csv_files = glob.glob(dataset_pattern)
    
    if not csv_files:
        default_path = os.path.join(dataset_dir, "processed_training_set.csv")
        if os.path.exists(default_path):
            csv_files = [default_path]
        else:
            raise FileNotFoundError(
                f"No dataset found. Please check:\n"
                f"  - Pattern: {dataset_pattern}\n"
                f"  - Default: {default_path}"
            )
    
    print(f"  找到 {len(csv_files)} 个数据文件")
    
    dfs = []
    symbol_stats = {}
    
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            print(f"  ✓ 加载：{f} ({len(df)} 行)")
            
            if 'symbol' not in df.columns:
                print(f"  ⚠️ 警告：{f} 缺少 'symbol' 列，跳过")
                continue
            
            for sym in df['symbol'].unique():
                sym_count = len(df[df['symbol'] == sym])
                symbol_stats[sym] = symbol_stats.get(sym, 0) + sym_count
            
            dfs.append(df)
        except Exception as e:
            print(f"  ✗ 读取失败 {f}: {e}")
    
    if not dfs:
        raise FileNotFoundError("No valid data files loaded.")
    
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    print(f"\n📊 数据概览:")
    print(f"  总样本数：{len(full_df):,}")
    print(f"  Symbol 数量：{len(symbol_stats)}")
    print(f"  时间范围：{full_df['timestamp'].min()} - {full_df['timestamp'].max()}")
    
    return full_df, symbol_stats


def split_by_symbol(df, test_ratio=0.3, val_ratio=0.2, random_state=42):
    """按 Symbol 进行 off-the-sample 分割"""
    np.random.seed(random_state)
    
    all_symbols = df['symbol'].unique()
    n_symbols = len(all_symbols)
    
    if n_symbols < 3:
        raise ValueError(
            f"至少需要 3 个 symbol 进行 off-the-sample 测试，当前只有 {n_symbols} 个"
        )
    
    shuffled_symbols = np.random.permutation(all_symbols)
    
    n_test = max(1, int(n_symbols * test_ratio))
    n_val = max(1, int(n_symbols * val_ratio))
    n_train = n_symbols - n_test - n_val
    
    if n_train < 1:
        n_train = 1
        n_val = max(1, n_symbols - n_train - n_test)
    
    test_symbols = shuffled_symbols[:n_test]
    val_symbols = shuffled_symbols[n_test:n_test + n_val]
    train_symbols = shuffled_symbols[n_test + n_val:]
    
    print(f"\n📐 Symbol 分割方案:")
    print(f"  训练集 Symbol ({len(train_symbols)}): {list(train_symbols)[:5]}{'...' if len(train_symbols) > 5 else ''}")
    print(f"  验证集 Symbol ({len(val_symbols)}): {list(val_symbols)[:5]}{'...' if len(val_symbols) > 5 else ''}")
    print(f"  测试集 Symbol ({len(test_symbols)}): {list(test_symbols)[:5]}{'...' if len(test_symbols) > 5 else ''}")
    
    train_df = df[df['symbol'].isin(train_symbols)].copy()
    val_df = df[df['symbol'].isin(val_symbols)].copy()
    test_df = df[df['symbol'].isin(test_symbols)].copy()
    
    print(f"\n📊 数据集分割结果:")
    print(f"  训练集：{len(train_df):,} 样本 ({len(train_symbols)} 个 symbol)")
    print(f"  验证集：{len(val_df):,} 样本 ({len(val_symbols)} 个 symbol)")
    print(f"  测试集：{len(test_df):,} 样本 ({len(test_symbols)} 个 symbol)")
    
    return train_df, val_df, test_df, train_symbols, val_symbols, test_symbols


def prepare_features(df):
    """准备特征和标签"""
    feature_cols = [c for c in df.columns if c.startswith('factor_')]
    if not feature_cols:
        raise ValueError("No factor columns found in dataset.")
    
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


def create_sequence_data(X, y, df, sequence_length=10):
    """
    为 CNN 创建序列数据
    修复：兼容 DataFrame 和 numpy.ndarray 两种输入类型
    """
    print(f"  创建序列数据 (length={sequence_length})...")
    
    # 修复：处理不同类型输入
    if isinstance(X, pd.DataFrame):
        X_values = X.values
        print(f"    X 类型：DataFrame -> 转换为 numpy array")
    else:
        X_values = X
        print(f"    X 类型：numpy.ndarray (形状：{X.shape})")
    
    if isinstance(y, pd.Series):
        y_values = y.values
    else:
        y_values = y
    
    # 调试信息
    print(f"  输入形状检查：X={X_values.shape}, y={len(y_values)}, df={len(df)}")
    assert len(X_values) == len(y_values) == len(df), "X, y, df 长度不一致！"
    
    X_seq = []
    y_seq = []
    valid_indices = []
    
    df = df.copy()
    df['original_idx'] = range(len(df))
    
    for symbol in df['symbol'].unique():
        sym_mask = df['symbol'] == symbol
        sym_df = df[sym_mask].sort_values('timestamp').reset_index(drop=True)
        
        sym_X = X_values[sym_mask]
        sym_y = y_values[sym_mask]
        
        for i in range(sequence_length, len(sym_df)):
            seq = sym_X[i-sequence_length:i]
            X_seq.append(seq)
            y_seq.append(sym_y[i])
            valid_indices.append(sym_df.iloc[i]['original_idx'])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    print(f"  序列数据形状：{X_seq.shape}")
    print(f"  有效样本数：{len(y_seq)} (原始：{len(y_values)})")
    
    return X_seq, y_seq, valid_indices


def train_model(X_train, y_train, model_type='lgbm'):
    """训练模型"""
    if model_type == 'lr':
        model = LinearRegression()
        model.fit(X_train, y_train)
    elif model_type == 'lgbm':
        model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=N_JOBS,
            force_col_wise=True,
            feature_pre_filter=False
        )
        model.fit(X_train, y_train)
    return model


def train_cnn_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq, 
                    num_features, sequence_length, epochs=50, batch_size=64, 
                    learning_rate=0.001, device='cpu'):
    """训练 CNN 模型"""
    print(f"  初始化 CNN 模型...")
    print(f"    输入形状：{X_train_seq.shape}")
    print(f"    num_features={num_features}, sequence_length={sequence_length}")
    
    model = SlippageCNN(
        num_features=num_features,
        sequence_length=sequence_length,
        hidden_dim=CNN_HIDDEN_DIM,
        num_filters=CNN_NUM_FILTERS,
        kernel_size=CNN_KERNEL_SIZE,
        dropout=CNN_DROPOUT
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    train_dataset = SlippageDataset(
        torch.FloatTensor(X_train_seq), 
        torch.FloatTensor(y_train_seq)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    val_dataset = SlippageDataset(
        torch.FloatTensor(X_val_seq), 
        torch.FloatTensor(y_val_seq)
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    max_patience = 10
    
    print(f"  开始训练 (epochs={epochs}, device={device})...")
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        epoch_val_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")
        
        if patience_counter >= max_patience:
            print(f"    早停于 epoch {epoch+1}")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"  训练完成，最佳验证损失：{best_val_loss:.6f}")
    
    return model, train_losses, val_losses


def evaluate_model(model, X_test, y_test, df_test, model_name, device='cpu'):
    """评估模型"""
    is_cnn = isinstance(model, SlippageCNN)
    
    if is_cnn:
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(device)
            y_pred = model(X_tensor).cpu().numpy()
    else:
        y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    ic = np.corrcoef(y_pred, y_test)[0, 1] if len(y_test) > 1 else 0
    
    print(f"\n--- {model_name} 整体结果 ---")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²:  {r2:.4f}")
    print(f"  IC:  {ic:.4f}")
    
    symbol_metrics = []
    df_test = df_test.copy()
    df_test['pred'] = y_pred
    
    for symbol in df_test['symbol'].unique():
        sym_mask = df_test['symbol'] == symbol
        sym_y_true = y_test[sym_mask]
        sym_y_pred = y_pred[sym_mask]
        
        if len(sym_y_true) < 2:
            continue
        
        sym_mse = mean_squared_error(sym_y_true, sym_y_pred)
        sym_mae = mean_absolute_error(sym_y_true, sym_y_pred)
        sym_r2 = r2_score(sym_y_true, sym_y_pred)
        sym_ic = np.corrcoef(sym_y_pred, sym_y_true)[0, 1] if len(sym_y_true) > 1 else 0
        
        symbol_metrics.append({
            'symbol': symbol,
            'samples': len(sym_y_true),
            'mse': sym_mse,
            'mae': sym_mae,
            'r2': sym_r2,
            'ic': sym_ic,
            'model': model_name
        })
    
    symbol_metrics_df = pd.DataFrame(symbol_metrics)
    
    print(f"\n--- 按 Symbol 评估 (Off-the-Sample) ---")
    print(symbol_metrics_df[['symbol', 'samples', 'mae', 'ic']].to_string(index=False))
    
    avg_ic = symbol_metrics_df['ic'].mean()
    ic_std = symbol_metrics_df['ic'].std()
    positive_ic_ratio = (symbol_metrics_df['ic'] > 0).mean()
    
    print(f"\n--- 泛化能力分析 ---")
    print(f"  平均 IC: {avg_ic:.4f} ± {ic_std:.4f}")
    print(f"  正 IC 比例：{positive_ic_ratio:.2%}")
    
    return y_pred, mse, mae, r2, ic, symbol_metrics_df

def plot_results(y_test, pred_lr, pred_lgbm, pred_cnn, feature_cols, 
                 lgbm_model, lr_model, df_test, symbol_metrics_df, cnn_losses=None):
    """
    可视化结果 - 每张子图分开保存
    新增：Linear Regression 特征重要性
    """
    os.makedirs('./dataset', exist_ok=True)
    
    # 确保所有预测长度一致
    min_len = len(y_test)
    if pred_lr is not None:
        min_len = min(min_len, len(pred_lr))
    if pred_lgbm is not None:
        min_len = min(min_len, len(pred_lgbm))
    if pred_cnn is not None:
        min_len = min(min_len, len(pred_cnn))
    
    # 截断到相同长度
    y_test = y_test[:min_len]
    if pred_lr is not None:
        pred_lr = pred_lr[:min_len]
    if pred_lgbm is not None:
        pred_lgbm = pred_lgbm[:min_len]
    if pred_cnn is not None:
        pred_cnn = pred_cnn[:min_len]
    
    print(f"  可视化数据尺寸：y_test={len(y_test)}")
    
    saved_files = []
    
    # ================= 图 1: LGBM 预测 vs 实际 =================
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    if pred_lgbm is not None:
        ax1.scatter(y_test, pred_lgbm, alpha=0.3, s=10)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ic_lgbm = np.corrcoef(y_test, pred_lgbm)[0, 1] if len(y_test) > 1 else 0
        ax1.set_title(f'Prediction vs Actual (LGBM)\nIC = {ic_lgbm:.4f}')
    else:
        ax1.text(0.5, 0.5, 'No LGBM Prediction', transform=ax1.transAxes, ha='center')
        ax1.set_title('Prediction vs Actual (LGBM)')
    ax1.set_xlabel('Actual Slippage')
    ax1.set_ylabel('Predicted Slippage (LGBM)')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    path1 = './dataset/01_lgbm_prediction_vs_actual.png'
    plt.savefig(path1, dpi=150)
    plt.close()
    saved_files.append(path1)
    print(f"  ✓ 保存：{path1}")
    
    # ================= 图 2: LR 预测 vs 实际 =================
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    if pred_lr is not None:
        ax2.scatter(y_test, pred_lr, alpha=0.3, s=10, color='orange')
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ic_lr = np.corrcoef(y_test, pred_lr)[0, 1] if len(y_test) > 1 else 0
        ax2.set_title(f'Prediction vs Actual (Linear Regression)\nIC = {ic_lr:.4f}')
    else:
        ax2.text(0.5, 0.5, 'No LR Prediction', transform=ax2.transAxes, ha='center')
        ax2.set_title('Prediction vs Actual (Linear Regression)')
    ax2.set_xlabel('Actual Slippage')
    ax2.set_ylabel('Predicted Slippage (LR)')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = './dataset/02_lr_prediction_vs_actual.png'
    plt.savefig(path2, dpi=150)
    plt.close()
    saved_files.append(path2)
    print(f"  ✓ 保存：{path2}")
    
    # ================= 图 3: CNN 预测 vs 实际 =================
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    if pred_cnn is not None:
        ax3.scatter(y_test, pred_cnn, alpha=0.3, s=10, color='green')
        ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ic_cnn = np.corrcoef(y_test, pred_cnn)[0, 1] if len(y_test) > 1 else 0
        ax3.set_title(f'Prediction vs Actual (CNN)\nIC = {ic_cnn:.4f}')
    else:
        ax3.text(0.5, 0.5, 'No CNN Prediction', transform=ax3.transAxes, ha='center')
        ax3.set_title('Prediction vs Actual (CNN)')
    ax3.set_xlabel('Actual Slippage')
    ax3.set_ylabel('Predicted Slippage (CNN)')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    path3 = './dataset/03_cnn_prediction_vs_actual.png'
    plt.savefig(path3, dpi=150)
    plt.close()
    saved_files.append(path3)
    print(f"  ✓ 保存：{path3}")
    
    # ================= 图 4: LGBM 特征重要性 =================
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    if hasattr(lgbm_model, 'feature_importances_') and lgbm_model is not None:
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': lgbm_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        sns.barplot(data=importance_df.head(15), x='Importance', y='Feature', ax=ax4)
        ax4.set_title('Top 15 Feature Importance (LGBM)')
        ax4.set_xlabel('Importance Score')
        ax4.set_ylabel('Feature')
    else:
        ax4.text(0.5, 0.5, 'No Importance Data', transform=ax4.transAxes, ha='center')
        ax4.set_title('Feature Importance (LGBM)')
    ax4.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    path4 = './dataset/04_lgbm_feature_importance.png'
    plt.savefig(path4, dpi=150)
    plt.close()
    saved_files.append(path4)
    print(f"  ✓ 保存：{path4}")
    
    # ================= 🔹 新增 图 5: LR 特征重要性 =================
    fig5, ax5 = plt.subplots(figsize=(10, 8))
    if lr_model is not None and hasattr(lr_model, 'coef_'):
        # 使用系数绝对值作为重要性
        coef_importance = np.abs(lr_model.coef_)
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Coefficient': lr_model.coef_,
            'Importance': coef_importance
        }).sort_values('Importance', ascending=False)
        
        # 创建带颜色编码的条形图（正负系数不同颜色）
        top10 = importance_df.head(15)
        colors = ['green' if c > 0 else 'red' for c in top10['Coefficient']]
        
        bars = ax5.barh(range(len(top10)), top10['Importance'], color=colors, alpha=0.7)
        ax5.set_yticks(range(len(top10)))
        ax5.set_yticklabels(top10['Feature'])
        ax5.set_xlabel('Absolute Coefficient Value')
        ax5.set_title('Top 15 Feature Importance (Linear Regression)\n(Green=Positive, Red=Negative)')
        ax5.grid(True, alpha=0.3, axis='x')
        
        # 在条形图旁标注系数值
        for i, (coef, imp) in enumerate(zip(top10['Coefficient'], top10['Importance'])):
            ax5.text(imp + 0.01, i, f'{coef:+.4f}', va='center', fontsize=9)
    else:
        ax5.text(0.5, 0.5, 'No LR Coefficient Data', transform=ax5.transAxes, ha='center')
        ax5.set_title('Feature Importance (Linear Regression)')
    plt.tight_layout()
    path5 = './dataset/05_lr_feature_importance.png'
    plt.savefig(path5, dpi=150)
    plt.close()
    saved_files.append(path5)
    print(f"  ✓ 保存：{path5}")
    
    # ================= 图 6: CNN 训练损失曲线 =================
    fig6, ax6 = plt.subplots(figsize=(10, 8))
    if cnn_losses is not None:
        train_losses, val_losses = cnn_losses
        ax6.plot(train_losses, label='Train Loss', alpha=0.7, linewidth=2)
        ax6.plot(val_losses, label='Val Loss', alpha=0.7, linewidth=2)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('MSE Loss')
        ax6.set_title('CNN Training History')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No CNN Training Data', transform=ax6.transAxes, ha='center')
        ax6.set_title('CNN Training History')
    plt.tight_layout()
    path6 = './dataset/06_cnn_training_history.png'
    plt.savefig(path6, dpi=150)
    plt.close()
    saved_files.append(path6)
    print(f"  ✓ 保存：{path6}")
    
    # ================= 图 7: LGBM 残差分布 =================
    fig7, ax7 = plt.subplots(figsize=(10, 8))
    if pred_lgbm is not None:
        residuals = y_test - pred_lgbm
        ax7.hist(residuals, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax7.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Residual')
        mean_residual = residuals.mean()
        std_residual = residuals.std()
        ax7.set_title(f'Residual Distribution (LGBM)\nMean = {mean_residual:.6f}, Std = {std_residual:.6f}')
        ax7.legend()
    else:
        ax7.text(0.5, 0.5, 'No LGBM Prediction', transform=ax7.transAxes, ha='center')
        ax7.set_title('Residual Distribution (LGBM)')
    ax7.set_xlabel('Residuals')
    ax7.set_ylabel('Frequency')
    ax7.grid(True, alpha=0.3)
    plt.tight_layout()
    path7 = './dataset/07_lgbm_residual_distribution.png'
    plt.savefig(path7, dpi=150)
    plt.close()
    saved_files.append(path7)
    print(f"  ✓ 保存：{path7}")
    
    # ================= 图 8: LR 残差分布 =================
    fig8, ax8 = plt.subplots(figsize=(10, 8))
    if pred_lr is not None:
        residuals = y_test - pred_lr
        ax8.hist(residuals, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax8.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Residual')
        mean_residual = residuals.mean()
        std_residual = residuals.std()
        ax8.set_title(f'Residual Distribution (Linear Regression)\nMean = {mean_residual:.6f}, Std = {std_residual:.6f}')
        ax8.legend()
    else:
        ax8.text(0.5, 0.5, 'No LR Prediction', transform=ax8.transAxes, ha='center')
        ax8.set_title('Residual Distribution (Linear Regression)')
    ax8.set_xlabel('Residuals')
    ax8.set_ylabel('Frequency')
    ax8.grid(True, alpha=0.3)
    plt.tight_layout()
    path8 = './dataset/08_lr_residual_distribution.png'
    plt.savefig(path8, dpi=150)
    plt.close()
    saved_files.append(path8)
    print(f"  ✓ 保存：{path8}")
    
    # ================= 图 9: 模型对比 (IC by Symbol) =================
    fig9, ax9 = plt.subplots(figsize=(10, 8))
    if not symbol_metrics_df.empty:
        models = symbol_metrics_df['model'].unique()
        x = np.arange(len(models))
        mean_ics = [symbol_metrics_df[symbol_metrics_df['model']==m]['ic'].mean() 
                   for m in models]
        std_ics = [symbol_metrics_df[symbol_metrics_df['model']==m]['ic'].std() 
                  for m in models]
        
        bars = ax9.bar(x, mean_ics, yerr=std_ics, capsize=5, alpha=0.7)
        ax9.set_xticks(x)
        ax9.set_xticklabels(models, rotation=45, ha='right')
        ax9.set_ylabel('Mean IC')
        ax9.set_title('Model Comparison (IC by Symbol - Off-the-Sample)')
        ax9.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax9.grid(True, alpha=0.3, axis='y')
        
        # 在柱子上标注数值
        for i, (mean_ic, std_ic) in enumerate(zip(mean_ics, std_ics)):
            ax9.text(i, mean_ic + std_ic + 0.01, f'{mean_ic:.3f}±{std_ic:.3f}', 
                    ha='center', va='bottom', fontsize=9)
    else:
        ax9.text(0.5, 0.5, 'No Symbol Metrics', transform=ax9.transAxes, ha='center')
        ax9.set_title('Model Comparison (IC by Symbol)')
    plt.tight_layout()
    path9 = './dataset/09_model_comparison_ic.png'
    plt.savefig(path9, dpi=150)
    plt.close()
    saved_files.append(path9)
    print(f"  ✓ 保存：{path9}")
    
    # ================= 图 10: 模型对比 (MAE by Symbol) =================
    fig10, ax10 = plt.subplots(figsize=(10, 8))
    if not symbol_metrics_df.empty:
        models = symbol_metrics_df['model'].unique()
        x = np.arange(len(models))
        mean_maes = [symbol_metrics_df[symbol_metrics_df['model']==m]['mae'].mean() 
                    for m in models]
        std_maes = [symbol_metrics_df[symbol_metrics_df['model']==m]['mae'].std() 
                   for m in models]
        
        bars = ax10.bar(x, mean_maes, yerr=std_maes, capsize=5, alpha=0.7, color='orange')
        ax10.set_xticks(x)
        ax10.set_xticklabels(models, rotation=45, ha='right')
        ax10.set_ylabel('Mean MAE')
        ax10.set_title('Model Comparison (MAE by Symbol - Off-the-Sample)')
        ax10.grid(True, alpha=0.3, axis='y')
        
        # 在柱子上标注数值
        for i, (mean_mae, std_mae) in enumerate(zip(mean_maes, std_maes)):
            ax10.text(i, mean_mae + std_mae + 0.001, f'{mean_mae:.5f}±{std_mae:.5f}', 
                    ha='center', va='bottom', fontsize=9)
    else:
        ax10.text(0.5, 0.5, 'No Symbol Metrics', transform=ax10.transAxes, ha='center')
        ax10.set_title('Model Comparison (MAE by Symbol)')
    plt.tight_layout()
    path10 = './dataset/10_model_comparison_mae.png'
    plt.savefig(path10, dpi=150)
    plt.close()
    saved_files.append(path10)
    print(f"  ✓ 保存：{path10}")
    
    # ================= 图 11: 按 Symbol 的 IC 分布 (LGBM) =================
    fig11, ax11 = plt.subplots(figsize=(12, 6))
    lgbm_metrics = symbol_metrics_df[symbol_metrics_df['model'] == 'LightGBM']
    if not lgbm_metrics.empty:
        ax11.bar(range(len(lgbm_metrics)), lgbm_metrics['ic'], alpha=0.7, color='steelblue')
        ax11.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax11.axhline(y=lgbm_metrics['ic'].mean(), color='g', linestyle='-', linewidth=2, 
                   label=f"Mean IC = {lgbm_metrics['ic'].mean():.4f}")
        ax11.set_xlabel('Symbol Index')
        ax11.set_ylabel('IC')
        ax11.set_title('IC by Symbol (LightGBM - Off-the-Sample)')
        ax11.legend()
        ax11.grid(True, alpha=0.3, axis='y')
    else:
        ax11.text(0.5, 0.5, 'No LGBM Symbol Metrics', transform=ax11.transAxes, ha='center')
        ax11.set_title('IC by Symbol (LightGBM)')
    plt.tight_layout()
    path11 = './dataset/11_lgbm_ic_by_symbol.png'
    plt.savefig(path11, dpi=150)
    plt.close()
    saved_files.append(path11)
    print(f"  ✓ 保存：{path11}")
    
    # ================= 图 12: 按 Symbol 的 MAE 分布 (LGBM) =================
    fig12, ax12 = plt.subplots(figsize=(12, 6))
    if not lgbm_metrics.empty:
        ax12.bar(range(len(lgbm_metrics)), lgbm_metrics['mae'], alpha=0.7, color='coral')
        ax12.axhline(y=lgbm_metrics['mae'].mean(), color='g', linestyle='-', linewidth=2, 
                    label=f"Mean MAE = {lgbm_metrics['mae'].mean():.6f}")
        ax12.set_xlabel('Symbol Index')
        ax12.set_ylabel('MAE')
        ax12.set_title('MAE by Symbol (LightGBM - Off-the-Sample)')
        ax12.legend()
        ax12.grid(True, alpha=0.3, axis='y')
    else:
        ax12.text(0.5, 0.5, 'No LGBM Symbol Metrics', transform=ax12.transAxes, ha='center')
        ax12.set_title('MAE by Symbol (LightGBM)')
    plt.tight_layout()
    path12 = './dataset/12_lgbm_mae_by_symbol.png'
    plt.savefig(path12, dpi=150)
    plt.close()
    saved_files.append(path12)
    print(f"  ✓ 保存：{path12}")
    
    print(f"\n📊 共保存 {len(saved_files)} 张可视化图片至 ./dataset/")
    
    return saved_files

def main():
    print("=" * 70)
    print("🚀 LightGBM + CNN Training Script (Multi-Symbol Off-the-Sample)")
    print("=" * 70)
    print(f"💻 CPU Cores: {multiprocessing.cpu_count()}")
    print(f"⚙️  LightGBM n_jobs: {N_JOBS}")
    print(f"🔥 PyTorch Available: {TORCH_AVAILABLE}")
    if TORCH_AVAILABLE:
        print(f"  - CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    print(f"📁 数据目录：{DATASET_DIR}")
    print(f"🔍 文件模式：{DATASET_PATTERN}")
    print("=" * 70)
    
    device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
    print(f"🖥️  计算设备：{device}")
    print("=" * 70)
    
    # 1. 加载所有 symbol 数据
    print("\n📂 加载数据...")
    try:
        df, symbol_stats = load_all_symbol_data(DATASET_DIR, DATASET_PATTERN)
    except FileNotFoundError as e:
        print(f"❌ 错误：{e}")
        return
    except Exception as e:
        print(f"❌ 加载失败：{e}")
        return
    
    if len(df) == 0:
        print("❌ 训练集为空")
        return
    
    # 2. 按 symbol 分割数据（off-the-sample）
    print("\n📐 分割数据 (Off-the-Sample)...")
    try:
        train_df, val_df, test_df, train_syms, val_syms, test_syms = split_by_symbol(
            df, 
            test_ratio=TEST_SYMBOL_RATIO, 
            val_ratio=VAL_SYMBOL_RATIO,
            random_state=RANDOM_STATE
        )
    except ValueError as e:
        print(f"❌ 分割失败：{e}")
        print("💡 提示：需要更多 symbol 才能进行 off-the-sample 测试")
        return
    
    # 3. 准备特征
    print("\n🔧 准备特征...")
    X_train, y_train, feature_cols = prepare_features(train_df)
    X_val, y_val, _ = prepare_features(val_df)
    X_test, y_test, _ = prepare_features(test_df)
    
    print(f"  特征数量：{len(feature_cols)}")
    print(f"  特征列表：{feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")
    
    # 特征标准化（对 CNN 重要）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. 训练 Linear Regression
    # print("\n📈 训练 Linear Regression...")
    # start_time = time.time()
    # try:
    #     model_lr = train_model(X_train_scaled, y_train, model_type='lr')
    #     lr_time = time.time() - start_time
    #     pred_lr, _, _, _, _, metrics_lr = evaluate_model(
    #         model_lr, X_test_scaled, y_test, test_df, "Linear Regression"
    #     )
    #     print(f"  训练时间：{lr_time:.2f}秒")
    # except Exception as e:
    #     print(f"⚠️  LR 训练失败：{e}")
    #     pred_lr = None
    #     metrics_lr = pd.DataFrame()
    #     lr_time = 0

    print("\n📈 训练 Linear Regression...")
    start_time = time.time()
    try:
        model_lr = train_model(X_train_scaled, y_train, model_type='lr')
        lr_time = time.time() - start_time
        pred_lr, _, _, _, _, metrics_lr = evaluate_model(
            model_lr, X_test_scaled, y_test, test_df, "Linear Regression"
        )
        print(f"  训练时间：{lr_time:.2f}秒")
    except Exception as e:
        print(f"⚠️  LR 训练失败：{e}")
        model_lr = None  # 🔹 保存模型引用
        pred_lr = None
        metrics_lr = pd.DataFrame()
        lr_time = 0
    
    # 5. 训练 LightGBM
    print("\n📈 训练 LightGBM...")
    start_time = time.time()
    try:
        model_lgbm = train_model(X_train_scaled, y_train, model_type='lgbm')
        lgbm_time = time.time() - start_time
        pred_lgbm, mse, mae, r2, ic, metrics_lgbm = evaluate_model(
            model_lgbm, X_test_scaled, y_test, test_df, "LightGBM"
        )
        print(f"  训练时间：{lgbm_time:.2f}秒")
    except Exception as e:
        print(f"⚠️  LGBM 训练失败：{e}")
        print("💡 提示：pip install --upgrade lightgbm")
        pred_lgbm = None
        metrics_lgbm = pd.DataFrame()
        ic = 0
        lgbm_time = 0
    
    # 6. 训练 CNN
    pred_cnn = None
    metrics_cnn = pd.DataFrame()
    cnn_losses = None
    cnn_time = 0
    test_indices = None
    y_test_filtered = y_test
    test_df_filtered = test_df
    pred_lr_filtered = pred_lr
    pred_lgbm_filtered = pred_lgbm
    
    if TORCH_AVAILABLE:
        print("\n📈 训练 CNN...")
        start_time = time.time()
        try:
            # 创建序列数据
            X_train_seq, y_train_seq, train_indices = create_sequence_data(
                X_train_scaled, y_train, train_df, 
                sequence_length=CNN_SEQUENCE_LENGTH
            )
            X_val_seq, y_val_seq, val_indices = create_sequence_data(
                X_val_scaled, y_val, val_df,
                sequence_length=CNN_SEQUENCE_LENGTH
            )
            X_test_seq, y_test_seq, test_indices = create_sequence_data(
                X_test_scaled, y_test, test_df,
                sequence_length=CNN_SEQUENCE_LENGTH
            )
            
            # 过滤测试集
            test_df_filtered = test_df.iloc[test_indices].copy()
            y_test_filtered = y_test.iloc[test_indices].values
            
            # 重新评估 LR 和 LGBM 在 CNN 测试集上
            pred_lr_filtered = pred_lr[test_indices] if pred_lr is not None else None
            pred_lgbm_filtered = pred_lgbm[test_indices] if pred_lgbm is not None else None
            
            # 训练 CNN
            model_cnn, train_losses, val_losses = train_cnn_model(
                X_train_seq, y_train_seq,
                X_val_seq, y_val_seq,
                num_features=len(feature_cols),
                sequence_length=CNN_SEQUENCE_LENGTH,
                epochs=CNN_EPOCHS,
                batch_size=CNN_BATCH_SIZE,
                learning_rate=CNN_LEARNING_RATE,
                device=device
            )
            cnn_losses = (train_losses, val_losses)
            
            # 评估 CNN
            pred_cnn, mse, mae, r2, ic, metrics_cnn = evaluate_model(
                model_cnn, X_test_seq, y_test_filtered, 
                test_df_filtered, "CNN", device=device
            )
            cnn_time = time.time() - start_time
            print(f"  训练时间：{cnn_time:.2f}秒")
            
        except Exception as e:
            print(f"⚠️  CNN 训练失败：{e}")
            import traceback
            traceback.print_exc()
            cnn_time = 0
    else:
        print("\n⚠️  跳过 CNN 训练 (PyTorch 不可用)")
    
    # 7. 合并所有指标
    all_metrics = pd.concat([metrics_lr, metrics_lgbm, metrics_cnn], ignore_index=True)
    
    # 8. 可视化 - 使用过滤后的数据确保尺寸一致
    print("\n📊 生成可视化...")
    plot_results(
        y_test_filtered,
        pred_lr_filtered,
        pred_lgbm_filtered,
        pred_cnn,
        feature_cols,
        model_lgbm if pred_lgbm is not None else None,
        model_lr,  # 🔹 新增：传入 LR 模型
        test_df_filtered,
        all_metrics,
        cnn_losses
    )
    
    # 9. 模型对比总结
    print("\n" + "=" * 70)
    print("--- 模型对比总结 ---")
    models_summary = []
    
    if pred_lr_filtered is not None:
        models_summary.append({
            'Model': 'Linear Regression',
            'IC': metrics_lr['ic'].mean() if not metrics_lr.empty else 0,
            'MAE': metrics_lr['mae'].mean() if not metrics_lr.empty else 0,
            'Time(s)': lr_time
        })
    
    if pred_lgbm_filtered is not None:
        models_summary.append({
            'Model': 'LightGBM',
            'IC': metrics_lgbm['ic'].mean() if not metrics_lgbm.empty else 0,
            'MAE': metrics_lgbm['mae'].mean() if not metrics_lgbm.empty else 0,
            'Time(s)': lgbm_time
        })
    
    if pred_cnn is not None:
        models_summary.append({
            'Model': 'CNN',
            'IC': metrics_cnn['ic'].mean() if not metrics_cnn.empty else 0,
            'MAE': metrics_cnn['mae'].mean() if not metrics_cnn.empty else 0,
            'Time(s)': cnn_time
        })
    
    summary_df = pd.DataFrame(models_summary)
    print(summary_df.to_string(index=False))
    
    # 10. 结论
    print("\n--- 结论 ---")
    if not summary_df.empty:
        best_model = summary_df.loc[summary_df['IC'].idxmax()]
        print(f"✅ 最佳模型：{best_model['Model']} (IC={best_model['IC']:.4f})")
        
        if best_model['IC'] > 0.05:
            print("✅ 因子在未见过的 Symbol 上表现出良好的泛化能力 (IC > 0.05)")
            print("💡 模型可以跨交易对使用")
        elif best_model['IC'] > 0:
            print("⚠️ 因子有一定预测能力，但泛化能力有限 (0 < IC ≤ 0.05)")
            print("💡 建议增加更多 symbol 数据或改进特征工程")
        else:
            print("❌ 因子在未见过的 Symbol 上预测能力较弱 (IC ≤ 0)")
            print("💡 可能存在过拟合，建议简化模型或增加正则化")
    print("=" * 70)
    
    # 11. 保存结果
    if not all_metrics.empty:
        metrics_path = './dataset/off_the_sample_metrics.csv'
        all_metrics.to_csv(metrics_path, index=False)
        print(f"📋 Symbol 级别指标保存至：{metrics_path}")
    
    summary_path = './dataset/model_comparison_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"📋 模型对比总结保存至：{summary_path}")
    
    return model_lgbm if pred_lgbm is not None else None, all_metrics


if __name__ == "__main__":
    main()