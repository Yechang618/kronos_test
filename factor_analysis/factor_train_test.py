import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os
import multiprocessing

# ================= 关键修复：设置环境变量 =================
# 在导入 lightgbm 之前设置，禁用 joblib 的 wmic 调用
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
# ========================================================

warnings.filterwarnings('ignore')

# ================= 配置区域 =================
symbol = "AAVEUSDT"
# DATASET_PATH = "./dataset/processed_training_set.csv"
DATASET_PATH = f"./dataset/samples_20260101_{symbol}.csv"
TEST_SIZE_RATIO = 0.2
VAL_SIZE_RATIO = 0.2
# 修复：设置 n_jobs=1 避免 Windows 下的并行问题
N_JOBS = 1  
# ===========================================

def load_and_split_data(path):
    """加载数据并按时间分割"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please run factor_processor.py first.")
        
    df = pd.read_csv(path)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    n = len(df)
    test_start = int(n * (1 - TEST_SIZE_RATIO))
    val_start = int(n * (1 - TEST_SIZE_RATIO - VAL_SIZE_RATIO))
    
    train_df = df.iloc[:val_start].copy()
    val_df = df.iloc[val_start:test_start].copy()
    test_df = df.iloc[test_start:].copy()
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

def prepare_features(df):
    """准备特征和标签"""
    feature_cols = [c for c in df.columns if c.startswith('factor_')]
    if not feature_cols:
        raise ValueError("No factor columns found in dataset. Check factor_processor.py output.")
        
    X = df[feature_cols].fillna(0)
    y = df['label_slippage']
    return X, y, feature_cols

def train_model(X_train, y_train, model_type='lgbm'):
    """训练模型"""
    if model_type == 'lr':
        model = LinearRegression()
        model.fit(X_train, y_train)
    elif model_type == 'lgbm':
        # 修复：明确设置 n_jobs=1 和 force_col_wise=True
        model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
            n_jobs=N_JOBS,              # 关键修复：设置为 1
            force_col_wise=True,        # 关键修复：使用列-wise 分割
            feature_pre_filter=False    # 关键修复：禁用预过滤
        )
        model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name):
    """评估模型"""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # 计算 IC (预测值与真实值的相关性)
    ic = np.corrcoef(y_pred, y_test)[0, 1]
    
    print(f"\n--- {model_name} Results ---")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R2: {r2:.4f}")
    print(f"IC: {ic:.4f}")
    return y_pred, mse, mae, r2, ic

def plot_results(y_test, pred_lr, pred_lgbm, feature_cols, lgbm_model):
    """可视化结果"""
    os.makedirs('./dataset', exist_ok=True)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 预测 vs 实际 (LGBM)
    axs[0, 0].scatter(y_test, pred_lgbm, alpha=0.3, s=10)
    axs[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axs[0, 0].set_xlabel('Actual Slippage')
    axs[0, 0].set_ylabel('Predicted Slippage (LGBM)')
    axs[0, 0].set_title('Prediction vs Actual (LGBM)')
    axs[0, 0].grid(True)
    
    # 2. 预测 vs 实际 (LR)
    axs[0, 1].scatter(y_test, pred_lr, alpha=0.3, s=10, color='orange')
    axs[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axs[0, 1].set_xlabel('Actual Slippage')
    axs[0, 1].set_ylabel('Predicted Slippage (LR)')
    axs[0, 1].set_title('Prediction vs Actual (Linear Regression)')
    axs[0, 1].grid(True)
    
    # 3. 特征重要性 (LGBM)
    if hasattr(lgbm_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': lgbm_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        sns.barplot(data=importance_df.head(10), x='Importance', y='Feature', ax=axs[1, 0])
        axs[1, 0].set_title('Top 10 Feature Importance (LGBM)')
    else:
        axs[1, 0].text(0.5, 0.5, 'No Importance Data', transform=axs[1, 0].transAxes)
    
    # 4. 残差分布
    residuals = y_test - pred_lgbm
    axs[1, 1].hist(residuals, bins=50, alpha=0.7, color='green')
    axs[1, 1].set_xlabel('Residuals')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].set_title('Residual Distribution (LGBM)')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    output_path = './dataset/model_evaluation_results.png'
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    plt.show()

def main():
    print("=" * 70)
    print("🚀 LightGBM Training Script (Windows Compatible)")
    print("=" * 70)
    print(f"💻 CPU Cores Detected: {multiprocessing.cpu_count()}")
    print(f"⚙️  LightGBM n_jobs: {N_JOBS} (Set to 1 for Windows stability)")
    print("=" * 70)
    
    print("Loading dataset...")
    try:
        train_df, val_df, test_df = load_and_split_data(DATASET_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error loading  {e}")
        return
        
    if len(train_df) == 0:
        print("Error: Training set is empty.")
        return

    X_train, y_train, feature_cols = prepare_features(train_df)
    X_val, y_val, _ = prepare_features(val_df)
    X_test, y_test, _ = prepare_features(test_df)
    
    print(f"Features used ({len(feature_cols)}): {feature_cols}")
    
    # 训练 Linear Regression
    print("\nTraining Linear Regression...")
    try:
        model_lr = train_model(X_train, y_train, model_type='lr')
        pred_lr, _, _, _, _ = evaluate_model(model_lr, X_test, y_test, "Linear Regression")
    except Exception as e:
        print(f"LR Training failed: {e}")
        pred_lr = None
    
    # 训练 LightGBM
    print("\nTraining LightGBM...")
    try:
        model_lgbm = train_model(X_train, y_train, model_type='lgbm')
        pred_lgbm, mse, mae, r2, ic = evaluate_model(model_lgbm, X_test, y_test, "LightGBM")
    except Exception as e:
        print(f"LGBM Training failed: {e}")
        print("⚠️  Try updating lightgbm: pip install --upgrade lightgbm")
        pred_lgbm = None
        ic = 0
    
    # 可视化
    if pred_lr is not None and pred_lgbm is not None:
        plot_results(y_test, pred_lr, pred_lgbm, feature_cols, model_lgbm)
    
    # 简单结论
    print("\n--- Conclusion ---")
    if ic > 0.05:
        print("✅ Factors show promising predictive power (IC > 0.05).")
    else:
        print("⚠️  Factors show weak predictive power. Consider feature engineering.")
    print("=" * 70)

if __name__ == "__main__":
    main()