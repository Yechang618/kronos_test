#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer Factor Testing Script
基于训练好的模型进行高频因子预测测试
输入：30 秒因子序列 | 输出：下一秒 label 预测
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import pickle
import json
warnings.filterwarnings('ignore')

import sys 


# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 可视化
import matplotlib.pyplot as plt

# 机器学习
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy import stats

# ============================
# 配置区域
# ============================
class TestConfig:
    # 数据路径
    FACTOR_DIR = Path("./datasets/factors/hf_factors")
    MODEL_DIR = Path("./datasets/model_training")
    OUTPUT_DIR = Path("./datasets/model_testing")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ✅ 必须与训练时一致
    SEQ_LENGTH = 60              # 30 秒 @ 500ms = 60 个时间点
    PREDICTION_HORIZON = 1       # 预测下一秒
    
    # 测试参数
    BATCH_SIZE = 64
    
    # 设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = TestConfig()

# ============================
# 数据集类 (测试用)
# ============================
class TestFactorSequenceDataset(Dataset):
    """测试因子序列数据集"""
    
    def __init__(self, factor_df: pd.DataFrame, seq_length: int,
                 prediction_horizon: int, target_col: str,
                 scaler=None, target_mean=None, target_std=None):
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        self.target_col = target_col
        self.scaler = scaler
        self.target_mean = target_mean
        self.target_std = target_std
        
        # 获取因子列
        exclude_cols = [target_col, 'target', 'timestamp', 'year_month', 'timestampes']
        self.factor_cols = [c for c in factor_df.columns if c not in exclude_cols]
        
        # 数据预处理
        factor_data_raw = factor_df[self.factor_cols].copy()
        factor_data_raw = factor_data_raw.fillna(0)
        
        # 标准化
        if scaler is not None:
            self.factor_data = scaler.transform(factor_data_raw)
        else:
            self.factor_data = factor_data_raw.values
        
        # 目标变量处理
        if target_col not in factor_df.columns:
            self.targets = factor_df['mid_basis'].shift(-prediction_horizon).pct_change(
                prediction_horizon
            ).values
        else:
            self.targets = factor_df[target_col].values
        
        self.targets = np.nan_to_num(self.targets, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 目标变量标准化 (使用训练时的统计量)
        if target_mean is not None and target_std is not None:
            self.targets = (self.targets - target_mean) / (target_std + 1e-10)
            self.targets = np.clip(self.targets, -5, 5)
        
        # 时间戳
        self.timestamps = factor_df.index if hasattr(factor_df, 'index') else None
        
        # 有效样本索引
        self.valid_indices = self._get_valid_indices()
        
        print(f"  📊 测试数据集：{len(self.valid_indices)} 个有效序列")
    
    def _get_valid_indices(self):
        """获取有效序列起始索引"""
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
        
        seq_tensor = torch.FloatTensor(seq)
        target_reg_tensor = torch.FloatTensor([target_reg])[0]
        target_cls_tensor = torch.LongTensor([target_cls])[0]
        
        return seq_tensor, target_reg_tensor, target_cls_tensor, start_idx
    
    def get_factor_names(self):
        return self.factor_cols

# ============================
# Transformer 模型 (与训练一致)
# ============================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class FactorTransformer(nn.Module):
    def __init__(self, n_factors: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.n_factors = n_factors
        self.d_model = d_model
        
        self.input_embedding = nn.Sequential(
            nn.Linear(n_factors, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=False,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.permute(1, 0, 2)
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x)
        pooled = encoded.permute(1, 2, 0)
        pooled = self.pooling(pooled).squeeze(-1)
        
        reg_out = self.regression_head(pooled).squeeze(-1)
        cls_out = self.classification_head(pooled)
        
        reg_out = torch.tanh(reg_out) * 0.1
        
        return reg_out, cls_out

# ============================
# 测试器
# ============================
class ModelTester:
    """模型测试器"""
    
    def __init__(self, model: nn.Module, config: TestConfig):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.model.eval()
    
    def evaluate(self, test_loader, target_mean=None, target_std=None):
        """测试集评估"""
        all_preds = []
        all_targets = []
        all_cls_preds = []
        all_cls_targets = []
        all_indices = []
        
        with torch.no_grad():
            for seq, target_reg, target_cls, indices in test_loader:
                seq = seq.to(self.config.DEVICE)
                pred_reg, pred_cls = self.model(seq)
                
                if torch.isnan(pred_reg).any() or torch.isinf(pred_reg).any():
                    continue
                
                pred_np = pred_reg.detach().cpu().numpy()
                target_np = target_reg.cpu().numpy()
                valid_mask = ~(np.isnan(pred_np) | np.isnan(target_np))
                if valid_mask.sum() > 0:
                    all_preds.extend(pred_np[valid_mask].tolist())
                    all_targets.extend(target_np[valid_mask].tolist())
                
                cls_pred_np = pred_cls.argmax(dim=1).detach().cpu().numpy()
                cls_target_np = target_cls.cpu().numpy()
                all_cls_preds.extend(cls_pred_np.tolist())
                all_cls_targets.extend(cls_target_np.tolist())
                all_indices.extend(indices.cpu().numpy().tolist())
        
        if len(all_preds) < 10:
            print(f"  ⚠️ 有效预测不足")
            return None
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # 恢复原始尺度
        if target_mean is not None and target_std is not None:
            all_preds_original = all_preds * target_std + target_mean
            all_targets_original = all_targets * target_std + target_mean
        else:
            all_preds_original = all_preds
            all_targets_original = all_targets
        
        # 计算 IC
        ic = self._calculate_ic(all_preds, all_targets)
        
        # ICIR
        ic_samples = [self._calculate_ic(
            np.random.permutation(all_preds), all_targets
        ) for _ in range(10)]
        ic_std = np.std(ic_samples)
        ic_ir = ic / ic_std if ic_std > 0 else 0
        
        # 方向准确率
        pred_sign = np.sign(all_preds)
        target_sign = np.sign(all_targets)
        valid_mask = ~(np.isnan(pred_sign) | np.isnan(target_sign))
        direction_acc = accuracy_score(pred_sign[valid_mask], target_sign[valid_mask]) if valid_mask.sum() > 0 else 0.5
        
        # 分类指标
        try:
            cls_metrics = precision_recall_fscore_support(
                all_cls_targets, all_cls_preds, average='binary', zero_division=0
            )
        except:
            cls_metrics = (0.5, 0.5, 0.5, 0)
        
        # 分层回测
        quantile_returns = self._quantile_backtest(all_preds_original, all_targets_original)
        
        return {
            'ic': ic,
            'ic_ir': ic_ir,
            'direction_accuracy': direction_acc,
            'precision': cls_metrics[0],
            'recall': cls_metrics[1],
            'f1': cls_metrics[2],
            'quantile_returns': quantile_returns,
            'predictions': all_preds_original,
            'targets': all_targets_original,
            'indices': all_indices
        }
    
    def _calculate_ic(self, preds, targets):
        preds = np.array(preds)
        targets = np.array(targets)
        valid_mask = ~(np.isnan(preds) | np.isnan(targets))
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        if len(preds) < 10:
            return 0.0
        ic, _ = stats.spearmanr(preds, targets)
        return ic if not np.isnan(ic) else 0.0
    
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
# 可视化器
# ============================
class TestingVisualizer:
    """测试可视化"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_quantile_returns(self, quantile_returns: dict, symbol: str):
        fig, ax = plt.subplots(figsize=(10, 6))
        quantiles = list(quantile_returns.keys())
        returns = list(quantile_returns.values())
        colors = ['red' if r < 0 else 'green' for r in returns]
        ax.bar(range(len(quantiles)), returns, color=colors, edgecolor='black')
        ax.set_xticks(range(len(quantiles)))
        ax.set_xticklabels([f'Q{i+1}' for i in quantiles])
        ax.set_xlabel('Quantile')
        ax.set_ylabel('Average Return')
        ax.set_title('Quantile Backtest Returns')
        ax.grid(True, alpha=0.3, axis='y')
        if len(returns) >= 2:
            long_short = returns[-1] - returns[0]
            ax.axhline(long_short, color='blue', linestyle='--', label=f'Long-Short: {long_short:.6f}')
            ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_quantile_returns.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存分层收益图：{symbol}_quantile_returns.png")
    
    def plot_prediction_scatter(self, predictions: np.ndarray, targets: np.ndarray, symbol: str):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(targets, predictions, alpha=0.3, s=10)
        axes[0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', linewidth=2)
        axes[0].set_xlabel('Actual Return')
        axes[0].set_ylabel('Predicted Return')
        axes[0].set_title('Prediction vs Actual')
        axes[0].grid(True, alpha=0.3)
        
        direction_correct = np.sign(predictions) == np.sign(targets)
        axes[1].hist(targets[direction_correct], bins=50, alpha=0.7, label='Correct', color='green')
        axes[1].hist(targets[~direction_correct], bins=50, alpha=0.7, label='Wrong', color='red')
        axes[1].set_xlabel('Return')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Direction Accuracy: {direction_correct.mean():.4f}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_prediction_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存预测散点图：{symbol}_prediction_scatter.png")

# ============================
# 主测试流程
# ============================
def test_symbol(symbol: str, config: TestConfig) -> dict:
    """测试单个交易对"""
    print(f"\n{'='*60}")
    print(f"🧪 测试交易对：{symbol}")
    print(f"{'='*60}")
    
    # 1. 加载训练配置
    model_dir = config.MODEL_DIR / symbol
    if not model_dir.exists():
        print(f"  ❌ 模型目录不存在：{model_dir}")
        return {'status': 'failed', 'reason': 'no_model'}
    
    config_file = model_dir / 'train_config.json'
    if not config_file.exists():
        print(f"  ❌ 配置文件不存在：{config_file}")
        return {'status': 'failed', 'reason': 'no_config'}
    
    with open(config_file, 'r') as f:
        train_config = json.load(f)
    
    print(f"  📋 加载训练配置：{train_config['symbol']}")
    print(f"     序列长度：{train_config['seq_length']}")
    print(f"     预测期：{train_config['prediction_horizon']}")
    print(f"     因子数：{train_config['n_factors']}")
    
    # 2. 加载模型
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
    print(f"  ✅ 加载模型，最佳验证 IC: {checkpoint['val_ic']:.4f}")
    
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
    test_dataset = TestFactorSequenceDataset(
        full_df,
        seq_length=config.SEQ_LENGTH,
        prediction_horizon=config.PREDICTION_HORIZON,
        target_col=train_config['target_col'],
        scaler=scaler,
        target_mean=target_mean,
        target_std=target_std
    )
    
    if len(test_dataset) < 10:
        print(f"  ❌ 有效测试序列不足：{len(test_dataset)}")
        return {'status': 'failed', 'reason': 'insufficient_data'}
    
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 6. 测试评估
    print(f"\n📊 测试集评估...")
    tester = ModelTester(model, config)
    test_results = tester.evaluate(test_loader, target_mean, target_std)
    
    if test_results is None:
        return {'status': 'failed', 'reason': 'evaluation_failed'}
    
    # 7. 可视化
    print(f"  🎨 生成可视化...")
    symbol_output_dir = config.OUTPUT_DIR / symbol
    symbol_output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = TestingVisualizer(symbol_output_dir)
    visualizer.plot_quantile_returns(test_results['quantile_returns'], symbol)
    visualizer.plot_prediction_scatter(
        test_results['predictions'], test_results['targets'], symbol
    )
    
    # 8. 保存结果
    summary = {
        'symbol': symbol,
        'n_samples': len(test_dataset),
        'n_factors': train_config['n_factors'],
        'test_ic': test_results['ic'],
        'test_ic_ir': test_results['ic_ir'],
        'direction_accuracy': test_results['direction_accuracy'],
        'precision': test_results['precision'],
        'recall': test_results['recall'],
        'f1': test_results['f1'],
        'long_short_return': (
            test_results['quantile_returns'].get(max(test_results['quantile_returns'].keys()), 0) -
            test_results['quantile_returns'].get(min(test_results['quantile_returns'].keys()), 0)
        ),
        'train_best_val_ic': checkpoint['val_ic'],
        'status': 'success'
    }
    
    # 保存预测结果
    pd.DataFrame({
        'prediction': test_results['predictions'],
        'target': test_results['targets']
    }).to_csv(symbol_output_dir / f'{symbol}_predictions.csv', index=False)
    
    print(f"\n📋 测试摘要:")
    print(f"     测试 IC: {summary['test_ic']:.4f}")
    print(f"     ICIR: {summary['test_ic_ir']:.4f}")
    print(f"     方向准确率：{summary['direction_accuracy']:.4f}")
    print(f"     多空收益：{summary['long_short_return']:.6f}")
    print(f"     训练最佳 IC: {summary['train_best_val_ic']:.4f}")
    
    return summary


def discover_symbols(config: TestConfig) -> list:
    """发现所有有模型的交易对"""
    if not config.MODEL_DIR.exists():
        raise FileNotFoundError(f"模型目录不存在：{config.MODEL_DIR}")
    symbols = [d.name for d in config.MODEL_DIR.iterdir() if d.is_dir()]
    print(f"🔍 发现 {len(symbols)} 个交易对有训练模型")
    return symbols


def generate_summary_report(summaries: list, config: TestConfig):
    """生成汇总报告"""
    print(f"\n{'='*60}")
    print("📊 生成测试汇总报告")
    print(f"{'='*60}")
    
    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df[summary_df['status'] == 'success']
    
    if summary_df.empty:
        print("  ❌ 无成功测试的交易对")
        return
    
    summary_df.to_csv(config.OUTPUT_DIR / "all_symbols_testing.csv", index=False)
    
    top_by_ic = summary_df.nlargest(5, 'test_ic')
    top_by_acc = summary_df.nlargest(5, 'direction_accuracy')
    
    print("\n🏆 Top 5 交易对 (按测试 IC):")
    for _, row in top_by_ic.iterrows():
        print(f"   {row['symbol']}: IC={row['test_ic']:.4f}, 准确率={row['direction_accuracy']:.4f}")
    
    print("\n🏆 Top 5 交易对 (按方向准确率):")
    for _, row in top_by_acc.iterrows():
        print(f"   {row['symbol']}: IC={row['test_ic']:.4f}, 准确率={row['direction_accuracy']:.4f}")
    
    print(f"\n💾 汇总报告：{config.OUTPUT_DIR / 'all_symbols_testing.csv'}")


# ============================
# 主程序入口
# ============================
if __name__ == "__main__":
    print("="*60)
    print("🚀 Transformer 高频因子测试脚本")
    print("="*60)
    print(f"📁 因子目录：{config.FACTOR_DIR}")
    print(f"📁 模型目录：{config.MODEL_DIR}")
    print(f"📁 输出目录：{config.OUTPUT_DIR}")
    print(f"📐 序列长度：{config.SEQ_LENGTH} (30 秒 @ 500ms)")
    print(f"🎯 预测期：{config.PREDICTION_HORIZON} (下一秒)")
    print(f"📦 Batch Size: {config.BATCH_SIZE}")
    print(f"🔧 设备：{config.DEVICE}")
    print("="*60)
    
    symbols = discover_symbols(config)
    
    if not symbols:
        print("❌ 未发现任何训练好的模型")
        exit(1)
    
    all_summaries = []
    
    # 测试指定交易对
    # symbol = 'ZECUSDT'
    symbol = 'AVAXUSDT'
    print(f"\n[1/1] 处理进度")
    try:
        summary = test_symbol(symbol, config)
        all_summaries.append(summary)
    except Exception as e:
        print(f"❌ {symbol} 测试失败：{e}")
        import traceback
        traceback.print_exc()
        all_summaries.append({
            'symbol': symbol,
            'status': 'failed',
            'error': str(e)
        })
    
    generate_summary_report(all_summaries, config)
    
    print("\n" + "="*60)
    print("🎉 Transformer 因子测试完成!")
    print("="*60)