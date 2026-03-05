#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer-based Factor Predictive Power Evaluation
基于 Transformer 检验高频因子的预测能力 (已修复版)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 机器学习
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy import stats

# ============================
# 配置区域
# ============================
class Config:
    # 数据路径
    FACTOR_DIR = Path("./datasets/factors/hf_factors")
    OUTPUT_DIR = Path("./datasets/transformer_evaluation")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 模型参数
    SEQ_LENGTH = 20
    PREDICTION_HORIZON = 5
    HIDDEN_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 3
    DROPOUT = 0.2
    
    # 训练参数
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    
    # 数据分割
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # 设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# ============================
# 数据集类 (已修复时间戳问题)
# ============================
# ============================
# 数据集类 (已修复数据质量问题)
# ============================
class FactorSequenceDataset(Dataset):
    """因子序列数据集"""
    
    def __init__(self, factor_df: pd.DataFrame, seq_length: int, 
                 prediction_horizon: int, target_col: str = 'target'):
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        self.target_col = target_col
        
        # 获取因子列
        exclude_cols = [target_col, 'target', 'timestamp', 'year_month', 'timestampes']
        self.factor_cols = [c for c in factor_df.columns if c not in exclude_cols]
        
        # ✅ 修复：数据标准化前先处理极端值
        factor_data_raw = factor_df[self.factor_cols].copy()
        
        # Winsorization 处理极端值 (1%-99%)
        for col in self.factor_cols:
            if col in factor_data_raw.columns:
                lower = factor_data_raw[col].quantile(0.01)
                upper = factor_data_raw[col].quantile(0.99)
                factor_data_raw[col] = factor_data_raw[col].clip(lower, upper)
        
        # 填充 NaN
        factor_data_raw = factor_data_raw.fillna(0)
        
        # 标准化
        self.scaler = RobustScaler()
        self.factor_data = self.scaler.fit_transform(factor_data_raw)
        
        # # ✅ 修复：目标变量处理
        # if target_col not in factor_df.columns:
        #     self.targets = factor_df['mid_basis'].shift(-prediction_horizon).pct_change(
        #         prediction_horizon
        #     ).values
        # else:
        #     self.targets = factor_df[target_col].values
        
        # # 填充目标变量的 NaN
        # self.targets = np.nan_to_num(self.targets, nan=0.0, posinf=0.0, neginf=0.0)

        # ✅ 修复：目标变量标准化
        if target_col not in factor_df.columns:
            self.targets = factor_df['mid_basis'].shift(-prediction_horizon).pct_change(
                prediction_horizon
            ).values
        else:
            self.targets = factor_df[target_col].values
        
        # ✅ 修复：处理 Inf 和 NaN
        self.targets = np.nan_to_num(self.targets, nan=0.0, posinf=0.0, neginf=0.0)
        
        # ✅ 修复：目标变量标准化 (关键！)
        self.target_mean = np.mean(self.targets)
        self.target_std = np.std(self.targets) + 1e-10
        self.targets = (self.targets - self.target_mean) / self.target_std
        
        # ✅ 修复：限制极端值
        self.targets = np.clip(self.targets, -5, 5)        
        # 时间戳
        self.timestamps = factor_df.index if hasattr(factor_df, 'index') else None
        
        # 有效样本索引
        self.valid_indices = self._get_valid_indices()
        
        print(f"  📊 数据集：{len(self.valid_indices)} 个有效序列，{len(self.factor_cols)} 个因子")
    
    def _get_valid_indices(self):
        """获取有效序列起始索引"""
        valid = []
        for i in range(len(self.factor_data) - self.seq_length - self.prediction_horizon):
            seq = self.factor_data[i:i+self.seq_length]
            target_idx = i + self.seq_length + self.prediction_horizon - 1
            
            # 检查序列内 NaN 比例
            if np.isnan(seq).sum() / seq.size > 0.3:
                continue
            
            # 检查目标值是否有效
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
        
        # ✅ 确保无 NaN
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        
        seq_tensor = torch.FloatTensor(seq)
        target_reg_tensor = torch.FloatTensor([target_reg])[0]
        target_cls_tensor = torch.LongTensor([target_cls])[0]
        
        return seq_tensor, target_reg_tensor, target_cls_tensor
    
    def get_factor_names(self):
        """获取因子名称列表"""
        return self.factor_cols


# ============================
# Transformer 模型
# ============================
class PositionalEncoding(nn.Module):
    """位置编码"""
    
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
    """因子预测 Transformer"""
    
    def __init__(self, n_factors: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        
        self.n_factors = n_factors
        self.d_model = d_model
        
        # ✅ 修复：添加 LayerNorm 提高稳定性
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
            norm_first=True  # ✅ 修复：Pre-LN 架构更稳定
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # ✅ 修复：回归头添加 LayerNorm
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
        
        self.factor_attention = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
        # ✅ 修复：权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, return_attention=False):
        batch_size, seq_len, _ = x.shape
        
        x = x.permute(1, 0, 2)
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        
        encoded = self.transformer_encoder(x)
        
        pooled = encoded.permute(1, 2, 0)
        pooled = self.pooling(pooled).squeeze(-1)
        
        reg_out = self.regression_head(pooled).squeeze(-1)
        cls_out = self.classification_head(pooled)
        
        # ✅ 修复：限制输出范围防止 Inf
        reg_out = torch.tanh(reg_out) * 0.1  # 限制在 [-0.1, 0.1]
        
        if return_attention:
            last_step = encoded[-1, :, :]
            factor_weights = self.factor_attention(last_step)
            return reg_out, cls_out, factor_weights
        
        return reg_out, cls_out
    
    def get_factor_importance(self, dataloader, device):
        """计算因子重要性"""
        self.eval()
        all_weights = []
        
        with torch.no_grad():
            for seq, _, _ in dataloader:
                seq = seq.to(device)
                _, _, weights = self.forward(seq, return_attention=True)
                all_weights.append(weights.cpu())
        
        avg_weights = torch.cat(all_weights).mean(dim=0).numpy()
        return avg_weights.flatten()


# ============================
# 训练器 (已修复 Tensor 问题)
# ============================
# ============================
# 训练器 (已修复 NaN 问题)
# ============================
class TransformerTrainer:
    """Transformer 训练器"""
    
    def __init__(self, model: nn.Module, config: Config):
        self.model = model.to(config.DEVICE)
        self.config = config
        
        self.reg_criterion = nn.MSELoss()
        self.cls_criterion = nn.CrossEntropyLoss()
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=1e-4
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_ic': [], 'val_ic': [],
            'train_acc': [], 'val_acc': []
        }
    
    def train_epoch(self, dataloader):
        """训练一个 epoch"""
        self.model.train()
        total_reg_loss = 0
        total_cls_loss = 0
        all_preds = []
        all_targets = []
        
        for seq, target_reg, target_cls in dataloader:
            seq = seq.to(self.config.DEVICE)
            target_reg = target_reg.to(self.config.DEVICE)
            target_cls = target_cls.to(self.config.DEVICE)
            
            pred_reg, pred_cls = self.model(seq)
            
            # ✅ 修复 1: 检查预测值是否包含 NaN
            if torch.isnan(pred_reg).any() or torch.isinf(pred_reg).any():
                print(f"    ⚠️ 检测到 NaN/Inf，跳过此 batch")
                continue
            
            reg_loss = self.reg_criterion(pred_reg, target_reg)
            cls_loss = self.cls_criterion(pred_cls, target_cls)
            
            # ✅ 修复 2: 检查损失是否有效
            if torch.isnan(reg_loss) or torch.isinf(reg_loss):
                print(f"    ⚠️ 损失为 NaN/Inf，跳过此 batch")
                continue
            
            loss = reg_loss + 0.5 * cls_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # ✅ 修复 3: 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_reg_loss += reg_loss.item()
            total_cls_loss += cls_loss.item()
            
            # ✅ 修复 4: 使用 detach().cpu().numpy() 并检查 NaN
            pred_np = pred_reg.detach().cpu().numpy()
            target_np = target_reg.cpu().numpy()
            
            # 过滤 NaN 值
            valid_mask = ~(np.isnan(pred_np) | np.isnan(target_np))
            if valid_mask.sum() > 0:
                all_preds.extend(pred_np[valid_mask].tolist())
                all_targets.extend(target_np[valid_mask].tolist())
        
        # ✅ 修复 5: 确保有足够数据计算指标
        if len(all_preds) < 10:
            print(f"    ⚠️ 有效预测不足，使用默认指标")
            return {
                'reg_loss': total_reg_loss / max(len(dataloader), 1),
                'cls_loss': total_cls_loss / max(len(dataloader), 1),
                'ic': 0.0,
                'accuracy': 0.5
            }
        
        ic = self._calculate_ic(all_preds, all_targets)
        
        # ✅ 修复 6: 处理 sign 后的 NaN
        pred_sign = np.sign(all_preds)
        target_sign = np.sign(all_targets)
        valid_acc_mask = ~(np.isnan(pred_sign) | np.isnan(target_sign))
        
        if valid_acc_mask.sum() > 0:
            acc = accuracy_score(pred_sign[valid_acc_mask], target_sign[valid_acc_mask])
        else:
            acc = 0.5
        
        return {
            'reg_loss': total_reg_loss / max(len(dataloader), 1),
            'cls_loss': total_cls_loss / max(len(dataloader), 1),
            'ic': ic,
            'accuracy': acc
        }
    
    def validate(self, dataloader):
        """验证"""
        self.model.eval()
        total_reg_loss = 0
        total_cls_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for seq, target_reg, target_cls in dataloader:
                seq = seq.to(self.config.DEVICE)
                target_reg = target_reg.to(self.config.DEVICE)
                target_cls = target_cls.to(self.config.DEVICE)
                
                pred_reg, pred_cls = self.model(seq)
                
                # ✅ 检查 NaN
                if torch.isnan(pred_reg).any() or torch.isinf(pred_reg).any():
                    continue
                
                reg_loss = self.reg_criterion(pred_reg, target_reg)
                cls_loss = self.cls_criterion(pred_cls, target_cls)
                
                if torch.isnan(reg_loss) or torch.isinf(reg_loss):
                    continue
                
                loss = reg_loss + 0.5 * cls_loss
                
                total_reg_loss += reg_loss.item()
                total_cls_loss += cls_loss.item()
                
                pred_np = pred_reg.detach().cpu().numpy()
                target_np = target_reg.cpu().numpy()
                
                valid_mask = ~(np.isnan(pred_np) | np.isnan(target_np))
                if valid_mask.sum() > 0:
                    all_preds.extend(pred_np[valid_mask].tolist())
                    all_targets.extend(target_np[valid_mask].tolist())
        
        if len(all_preds) < 10:
            return {
                'reg_loss': total_reg_loss / max(len(dataloader), 1),
                'cls_loss': total_cls_loss / max(len(dataloader), 1),
                'ic': 0.0,
                'accuracy': 0.5
            }
        
        ic = self._calculate_ic(all_preds, all_targets)
        
        pred_sign = np.sign(all_preds)
        target_sign = np.sign(all_targets)
        valid_acc_mask = ~(np.isnan(pred_sign) | np.isnan(target_sign))
        
        if valid_acc_mask.sum() > 0:
            acc = accuracy_score(pred_sign[valid_acc_mask], target_sign[valid_acc_mask])
        else:
            acc = 0.5
        
        return {
            'reg_loss': total_reg_loss / max(len(dataloader), 1),
            'cls_loss': total_cls_loss / max(len(dataloader), 1),
            'ic': ic,
            'accuracy': acc
        }
    
    def _calculate_ic(self, preds, targets):
        """计算 Rank IC"""
        preds = np.array(preds)
        targets = np.array(targets)
        
        # ✅ 过滤 NaN
        valid_mask = ~(np.isnan(preds) | np.isnan(targets))
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        
        if len(preds) < 10:
            return 0.0
        
        ic, _ = stats.spearmanr(preds, targets)
        return ic if not np.isnan(ic) else 0.0
    
    def train(self, train_loader, val_loader, num_epochs=None):
        """完整训练流程"""
        num_epochs = num_epochs or self.config.NUM_EPOCHS
        best_val_ic = -1
        patience_counter = 0
        
        print(f"\n🚀 开始训练 ({num_epochs} epochs)...")
        print(f"   设备：{self.config.DEVICE}")
        print(f"   训练样本：{len(train_loader.dataset)}")
        print(f"   验证样本：{len(val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            self.history['train_loss'].append(train_metrics['reg_loss'])
            self.history['val_loss'].append(val_metrics['reg_loss'])
            self.history['train_ic'].append(train_metrics['ic'])
            self.history['val_ic'].append(val_metrics['ic'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            self.scheduler.step(val_metrics['reg_loss'])
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}:")
                print(f"    Train Loss: {train_metrics['reg_loss']:.6f}, "
                      f"IC: {train_metrics['ic']:.4f}, "
                      f"Acc: {train_metrics['accuracy']:.4f}")
                print(f"    Val Loss: {val_metrics['reg_loss']:.6f}, "
                      f"IC: {val_metrics['ic']:.4f}, "
                      f"Acc: {val_metrics['accuracy']:.4f}")
            
            if val_metrics['ic'] > best_val_ic:
                best_val_ic = val_metrics['ic']
                patience_counter = 0
                torch.save(self.model.state_dict(), 
                          config.OUTPUT_DIR / 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"  ⏹️ 早停于 epoch {epoch+1}")
                    break
        
        if (config.OUTPUT_DIR / 'best_model.pth').exists():
            self.model.load_state_dict(
                torch.load(config.OUTPUT_DIR / 'best_model.pth', 
                          weights_only=True,
                          map_location=self.config.DEVICE)
            )
        
        print(f"\n✅ 训练完成！最佳验证 IC: {best_val_ic:.4f}")
        return self.history
    
    def evaluate(self, test_loader):
        """测试集评估"""
        self.model.eval()
        all_preds = []
        all_targets = []
        all_cls_preds = []
        all_cls_targets = []
        
        with torch.no_grad():
            for seq, target_reg, target_cls in test_loader:
                seq = seq.to(self.config.DEVICE)
                pred_reg, pred_cls = self.model(seq)
                
                # ✅ 检查 NaN
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
        
        if len(all_preds) < 10:
            print(f"  ⚠️ 有效预测不足，返回默认结果")
            return {
                'ic': 0.0,
                'ic_ir': 0.0,
                'direction_accuracy': 0.5,
                'precision': 0.5,
                'recall': 0.5,
                'f1': 0.5,
                'quantile_returns': {0: 0, 1: 0},
                'predictions': np.array([]),
                'targets': np.array([])
            }
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # ✅ 恢复目标变量原始尺度 (用于分析)
        # all_preds = np.array(all_preds) * self.target_std + self.target_mean
        # all_targets = np.array(all_targets) * self.target_std + self.target_mean
                
        ic = self._calculate_ic(all_preds, all_targets)
        
        # ICIR 计算
        ic_samples = [self._calculate_ic(
            np.random.permutation(all_preds), all_targets
        ) for _ in range(10)]
        ic_std = np.std(ic_samples)
        ic_ir = ic / ic_std if ic_std > 0 else 0
        
        # 方向准确率
        pred_sign = np.sign(all_preds)
        target_sign = np.sign(all_targets)
        valid_mask = ~(np.isnan(pred_sign) | np.isnan(target_sign))
        if valid_mask.sum() > 0:
            direction_acc = accuracy_score(pred_sign[valid_mask], target_sign[valid_mask])
        else:
            direction_acc = 0.5
        
        # 分类指标
        try:
            cls_metrics = precision_recall_fscore_support(
                all_cls_targets, all_cls_preds, average='binary', zero_division=0
            )
        except:
            cls_metrics = (0.5, 0.5, 0.5, 0)
        
        quantile_returns = self._quantile_backtest(all_preds, all_targets)
        
        results = {
            'ic': ic,
            'ic_ir': ic_ir,
            'direction_accuracy': direction_acc,
            'precision': cls_metrics[0],
            'recall': cls_metrics[1],
            'f1': cls_metrics[2],
            'quantile_returns': quantile_returns,
            'predictions': all_preds,
            'targets': all_targets
        }
        
        return results
    
    def _quantile_backtest(self, preds, targets, n_quantiles=5):
        """分层回测"""
        try:
            # ✅ 过滤 NaN
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
class EvaluationVisualizer:
    """评估结果可视化"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_training_history(self, history: dict, symbol: str):
        """训练历史图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].plot(history['train_loss'], label='Train', alpha=0.7)
        axes[0, 0].plot(history['val_loss'], label='Val', alpha=0.7)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(history['train_ic'], label='Train IC', alpha=0.7)
        axes[0, 1].plot(history['val_ic'], label='Val IC', alpha=0.7)
        axes[0, 1].axhline(0.05, color='green', linestyle='--', alpha=0.5)
        axes[0, 1].axhline(-0.05, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IC')
        axes[0, 1].set_title('Information Coefficient')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(history['train_acc'], label='Train', alpha=0.7)
        axes[1, 0].plot(history['val_acc'], label='Val', alpha=0.7)
        axes[1, 0].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Direction Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].scatter(range(len(history['val_ic'])), history['val_ic'], alpha=0.6)
        axes[1, 1].axhline(np.mean(history['val_ic']), color='red', linestyle='--')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Val IC')
        axes[1, 1].set_title(f'Val IC Distribution (Mean={np.mean(history["val_ic"]):.4f})')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存训练历史图：{symbol}_training_history.png")
    
    def plot_quantile_returns(self, quantile_returns: dict, symbol: str):
        """分层收益图"""
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
            ax.axhline(long_short, color='blue', linestyle='--', 
                      label=f'Long-Short: {long_short:.6f}')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_quantile_returns.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存分层收益图：{symbol}_quantile_returns.png")
    
    def plot_factor_importance(self, factor_names: list, importance: np.ndarray, 
                               symbol: str, top_n: int = 20):
        """因子重要性图"""
        indices = np.argsort(np.abs(importance))[::-1][:top_n]
        top_factors = [factor_names[i] for i in indices]
        top_importance = importance[indices]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['green' if x > 0 else 'red' for x in top_importance]
        ax.barh(range(len(top_factors)), top_importance, color=colors)
        ax.set_yticks(range(len(top_factors)))
        ax.set_yticklabels(top_factors, fontsize=9)
        ax.set_xlabel('Importance Weight')
        ax.set_title(f'Top {top_n} Factor Importance')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{symbol}_factor_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 保存因子重要性图：{symbol}_factor_importance.png")
    
    def plot_prediction_scatter(self, predictions: np.ndarray, targets: np.ndarray, 
                                symbol: str):
        """预测 vs 实际散点图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].scatter(targets, predictions, alpha=0.3, s=10)
        axes[0].plot([targets.min(), targets.max()], 
                    [targets.min(), targets.max()], 'r--', linewidth=2)
        axes[0].set_xlabel('Actual Return')
        axes[0].set_ylabel('Predicted Return')
        axes[0].set_title('Prediction vs Actual')
        axes[0].grid(True, alpha=0.3)
        
        direction_correct = np.sign(predictions) == np.sign(targets)
        axes[1].hist(targets[direction_correct], bins=50, alpha=0.7, 
                    label='Correct', color='green')
        axes[1].hist(targets[~direction_correct], bins=50, alpha=0.7, 
                    label='Wrong', color='red')
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
# 主评估流程 (已修复时间戳问题)
# ============================
def evaluate_symbol(symbol: str, config: Config) -> dict:
    """评估单个交易对"""
    print(f"\n{'='*60}")
    print(f"🧠 评估交易对：{symbol}")
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
            # ✅ 修复：读取时不解析时间戳，后续处理
            df = pd.read_csv(f, compression='gzip')
            
            # ✅ 修复：智能处理时间戳列
            time_cols = ['timestamp', 'timestampes', 'time']
            time_col_found = None
            for col in time_cols:
                if col in df.columns:
                    time_col_found = col
                    break
            
            if time_col_found:
                # ✅ 修复：使用 ISO8601 格式解析，支持多种时间格式
                try:
                    df[time_col_found] = pd.to_datetime(df[time_col_found], format='ISO8601', utc=True)
                except:
                    try:
                        df[time_col_found] = pd.to_datetime(df[time_col_found], format='mixed')
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
    print(full_df.info())
    
    # 2. 创建数据集
    dataset = FactorSequenceDataset(
        full_df,
        seq_length=config.SEQ_LENGTH,
        prediction_horizon=config.PREDICTION_HORIZON
    )
    
    if len(dataset) < 100:
        print(f"  ❌ 有效序列不足：{len(dataset)}")
        return {'status': 'failed', 'reason': 'insufficient_data'}
    
    print(f"factor names: {dataset.get_factor_names()}, target name: {dataset.target_col}  ")
    # 3. 数据分割
    n_samples = len(dataset)
    train_size = int(n_samples * config.TRAIN_RATIO)
    val_size = int(n_samples * config.VAL_RATIO)
    test_size = n_samples - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"  📊 数据分割：训练={train_size}, 验证={val_size}, 测试={test_size}")
    
    # 4. 创建模型
    n_factors = len(dataset.get_factor_names())
    print(f"  📊 因子数量：{n_factors}")

    model = FactorTransformer(
        n_factors=n_factors,
        d_model=config.HIDDEN_DIM,
        nhead=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    )
    
    print(f"  🧠 模型参数：{sum(p.numel() for p in model.parameters()):,}")
    
    # 5. 训练
    trainer = TransformerTrainer(model, config)
    history = trainer.train(train_loader, val_loader)
    
    # 6. 测试评估
    print(f"\n📊 测试集评估...")
    test_results = trainer.evaluate(test_loader)
    
    # 7. 因子重要性
    print(f"  🔍 计算因子重要性...")
    factor_importance = model.get_factor_importance(test_loader, config.DEVICE)

    # ✅ 修复：长度检查
    factor_names = dataset.get_factor_names()
    if len(factor_importance) != len(factor_names):
        print(f"  ⚠️ 因子重要性长度不匹配：{len(factor_importance)} vs {len(factor_names)}")
        min_len = min(len(factor_importance), len(factor_names))
        factor_names = factor_names[:min_len]
        factor_importance = factor_importance[:min_len]
    
    # 8. 可视化
    print(f"  🎨 生成可视化...")
    visualizer = EvaluationVisualizer(config.OUTPUT_DIR / symbol)
    visualizer.plot_training_history(history, symbol)
    visualizer.plot_quantile_returns(test_results['quantile_returns'], symbol)
    visualizer.plot_factor_importance(
        dataset.get_factor_names(), factor_importance, symbol
    )
    visualizer.plot_prediction_scatter(
        test_results['predictions'], test_results['targets'], symbol
    )
    
    # 9. 保存结果
    summary = {
        'symbol': symbol,
        'n_samples': n_samples,
        'n_factors': len(factor_names),
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
        'top_5_factors': [
            factor_names[i] 
            for i in np.argsort(np.abs(factor_importance))[::-1][:5]
        ],
        'status': 'success'
    }
    
    # ✅ 修复：保存因子重要性
    if len(factor_names) > 0 and len(factor_importance) > 0:
        pd.DataFrame({
            'factor': factor_names,
            'importance': factor_importance
        }).to_csv(config.OUTPUT_DIR / symbol / f'{symbol}_factor_importance.csv', index=False)
        print(f"  💾 保存因子重要性：{len(factor_names)} 个因子")
    else:
        print(f"  ⚠️ 跳过因子重要性保存 (长度={len(factor_names)}, {len(factor_importance)})")
    
    # pd.DataFrame({
    #     'factor': dataset.get_factor_names(),
    #     'importance': factor_importance
    # }).to_csv(config.OUTPUT_DIR / symbol / f'{symbol}_factor_importance.csv', index=False)

    # ✅ 修复后：确保长度一致
    factor_names = dataset.get_factor_names()
    importance_len = len(factor_importance)
    names_len = len(factor_names)

    print(f"  📊 因子名称数：{names_len}, 重要性数组长度：{importance_len}")

    # 取较小长度，避免不匹配
    min_len = min(names_len, importance_len)
    factor_names_trimmed = factor_names[:min_len]
    importance_trimmed = factor_importance[:min_len]

    pd.DataFrame({
        'factor': factor_names_trimmed,
        'importance': importance_trimmed
    }).to_csv(config.OUTPUT_DIR / symbol / f'{symbol}_factor_importance.csv', index=False)

    pd.DataFrame(test_results['predictions'], columns=['prediction']).to_csv(
        config.OUTPUT_DIR / symbol / f'{symbol}_predictions.csv', index=False
    )
    
    print(f"\n  📋 评估摘要:")
    print(f"     测试 IC: {summary['test_ic']:.4f}")
    print(f"     ICIR: {summary['test_ic_ir']:.4f}")
    print(f"     方向准确率：{summary['direction_accuracy']:.4f}")
    print(f"     多空收益：{summary['long_short_return']:.6f}")
    print(f"     Top 因子：{summary['top_5_factors'][:3]}")
    
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
        print("  ❌ 无成功评估的交易对")
        return
    
    summary_df.to_csv(config.OUTPUT_DIR / "all_symbols_evaluation.csv", index=False)
    
    top_by_ic = summary_df.nlargest(5, 'test_ic')
    top_by_acc = summary_df.nlargest(5, 'direction_accuracy')
    
    print("\n🏆 Top 5 交易对 (按测试 IC):")
    for _, row in top_by_ic.iterrows():
        print(f"   {row['symbol']}: IC={row['test_ic']:.4f}, "
              f"准确率={row['direction_accuracy']:.4f}")
    
    print("\n🏆 Top 5 交易对 (按方向准确率):")
    for _, row in top_by_acc.iterrows():
        print(f"   {row['symbol']}: IC={row['test_ic']:.4f}, "
              f"准确率={row['direction_accuracy']:.4f}")
    
    print(f"\n💾 汇总报告：{config.OUTPUT_DIR / 'all_symbols_evaluation.csv'}")


# ============================
# 主程序入口
# ============================
if __name__ == "__main__":
    print("="*60)
    print("🚀 Transformer 高频因子预测能力检验")
    print("="*60)
    print(f"📁 因子目录：{config.FACTOR_DIR}")
    print(f"📁 输出目录：{config.OUTPUT_DIR}")
    print(f"📐 序列长度：{config.SEQ_LENGTH}")
    print(f"🎯 预测期：{config.PREDICTION_HORIZON}")
    print(f"🧠 模型维度：{config.HIDDEN_DIM}")
    print(f"📦 Batch Size: {config.BATCH_SIZE}")
    print("="*60)
    
    symbols = discover_symbols(config)
    
    if not symbols:
        print("❌ 未发现任何交易对因子数据")
        exit(1)
    
    all_summaries = []
    for i, symbol in enumerate(symbols[:10], 1):
        print(f"\n[{i}/{min(10, len(symbols))}] 处理进度")
        try:
            summary = evaluate_symbol(symbol, config)
            all_summaries.append(summary)
        except Exception as e:
            print(f"❌ {symbol} 评估失败：{e}")
            import traceback
            traceback.print_exc()
            all_summaries.append({
                'symbol': symbol,
                'status': 'failed',
                'error': str(e)
            })
    
    generate_summary_report(all_summaries, config)
    
    print("\n" + "="*60)
    print("🎉 Transformer 因子检验完成!")
    print("="*60)