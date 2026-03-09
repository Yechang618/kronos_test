#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer Factor Training Script
基于 Transformer 训练高频因子预测模型
输入：30 秒因子序列 | 输出：下一秒 label 预测
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import pickle
import json
warnings.filterwarnings('ignore')

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 可视化
import matplotlib.pyplot as plt

# 机器学习
from sklearn.preprocessing import RobustScaler
from scipy import stats

# ============================
# 配置区域
# ============================
class TrainConfig:
    # 数据路径
    FACTOR_DIR = Path("./datasets/factors/hf_factors")
    OUTPUT_DIR = Path("./datasets/model_training")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ✅ 修改：30 秒输入，预测下一秒
    SEQ_LENGTH = 60              # 30 秒 @ 500ms = 60 个时间点
    PREDICTION_HORIZON = 1       # 预测下一秒
    TARGET_COL = 'basis_ret_future_1'  # 目标列
    
    # 模型参数
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
    
    # 随机种子
    RANDOM_SEED = 42

config = TrainConfig()

# ============================
# 数据集类
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
        
        # 数据预处理
        factor_data_raw = factor_df[self.factor_cols].copy()
        
        # Winsorization 处理极端值
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
        
        # 目标变量处理
        if target_col not in factor_df.columns:
            print(target_col, "列不存在，使用 'mid_basis' 计算未来收益作为目标")
            self.targets = factor_df['mid_basis'].shift(-prediction_horizon).pct_change(
                prediction_horizon
            ).values
        else:
            self.targets = factor_df[target_col].values
        
        # 处理 Inf 和 NaN
        self.targets = np.nan_to_num(self.targets, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 目标变量标准化
        self.target_mean = np.mean(self.targets)
        self.target_std = np.std(self.targets) + 1e-10
        self.targets = (self.targets - self.target_mean) / self.target_std
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
        
        # 确保无 NaN
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        
        seq_tensor = torch.FloatTensor(seq)
        target_reg_tensor = torch.FloatTensor([target_reg])[0]
        target_cls_tensor = torch.LongTensor([target_cls])[0]
        
        return seq_tensor, target_reg_tensor, target_cls_tensor
    
    def get_factor_names(self):
        """获取因子名称列表"""
        return self.factor_cols
    
    def get_scaler(self):
        """获取标准化器"""
        return self.scaler
    
    def get_target_stats(self):
        """获取目标变量统计"""
        return {'mean': self.target_mean, 'std': self.target_std}

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
        
        # 限制输出范围
        reg_out = torch.tanh(reg_out) * 0.1
        
        if return_attention:
            return reg_out, cls_out, encoded
        return reg_out, cls_out

# ============================
# 训练器
# ============================
class TransformerTrainer:
    """Transformer 训练器"""
    
    def __init__(self, model: nn.Module, config: TrainConfig):
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
            
            if torch.isnan(pred_reg).any() or torch.isinf(pred_reg).any():
                continue
            
            reg_loss = self.reg_criterion(pred_reg, target_reg)
            cls_loss = self.cls_criterion(pred_cls, target_cls)
            
            if torch.isnan(reg_loss) or torch.isinf(reg_loss):
                continue
            
            loss = reg_loss + 0.5 * cls_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_reg_loss += reg_loss.item()
            total_cls_loss += cls_loss.item()
            
            pred_np = pred_reg.detach().cpu().numpy()
            target_np = target_reg.cpu().numpy()
            valid_mask = ~(np.isnan(pred_np) | np.isnan(target_np))
            if valid_mask.sum() > 0:
                all_preds.extend(pred_np[valid_mask].tolist())
                all_targets.extend(target_np[valid_mask].tolist())
        
        if len(all_preds) < 10:
            return {'reg_loss': 0, 'cls_loss': 0, 'ic': 0.0, 'accuracy': 0.5}
        
        ic = self._calculate_ic(all_preds, all_targets)
        pred_sign = np.sign(all_preds)
        target_sign = np.sign(all_targets)
        print(f"  🔍 训练 IC: {ic:.4f}, 方向准确率计算中...")
        print(f"[all_preds, all_targets]: {list(zip(all_preds[:10], all_targets[:10]))}")
        print(f"[pred_sign, target_sign]: {list(zip(pred_sign[:10], target_sign[:10]))}")
        valid_acc_mask = ~(np.isnan(pred_sign) | np.isnan(target_sign))
        acc = accuracy_score(pred_sign[valid_acc_mask], target_sign[valid_acc_mask]) if valid_acc_mask.sum() > 0 else 0.5
        
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
            return {'reg_loss': 0, 'cls_loss': 0, 'ic': 0.0, 'accuracy': 0.5}
        
        ic = self._calculate_ic(all_preds, all_targets)
        pred_sign = np.sign(all_preds)
        target_sign = np.sign(all_targets)
        valid_acc_mask = ~(np.isnan(pred_sign) | np.isnan(target_sign))
        acc = accuracy_score(pred_sign[valid_acc_mask], target_sign[valid_acc_mask]) if valid_acc_mask.sum() > 0 else 0.5
        
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
        valid_mask = ~(np.isnan(preds) | np.isnan(targets))
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        if len(preds) < 10:
            return 0.0
        ic, _ = stats.spearmanr(preds, targets)
        return ic if not np.isnan(ic) else 0.0
    
    def train(self, train_loader, val_loader, symbol_output_dir = None, num_epochs=None):
        """完整训练流程"""
        num_epochs = num_epochs or self.config.NUM_EPOCHS
        best_val_ic = -1
        patience_counter = 0
        
        if symbol_output_dir == None:
            symbol_output_dir = self.config.OUTPUT_DIR
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
                print(f"    Train Loss: {train_metrics['reg_loss']:.6f}, IC: {train_metrics['ic']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
                print(f"    Val Loss: {val_metrics['reg_loss']:.6f}, IC: {val_metrics['ic']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            
            if val_metrics['ic'] > best_val_ic:
                best_val_ic = val_metrics['ic']
                patience_counter = 0

                # 保存最佳模型
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_ic': val_metrics['ic']
                }, symbol_output_dir / 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"  ⏹️ 早停于 epoch {epoch+1}")
                    break
        
        print(f"\n✅ 训练完成！最佳验证 IC: {best_val_ic:.4f}")
        return self.history

# ============================
# 可视化器
# ============================
class TrainingVisualizer:
    """训练可视化"""
    
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

# ============================
# 主训练流程
# ============================
def train_symbol(symbol: str, config: TrainConfig) -> dict:
    """训练单个交易对"""
    print(f"\n{'='*60}")
    print(f"🧠 训练交易对：{symbol}")
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
    dataset = FactorSequenceDataset(
        full_df,
        seq_length=config.SEQ_LENGTH,
        prediction_horizon=config.PREDICTION_HORIZON,
        target_col=config.TARGET_COL
    )
    
    if len(dataset) < 100:
        print(f"  ❌ 有效序列不足：{len(dataset)}")
        return {'status': 'failed', 'reason': 'insufficient_data'}
    
    # 3. 数据分割
    n_samples = len(dataset)
    train_size = int(n_samples * config.TRAIN_RATIO)
    val_size = int(n_samples * config.VAL_RATIO)
    test_size = n_samples - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
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
    
    # 6. 保存模型和配置
    symbol_output_dir = config.OUTPUT_DIR / symbol
    symbol_output_dir.mkdir(parents=True, exist_ok=True)

    # 5. 训练
    trainer = TransformerTrainer(model, config)
    history = trainer.train(train_loader, val_loader, symbol_output_dir)
    

    
    # 保存配置
    train_config = {
        'symbol': symbol,
        'seq_length': config.SEQ_LENGTH,
        'prediction_horizon': config.PREDICTION_HORIZON,
        'target_col': config.TARGET_COL,
        'hidden_dim': config.HIDDEN_DIM,
        'num_heads': config.NUM_HEADS,
        'num_layers': config.NUM_LAYERS,
        'dropout': config.DROPOUT,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'n_factors': n_factors,
        'factor_names': dataset.get_factor_names(),
        'target_mean': dataset.get_target_stats()['mean'],
        'target_std': dataset.get_target_stats()['std'],
        'best_val_ic': max(history['val_ic']) if history['val_ic'] else 0
    }
    
    with open(symbol_output_dir / 'train_config.json', 'w') as f:
        json.dump(train_config, f, indent=2, default=str)
    
    # 保存标准化器
    with open(symbol_output_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(dataset.get_scaler(), f)
    
    # 7. 可视化
    visualizer = TrainingVisualizer(symbol_output_dir)
    visualizer.plot_training_history(history, symbol)
    
    # 8. 生成摘要
    summary = {
        'symbol': symbol,
        'n_samples': n_samples,
        'n_factors': n_factors,
        'best_val_ic': max(history['val_ic']) if history['val_ic'] else 0,
        'final_val_ic': history['val_ic'][-1] if history['val_ic'] else 0,
        'train_samples': train_size,
        'val_samples': val_size,
        'test_samples': test_size,
        'status': 'success'
    }
    
    print(f"\n📋 训练摘要:")
    print(f"     最佳验证 IC: {summary['best_val_ic']:.4f}")
    print(f"     最终验证 IC: {summary['final_val_ic']:.4f}")
    print(f"     模型保存至：{symbol_output_dir / 'best_model.pth'}")
    
    return summary


def discover_symbols(config: TrainConfig) -> list:
    """发现所有有因子数据的交易对"""
    if not config.FACTOR_DIR.exists():
        raise FileNotFoundError(f"因子目录不存在：{config.FACTOR_DIR}")
    symbols = [d.name for d in config.FACTOR_DIR.iterdir() if d.is_dir()]
    print(f"🔍 发现 {len(symbols)} 个交易对有因子数据")
    return symbols


def generate_summary_report(summaries: list, config: TrainConfig):
    """生成汇总报告"""
    print(f"\n{'='*60}")
    print("📊 生成训练汇总报告")
    print(f"{'='*60}")
    
    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df[summary_df['status'] == 'success']
    
    if summary_df.empty:
        print("  ❌ 无成功训练的交易对")
        return
    
    summary_df.to_csv(config.OUTPUT_DIR / "all_symbols_training.csv", index=False)
    
    top_by_ic = summary_df.nlargest(5, 'best_val_ic')
    print("\n🏆 Top 5 交易对 (按最佳验证 IC):")
    for _, row in top_by_ic.iterrows():
        print(f"   {row['symbol']}: IC={row['best_val_ic']:.4f}, 因子数={row['n_factors']}")
    
    print(f"\n💾 汇总报告：{config.OUTPUT_DIR / 'all_symbols_training.csv'}")


# ============================
# 主程序入口
# ============================
if __name__ == "__main__":
    from sklearn.metrics import accuracy_score
    
    print("="*60)
    print("🚀 Transformer 高频因子训练脚本")
    print("="*60)
    print(f"📁 因子目录：{config.FACTOR_DIR}")
    print(f"📁 输出目录：{config.OUTPUT_DIR}")
    print(f"📐 序列长度：{config.SEQ_LENGTH} (30 秒 @ 500ms)")
    print(f"🎯 预测期：{config.PREDICTION_HORIZON} (下一秒)")
    print(f"🧠 模型维度：{config.HIDDEN_DIM}")
    print(f"📦 Batch Size: {config.BATCH_SIZE}")
    print(f"🔧 设备：{config.DEVICE}")
    print("="*60)
    
    symbols = discover_symbols(config)
    
    if not symbols:
        print("❌ 未发现任何交易对因子数据")
        exit(1)
    
    all_summaries = []
    
    # 训练指定交易对
    # symbol = 'ZECUSDT'
    symbol = 'AVAXUSDT'
    print(f"\n[1/1] 处理进度")
    try:
        summary = train_symbol(symbol, config)
        all_summaries.append(summary)
    except Exception as e:
        print(f"❌ {symbol} 训练失败：{e}")
        import traceback
        traceback.print_exc()
        all_summaries.append({
            'symbol': symbol,
            'status': 'failed',
            'error': str(e)
        })
    
    generate_summary_report(all_summaries, config)
    
    print("\n" + "="*60)
    print("🎉 Transformer 因子训练完成!")
    print("="*60)