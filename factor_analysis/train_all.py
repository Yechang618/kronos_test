#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Model Factor Training Script
支持 Transformer、Linear Regression、Logistic Regression、LightGBM 训练
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import pickle
import json
import argparse
warnings.filterwarnings('ignore')

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 机器学习
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats

# 尝试导入 LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ 警告：lightgbm 未安装，将跳过 LightGBM 模型")

# 本地模块
from transformer import FactorTransformer, get_model_info

# ============================
# 配置区域
# ============================
class TrainConfig:
    # 数据路径
    FACTOR_DIR = Path("./datasets/factors/hf_factors")
    OUTPUT_DIR = Path("./datasets/model_training")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 序列参数
    SEQ_LENGTH = 60              # 30 秒 @ 500ms = 60 个时间点
    PREDICTION_HORIZON = 1       # 预测下一秒
    TARGET_COL = 'basis_ret_future_1'
    DEFAULT_TARGET_COL = 'kalman_spot_filtered'
    # DEFAULT_TARGET_COL =  'spot_mid'
    
    # Transformer 参数
    HIDDEN_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 3
    DROPOUT = 0.2
    
    # 训练参数
    BATCH_SIZE = 640
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    
    # 数据分割
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # 设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 随机种子
    RANDOM_SEED = 42
    
    # 模型类型
    MODEL_TYPE = 'transformer'  # transformer, linear, logistic, lightgbm

config = TrainConfig()


# ============================
# 数据集类
# ============================
class FactorSequenceDataset(Dataset):
    """因子序列数据集 (用于 Transformer)"""
    def __init__(self, factor_df: pd.DataFrame, seq_length: int,
                 prediction_horizon: int, target_col: str = 'target'):
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        self.target_col = target_col
        
        exclude_cols = [target_col, 'target', 'timestamp', 'year_month', 'timestampes']
        self.factor_cols = [c for c in factor_df.columns if c not in exclude_cols]
        
        factor_data_raw = factor_df[self.factor_cols].copy()
        for col in self.factor_cols:
            if col in factor_data_raw.columns:
                lower = factor_data_raw[col].quantile(0.01)
                upper = factor_data_raw[col].quantile(0.99)
                factor_data_raw[col] = factor_data_raw[col].clip(lower, upper)
        
        factor_data_raw = factor_data_raw.fillna(0)
        self.scaler = RobustScaler()
        self.factor_data = self.scaler.fit_transform(factor_data_raw)
        
        if target_col not in factor_df.columns:
            print(f"  {target_col} 列不存在，使用 '{config.DEFAULT_TARGET_COL}' 计算未来收益作为目标")
            self.target_col = f"{config.DEFAULT_TARGET_COL}_{prediction_horizon}"
            self.target_original_col = config.DEFAULT_TARGET_COL
            self.targets = factor_df[config.DEFAULT_TARGET_COL].shift(-prediction_horizon).pct_change(
                prediction_horizon).values
        else:
            self.targets = factor_df[target_col].values
        
        self.targets = np.nan_to_num(self.targets, nan=0.0, posinf=0.0, neginf=0.0)
        self.target_mean = np.mean(self.targets)
        self.target_std = np.std(self.targets) + 1e-10
        self.targets = (self.targets - self.target_mean) / self.target_std
        self.targets = np.clip(self.targets, -5, 5)
        
        self.valid_indices = self._get_valid_indices()
        print(f"  📊 数据集：{len(self.valid_indices)} 个有效序列，{len(self.factor_cols)} 个因子")
    
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
        return (torch.FloatTensor(seq), 
                torch.FloatTensor([target_reg])[0], 
                torch.LongTensor([target_cls])[0])
    
    def get_factor_names(self):
        return self.factor_cols

    def get_target_col(self):
        return self.target_col, self.target_original_col if hasattr(self, 'target_original_col') else self.target_col
    
    def get_scaler(self):
        return self.scaler
    
    def get_target_stats(self):
        return {'mean': self.target_mean, 'std': self.target_std}


class FactorFlatDataset(Dataset):
    """扁平化因子数据集 (用于 Linear/Logistic/LightGBM)"""
    def __init__(self, factor_df: pd.DataFrame, seq_length: int,
                 prediction_horizon: int, target_col: str = 'target'):
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        self.target_col = target_col
        
        exclude_cols = [target_col, 'target', 'timestamp', 'year_month', 'timestampes']
        self.factor_cols = [c for c in factor_df.columns if c not in exclude_cols]
        
        factor_data_raw = factor_df[self.factor_cols].copy()
        for col in self.factor_cols:
            if col in factor_data_raw.columns:
                lower = factor_data_raw[col].quantile(0.01)
                upper = factor_data_raw[col].quantile(0.99)
                factor_data_raw[col] = factor_data_raw[col].clip(lower, upper)
        
        factor_data_raw = factor_data_raw.fillna(0)
        self.scaler = RobustScaler()
        self.factor_data = self.scaler.fit_transform(factor_data_raw)
        
        if target_col not in factor_df.columns:
            print(f"  {target_col} 列不存在，使用 '{config.DEFAULT_TARGET_COL}' 计算未来收益作为目标")
            self.target_col = f"{config.DEFAULT_TARGET_COL}_{prediction_horizon}"
            self.target_original_col = config.DEFAULT_TARGET_COL
            self.targets = factor_df[config.DEFAULT_TARGET_COL].shift(-prediction_horizon).pct_change(
                prediction_horizon).values
        else:
            self.targets = factor_df[target_col].values
        
        self.targets = np.nan_to_num(self.targets, nan=0.0, posinf=0.0, neginf=0.0)
        self.target_mean = np.mean(self.targets)
        self.target_std = np.std(self.targets) + 1e-10
        self.targets = (self.targets - self.target_mean) / self.target_std
        self.targets = np.clip(self.targets, -5, 5)
        
        # 使用最后一个时间点的因子作为特征
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
            valid.append(i)
        return valid
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        idx = self.valid_indices[idx]
        features = self.factor_data[idx]
        target_reg = self.targets[idx + self.prediction_horizon]
        target_cls = 1 if target_reg > 0 else 0
        return (torch.FloatTensor(features),
                torch.FloatTensor([target_reg])[0],
                torch.LongTensor([target_cls])[0])
    
    def get_factor_names(self):
        return self.factor_cols

    def get_target_col(self):
        return self.target_col, self.target_original_col if hasattr(self, 'target_original_col') else self.target_col
    
    def get_scaler(self):
        return self.scaler
    
    def get_target_stats(self):
        return {'mean': self.target_mean, 'std': self.target_std}
    
    def get_numpy_data(self):
        """获取 numpy 格式数据 (用于 sklearn/lightgbm)"""
        X = []
        y_reg = []
        y_cls = []
        for idx in self.valid_indices:
            X.append(self.factor_data[idx])
            target_reg = self.targets[idx + self.prediction_horizon]
            y_reg.append(target_reg)
            y_cls.append(1 if target_reg > 0 else 0)
        return np.array(X), np.array(y_reg), np.array(y_cls)


# ============================
# Transformer 训练器
# ============================
class TransformerTrainer:
    def __init__(self, model: nn.Module, config: TrainConfig):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.reg_criterion = nn.MSELoss()
        self.cls_criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.history = {'train_loss': [], 'val_loss': [], 'train_ic': [], 'val_ic': [], 'train_acc': [], 'val_acc': []}
    
    def _calculate_ic(self, preds, targets):
        preds = np.array(preds)
        targets = np.array(targets)
        valid_mask = ~(np.isnan(preds) | np.isnan(targets))
        preds, targets = preds[valid_mask], targets[valid_mask]
        if len(preds) < 10:
            return 0.0
        ic, _ = stats.spearmanr(preds, targets)
        return ic if not np.isnan(ic) else 0.0
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_reg_loss, total_cls_loss = 0, 0
        all_preds, all_targets = [], []
        
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
        valid_acc_mask = ~(np.isnan(pred_sign) | np.isnan(target_sign))
        acc = accuracy_score(pred_sign[valid_acc_mask], target_sign[valid_acc_mask]) if valid_acc_mask.sum() > 0 else 0.5
        
        return {'reg_loss': total_reg_loss / max(len(dataloader), 1),
                'cls_loss': total_cls_loss / max(len(dataloader), 1),
                'ic': ic, 'accuracy': acc}
    
    def validate(self, dataloader):
        self.model.eval()
        total_reg_loss, total_cls_loss = 0, 0
        all_preds, all_targets = [], []
        
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
        
        return {'reg_loss': total_reg_loss / max(len(dataloader), 1),
                'cls_loss': total_cls_loss / max(len(dataloader), 1),
                'ic': ic, 'accuracy': acc}
    
    def train(self, train_loader, val_loader, symbol_output_dir=None, num_epochs=None):
        num_epochs = num_epochs or self.config.NUM_EPOCHS
        best_val_ic = -1
        patience_counter = 0
        
        if symbol_output_dir is None:
            symbol_output_dir = self.config.OUTPUT_DIR
        
        print(f"  🚀 开始训练 ({num_epochs} epochs)...")
        
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
                print(f"    Epoch {epoch+1}/{num_epochs}:")
                print(f"      Train Loss: {train_metrics['reg_loss']:.6f}, IC: {train_metrics['ic']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
                print(f"      Val Loss: {val_metrics['reg_loss']:.6f}, IC: {val_metrics['ic']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            
            if val_metrics['ic'] > best_val_ic:
                best_val_ic = val_metrics['ic']
                patience_counter = 0
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
        
        print(f"  ✅ 训练完成！最佳验证 IC: {best_val_ic:.4f}")
        return self.history


# ============================
# Sklearn/LightGBM 训练器
# ============================
# ============================
# Sklearn/LightGBM 训练器（修复版）
# ============================
class SklearnTrainer:
    def __init__(self, model_type: str, config: TrainConfig):
        self.model_type = model_type
        self.config = config
        self.reg_model = None
        self.cls_model = None
        self.history = {'train_loss': [], 'val_loss': [], 'train_ic': [], 'val_ic': [], 'train_acc': [], 'val_acc': []}
    
    def _calculate_ic(self, preds, targets):
        preds = np.array(preds)
        targets = np.array(targets)
        valid_mask = ~(np.isnan(preds) | np.isnan(targets))
        preds, targets = preds[valid_mask], targets[valid_mask]
        if len(preds) < 10:
            return 0.0
        ic, _ = stats.spearmanr(preds, targets)
        return ic if not np.isnan(ic) else 0.0
    
    def train(self, train_dataset, val_dataset, symbol_output_dir=None):
        if symbol_output_dir is None:
            symbol_output_dir = self.config.OUTPUT_DIR
        
        # ✅ 修复：支持 dict 和 Dataset 两种格式
        if isinstance(train_dataset, dict):
            X_train = train_dataset['X']
            y_train_reg = train_dataset['y_reg']
            y_train_cls = train_dataset['y_cls']
            X_val = val_dataset['X']
            y_val_reg = val_dataset['y_reg']
            y_val_cls = val_dataset['y_cls']
        else:
            X_train, y_train_reg, y_train_cls = train_dataset.get_numpy_data()
            X_val, y_val_reg, y_val_cls = val_dataset.get_numpy_data()
        
        print(f"  🚀 开始训练 {self.model_type.upper()}...")
        print(f"    训练样本：{len(X_train)}, 验证样本：{len(X_val)}")
        
        # ✅ 检查标签分布（针对 logistic）
        if self.model_type == 'logistic':
            pos_ratio = np.mean(y_train_cls == 1)
            print(f"    标签分布：0 类={np.mean(y_train_cls == 0):.2%}, 1 类={pos_ratio:.2%}")
            if pos_ratio < 0.1 or pos_ratio > 0.9:
                print(f"    ⚠️ 警告：类别严重不平衡，可能影响训练效果")
        
        if self.model_type == 'linear':
            self.reg_model = LinearRegression()
            self.reg_model.fit(X_train, y_train_reg)
            train_pred = self.reg_model.predict(X_train)
            val_pred = self.reg_model.predict(X_val)
        elif self.model_type == 'logistic':
            self.cls_model = LogisticRegression(
                max_iter=1000, 
                random_state=self.config.RANDOM_SEED,
                class_weight='balanced'  # ✅ 处理类别不平衡
            )
            self.cls_model.fit(X_train, y_train_cls)
            train_pred = self.cls_model.predict_proba(X_train)[:, 1]
            val_pred = self.cls_model.predict_proba(X_val)[:, 1]
            # 转换为回归格式
            train_pred = (train_pred - 0.5) * 2 * 0.1
            val_pred = (val_pred - 0.5) * 2 * 0.1
        elif self.model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM 未安装")
            self.reg_model = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                random_state=self.config.RANDOM_SEED,
                verbose=-1
            )
            self.reg_model.fit(X_train, y_train_reg, eval_set=[(X_val, y_val_reg)], 
                              callbacks=[lgb.early_stopping(50, verbose=False)])
            train_pred = self.reg_model.predict(X_train)
            val_pred = self.reg_model.predict(X_val)
        
        # 计算指标
        train_ic = self._calculate_ic(train_pred, y_train_reg)
        val_ic = self._calculate_ic(val_pred, y_val_reg)
        train_acc = accuracy_score(np.sign(train_pred), np.sign(y_train_reg))
        val_acc = accuracy_score(np.sign(val_pred), np.sign(y_val_reg))
        
        # 填充历史 (sklearn 模型不迭代)
        for _ in range(10):
            self.history['train_loss'].append(0)
            self.history['val_loss'].append(0)
            self.history['train_ic'].append(train_ic)
            self.history['val_ic'].append(val_ic)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
        
        # 保存模型
        model_data = {
            'reg_model': self.reg_model,
            'cls_model': self.cls_model,
            'val_ic': val_ic
        }
        with open(symbol_output_dir / 'best_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"  ✅ 训练完成！验证 IC: {val_ic:.4f}, 验证 Acc: {val_acc:.4f}")
        
        return self.history

# ============================
# 主训练流程
# ============================
# ============================
# 主训练流程（修复版）
# ============================
def train_symbol(symbol: str, config: TrainConfig, model_type: str = 'transformer') -> dict:
    config.MODEL_TYPE = model_type
    print(f"\n{'='*60}")
    print(f"🧠 训练交易对：{symbol} (模型：{model_type})")
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
    print(f"factor_columns: {', '.join([c for c in full_df.columns if c not in ['timestamp', 'year_month', 'timestampes']])}")
    
    # 2. 创建数据集
    if model_type == 'transformer':
        dataset = FactorSequenceDataset(full_df, seq_length=config.SEQ_LENGTH,
                                        prediction_horizon=config.PREDICTION_HORIZON,
                                        target_col=config.TARGET_COL)
        target_col, target_col_original = dataset.get_target_col()
        print(f"  🎯 目标列：{target_col}")
    else:
        dataset = FactorFlatDataset(full_df, seq_length=config.SEQ_LENGTH,
                                    prediction_horizon=config.PREDICTION_HORIZON,
                                    target_col=config.TARGET_COL)
        target_col, target_col_original = dataset.get_target_col()
        print(f"  🎯 目标列：{target_col}")
    
    if len(dataset) < 100:
        print(f"  ❌ 有效序列不足：{len(dataset)}")
        return {'status': 'failed', 'reason': 'insufficient_data'}
    
    # 3. 数据分割
    n_samples = len(dataset)
    train_size = int(n_samples * config.TRAIN_RATIO)
    val_size = int(n_samples * config.VAL_RATIO)
    test_size = n_samples - train_size - val_size
    
    # ✅ 修复：根据模型类型采用不同的分割方式
    if model_type == 'transformer':
        # Transformer 使用 torch 的 random_split
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(config.RANDOM_SEED)
        )
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    else:
        # ✅ sklearn/LightGBM 先获取 numpy 数据再分割
        X, y_reg, y_cls = dataset.get_numpy_data()
        
        # 先分出测试集
        X_temp, X_test, y_reg_temp, y_reg_test, y_cls_temp, y_cls_test = train_test_split(
            X, y_reg, y_cls, test_size=config.TEST_RATIO, 
            random_state=config.RANDOM_SEED
        )
        
        # 再分出训练集和验证集
        val_ratio = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
        X_train, X_val, y_reg_train, y_reg_val, y_cls_train, y_cls_val = train_test_split(
            X_temp, y_reg_temp, y_cls_temp, test_size=val_ratio, 
            random_state=config.RANDOM_SEED
        )
        
        # 创建简单的数据集对象用于训练器
        train_dataset = {
            'X': X_train, 'y_reg': y_reg_train, 'y_cls': y_cls_train,
            'get_numpy_data': lambda: (X_train, y_reg_train, y_cls_train)
        }
        val_dataset = {
            'X': X_val, 'y_reg': y_reg_val, 'y_cls': y_cls_val,
            'get_numpy_data': lambda: (X_val, y_reg_val, y_cls_val)
        }
        
        train_loader = train_dataset
        val_loader = val_dataset
    
    print(f"  📊 数据分割：训练={train_size}, 验证={val_size}, 测试={test_size}")
    
    # 4. 创建模型
    n_factors = len(dataset.get_factor_names())
    print(f"  📊 因子数量：{n_factors}")
    
    # 5. 训练
    symbol_output_dir = config.OUTPUT_DIR / symbol / model_type
    symbol_output_dir.mkdir(parents=True, exist_ok=True)
    
    if model_type == 'transformer':
        model = FactorTransformer(n_factors=n_factors, d_model=config.HIDDEN_DIM,
                                  nhead=config.NUM_HEADS, num_layers=config.NUM_LAYERS,
                                  dropout=config.DROPOUT)
        print(f"  🧠 模型参数：{sum(p.numel() for p in model.parameters()):,}")
        trainer = TransformerTrainer(model, config)
        history = trainer.train(train_loader, val_loader, symbol_output_dir)
        best_val_ic = max(history['val_ic']) if history['val_ic'] else 0
    else:
        trainer = SklearnTrainer(model_type, config)
        history = trainer.train(train_dataset, val_dataset, symbol_output_dir)
        best_val_ic = max(history['val_ic']) if history['val_ic'] else 0
    
    # 6. 保存配置
    train_config = {
        'symbol': symbol,
        'model_type': model_type,
        'seq_length': config.SEQ_LENGTH,
        'prediction_horizon': config.PREDICTION_HORIZON,
        'target_col': target_col,
        'target_col_original': target_col_original,
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
        'best_val_ic': best_val_ic
    }
    
    with open(symbol_output_dir / 'train_config.json', 'w') as f:
        json.dump(train_config, f, indent=2, default=str)
    
    with open(symbol_output_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(dataset.get_scaler(), f)
    
    # 7. 生成摘要
    summary = {
        'symbol': symbol,
        'model_type': model_type,
        'n_samples': n_samples,
        'n_factors': n_factors,
        'best_val_ic': best_val_ic,
        'train_samples': train_size,
        'val_samples': val_size,
        'test_samples': test_size,
        'status': 'success'
    }
    
    print(f"\n  📋 训练摘要:")
    print(f"      最佳验证 IC: {summary['best_val_ic']:.4f}")
    print(f"      模型保存至：{symbol_output_dir}")
    
    return summary


def discover_symbols(config: TrainConfig) -> list:
    if not config.FACTOR_DIR.exists():
        raise FileNotFoundError(f"因子目录不存在：{config.FACTOR_DIR}")
    symbols = [d.name for d in config.FACTOR_DIR.iterdir() if d.is_dir()]
    print(f"🔍 发现 {len(symbols)} 个交易对有因子数据")
    return symbols


def generate_summary_report(summaries: list, config: TrainConfig):
    print(f"\n{'='*60}")
    print("📊 生成训练汇总报告")
    print(f"{'='*60}")
    
    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df[summary_df['status'] == 'success']
    
    if summary_df.empty:
        print("  ❌ 无成功训练的交易对")
        return
    
    summary_df.to_csv(config.OUTPUT_DIR / "all_symbols_training.csv", index=False)
    
    # 按模型类型分组
    for model_type in summary_df['model_type'].unique():
        model_df = summary_df[summary_df['model_type'] == model_type]
        top_by_ic = model_df.nlargest(5, 'best_val_ic')
        print(f"\n🏆 Top 5 交易对 ({model_type.upper()} - 按最佳验证 IC):")
        for _, row in top_by_ic.iterrows():
            print(f"   {row['symbol']}: IC={row['best_val_ic']:.4f}, 因子数={row['n_factors']}")
    
    print(f"\n💾 汇总报告：{config.OUTPUT_DIR / 'all_symbols_training.csv'}")


# ============================
# 主程序入口
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='多模型因子训练脚本')
    parser.add_argument('--symbol', type=str, default='AAVEUSDT', help='交易对名称')
    parser.add_argument('--model', type=str, default='transformer', 
                       choices=['transformer', 'linear', 'logistic', 'lightgbm'],
                       help='模型类型')
    parser.add_argument('--all_models', action='store_true', help='训练所有模型')
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 多模型高频因子训练脚本")
    print("="*60)
    print(f"📁 因子目录：{config.FACTOR_DIR}")
    print(f"📁 输出目录：{config.OUTPUT_DIR}")
    print(f"🧠 设备：{config.DEVICE}")
    print(f"📦 模型：{args.model}")
    print("="*60)
    
    symbols = discover_symbols(config)
    if not symbols:
        print("❌ 未发现任何交易对因子数据")
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
            summary = train_symbol(args.symbol, config, model_type)
            all_summaries.append(summary)
        except Exception as e:
            print(f"❌ {args.symbol} ({model_type}) 训练失败：{e}")
            import traceback
            traceback.print_exc()
            all_summaries.append({'symbol': args.symbol, 'model_type': model_type, 'status': 'failed', 'error': str(e)})
    
    generate_summary_report(all_summaries, config)
    
    print("\n" + "="*60)
    print("🎉 多模型因子训练完成!")
    print("="*60)