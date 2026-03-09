#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer Model Definition
因子预测 Transformer 模型
"""
import torch
import torch.nn as nn
import numpy as np


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


def get_model_info(model_type: str, n_factors: int, **kwargs) -> dict:
    """获取模型信息"""
    if model_type == 'transformer':
        return {
            'type': 'transformer',
            'n_factors': n_factors,
            'd_model': kwargs.get('d_model', 128),
            'nhead': kwargs.get('nhead', 4),
            'num_layers': kwargs.get('num_layers', 3),
            'dropout': kwargs.get('dropout', 0.2)
        }
    elif model_type in ['linear', 'logistic', 'lightgbm']:
        return {
            'type': model_type,
            'n_factors': n_factors
        }
    else:
        raise ValueError(f"未知模型类型：{model_type}")