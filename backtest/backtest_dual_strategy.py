# backtest/backtest_dual_strategy.py
import os
# Fix OMP warning
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys
import torch

# Add project root
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from config import Config
from model.kronos import Kronos, KronosTokenizer
from model.kronos import sample_from_logits

class DualStrategyBacktest100ms:
    def __init__(self, symbol, start_time, end_time, static_params):
        """
        双策略回测：策略1（静态参数），策略2（动态预测参数）
        """
        self.symbol = symbol
        self.start_time = start_time
        self.end_time = end_time
        self.static_params = static_params
        
        # 回测参数
        self.P1 = 1000.0  # 策略1本金
        self.P2 = 1000.0  # 策略2本金
        self.alpha = 0.1
        self.beta = 0.5
        self.dt = 0.01
        
        # 交易成本
        self.c_t_swap = 0.000153
        self.c_t_spot = 0.0001725
        self.c_m_swap = 0.0
        self.c_m_spot = 0.0000825

        # Total cost
        self.c_tt = self.c_t_swap + self.c_t_spot
        self.c_tm = self.c_m_swap + self.c_m_spot   
        self.c_mt = self.c_t_swap + self.c_m_spot
        
        # 仓位初始化
        self.p1_swap = self.p2_swap = 0.0
        self.p1_spot = self.p2_spot = 0.0
        self.P1_swap = self.P2_swap = 0.0
        
        # 结果记录
        self.pnl1_history = []
        self.pnl2_history = []
        self.timestamps = []
        
        # 数据
        self.raw_df = None
        
        # 动态策略参数
        self.current_dynamic_params = [self.c_tt/2, -self.c_tt/2, self.c_mt/2, -self.c_mt/2, self.c_tm/2, -self.c_tm/2]
        self.predictor = None
        self.tokenizer = None

    def load_data(self):
        data_dir = Path("./backtest/data")
        start_dt = pd.Timestamp(self.start_time)
        end_dt = pd.Timestamp(self.end_time)
        date_range = pd.date_range(start_dt.date(), end_dt.date(), freq="D")
        
        daily_dfs = []
        for date in date_range:
            date_str = date.strftime("%Y-%m-%d")
            filename = f"backtest_{self.symbol}_{date_str}.csv"
            filepath = data_dir / filename
            if filepath.exists():
                df_daily = pd.read_csv(filepath, index_col=0, parse_dates=True)
                daily_dfs.append(df_daily)
        
        if not daily_dfs:
            raise ValueError("No data files found")
        
        self.raw_df = pd.concat(daily_dfs, axis=0)
        self.raw_df = self.raw_df.sort_index()
        self.raw_df = self.raw_df[self.start_time:self.end_time]
        
        # 确保所有需要的列都存在
        required_columns = [
            'spot_bid0_price', 'spot_ask0_price', 'spot_bid0_amount', 'spot_ask0_amount',
            'swap_bid0_price', 'swap_ask0_price', 'swap_bid0_amount', 'swap_ask0_amount'
        ]
        for col in required_columns:
            if col not in self.raw_df.columns:
                self.raw_df[col] = np.nan

    def initialize_predictor(self):
        """初始化 Kronos 预测器"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = KronosTokenizer.from_pretrained(
            "./outputs/models_10min/finetune_tokenizer_all/checkpoints/best_model"
        ).to(device).eval()
        self.predictor_model = Kronos.from_pretrained(
            "./outputs/models_10min/finetune_predictor_all/checkpoints/best_model"
        ).to(device).eval()
        
        class KronosPredictor:
            def __init__(self, model, tokenizer, device, max_context=2048):
                self.model = model
                self.tokenizer = tokenizer
                self.device = device
                self.max_context = max_context
            
            def predict(self, x, x_stamp, y_stamp, pred_len=1, T=1.0, top_p=0.9, top_k=0):
                with torch.no_grad():
                    x = torch.from_numpy(x).unsqueeze(0).to(self.device)
                    x_stamp = torch.from_numpy(x_stamp).unsqueeze(0).to(self.device)
                    y_stamp = torch.from_numpy(y_stamp).unsqueeze(0).to(self.device)
                    x_token = self.tokenizer.encode(x, half=True)
                    initial_seq_len = x.size(1)
                    batch_size = x_token[0].size(0)
                    total_seq_len = initial_seq_len + pred_len
                    full_stamp = torch.cat([x_stamp, y_stamp], dim=1)
                    generated_pre = x_token[0].new_empty(batch_size, pred_len)
                    generated_post = x_token[1].new_empty(batch_size, pred_len)
                    pre_buffer = x_token[0].new_zeros(batch_size, self.max_context)
                    post_buffer = x_token[1].new_zeros(batch_size, self.max_context)
                    buffer_len = min(initial_seq_len, self.max_context)
                    if buffer_len > 0:
                        start_idx = max(0, initial_seq_len - self.max_context)
                        pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
                        post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]
                    for i in range(pred_len):
                        current_seq_len = initial_seq_len + i
                        window_len = min(current_seq_len, self.max_context)
                        if current_seq_len <= self.max_context:
                            input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
                        else:
                            input_tokens = [pre_buffer, post_buffer]
                        context_end = current_seq_len
                        context_start = max(0, context_end - self.max_context)
                        current_stamp = full_stamp[:, context_start:context_end, :].contiguous()
                        s1_logits, context = self.model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
                        s1_logits = s1_logits[:, -1, :]
                        sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)
                        s2_logits = self.model.decode_s2(context, sample_pre)
                        s2_logits = s2_logits[:, -1, :]
                        sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)
                        generated_pre[:, i] = sample_pre.squeeze(-1)
                        generated_post[:, i] = sample_post.squeeze(-1)
                        if current_seq_len < self.max_context:
                            pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                            post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
                        else:
                            pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                            post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                            pre_buffer[:, -1] = sample_pre.squeeze(-1)
                            post_buffer[:, -1] = sample_post.squeeze(-1)
                    full_pre = torch.cat([x_token[0], generated_pre], dim=1)
                    full_post = torch.cat([x_token[1], generated_post], dim=1)
                    context_start = max(0, total_seq_len - self.max_context)
                    input_tokens = [full_pre[:, context_start:total_seq_len].contiguous(), full_post[:, context_start:total_seq_len].contiguous()]
                    z = self.tokenizer.decode(input_tokens, half=True)
                    return z[0, -pred_len:, :].cpu().numpy()
        
        self.predictor = KronosPredictor(self.predictor_model, self.tokenizer, device, max_context=2048)

    # def resample_to_10min(self, df_100ms):
    #     """将100ms数据重采样为10分钟K线"""
    #     def agg_10min(group):
    #         if group.empty:
    #             return pd.Series([np.nan]*6, index=['open', 'high', 'low', 'close', 'volume', 'amount'])
    #         return pd.Series({
    #             'open': group.iloc[0]['spot_bid0_price'],
    #             'high': group[['spot_bid0_price', 'spot_ask0_price']].max().max(),
    #             'low': group[['spot_bid0_price', 'spot_ask0_price']].min().min(),
    #             'close': group.iloc[-1]['spot_bid0_price'],
    #             'volume': group['spot_bid0_amount'].sum(),
    #             'amount': group['spot_ask0_amount'].sum()
    #         })
    #     return df_100ms.resample('10min').apply(agg_10min)

    def resample_to_10min(self, df_100ms):
        """将100ms数据重采样为10分钟K线，使用正确的volume和amount定义"""
        if df_100ms.empty:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'amount'])
        
        # 确保所有需要的列都存在
        required_cols = []
        for prefix in ['spot', 'swap']:
            for level in range(3):  # 前3档
                required_cols.extend([f"{prefix}_bid{level}_amount", f"{prefix}_ask{level}_amount"])
            required_cols.extend([f"{prefix}_bid0_price", f"{prefix}_ask0_price"])
        
        for col in required_cols:
            if col not in df_100ms.columns:
                df_100ms[col] = np.nan
        
        # 移除全 NaN 的行
        df_clean = df_100ms.dropna(subset=required_cols, how='all')
        if df_clean.empty:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'amount'])
        
        # 重采样为10分钟
        resampled = df_clean.resample('10min')
        
        # 计算 Open/Close (使用 spot bid0)
        open_prices = resampled['spot_bid0_price'].first()
        close_prices = resampled['spot_bid0_price'].last()
        
        # 计算 High/Low (取 spot bid0, spot ask0, swap bid0, swap ask0 的极值)
        price_cols = ['spot_bid0_price', 'spot_ask0_price', 'swap_bid0_price', 'swap_ask0_price']
        high_prices = resampled[price_cols].max().max(axis=1)
        low_prices = resampled[price_cols].min().min(axis=1)
        
        # 计算 Volume (合约订单簿不平衡度)
        def calculate_volume(group):
            if group.empty:
                return np.nan
            # swap bids sum (前3档)
            swap_bids_sum = 0
            swap_asks_sum = 0
            for level in range(3):
                bid_col = f'swap_bid{level}_amount'
                ask_col = f'swap_ask{level}_amount'
                if bid_col in group.columns and not group[bid_col].isna().all():
                    swap_bids_sum += group[bid_col].sum()
                if ask_col in group.columns and not group[ask_col].isna().all():
                    swap_asks_sum += group[ask_col].sum()
            
            if swap_bids_sum <= 0 or swap_asks_sum <= 0:
                return np.nan
            return np.log(swap_bids_sum) - np.log(swap_asks_sum)
        
        # 计算 Amount (现货订单簿不平衡度)
        def calculate_amount(group):
            if group.empty:
                return np.nan
            # spot bids sum (前3档)
            spot_bids_sum = 0
            spot_asks_sum = 0
            for level in range(3):
                bid_col = f'spot_bid{level}_amount'
                ask_col = f'spot_ask{level}_amount'
                if bid_col in group.columns and not group[bid_col].isna().all():
                    spot_bids_sum += group[bid_col].sum()
                if ask_col in group.columns and not group[ask_col].isna().all():
                    spot_asks_sum += group[ask_col].sum()
            
            if spot_bids_sum <= 0 or spot_asks_sum <= 0:
                return np.nan
            return np.log(spot_bids_sum) - np.log(spot_asks_sum)
        
        # 应用自定义聚合函数
        volume_series = df_clean.groupby(pd.Grouper(freq='10min')).apply(calculate_volume)
        amount_series = df_clean.groupby(pd.Grouper(freq='10min')).apply(calculate_amount)
        
        # 合并结果
        df_10min = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume_series,
            'amount': amount_series
        })
        
        # 确保索引一致
        df_10min.index = open_prices.index
        
        # 强制转换为数值类型并移除 NaN
        for col in df_10min.columns:
            df_10min[col] = pd.to_numeric(df_10min[col], errors='coerce')
        
        df_10min = df_10min.dropna()
        return df_10min

    def update_dynamic_params(self, current_time):
        """每10分钟更新动态交易参数"""
        # 获取过去144个10分钟K线（24小时）
        start_time = current_time - pd.Timedelta(hours=24)
        df_100ms = self.raw_df[start_time:current_time]
        
        if len(df_100ms) == 0:
            return
            
        df_10min = self.resample_to_10min(df_100ms)
        if len(df_10min) < 144 or df_10min.empty:
            return
        
        # 准备预测输入
        config = Config()
        feature_list = config.feature_list
        time_features = ['minute', 'hour', 'weekday', 'day', 'month']
        
        # 取最后144根K线 - 确保是副本
        x_df = df_10min[-144:].copy()
        
        # 安全地添加时间特征 - 避免 SettingWithCopyWarning
        x_df = x_df.assign(
            minute=x_df.index.minute,
            hour=x_df.index.hour,
            weekday=x_df.index.weekday,
            day=x_df.index.day,
            month=x_df.index.month
        )
        
        # 确保所有特征列存在且为数值类型
        for col in feature_list:
            if col not in x_df.columns:
                x_df[col] = np.nan
            # 强制转换为 float32
            x_df[col] = pd.to_numeric(x_df[col], errors='coerce')
        
        x = x_df[feature_list].values.astype(np.float32)
        x_stamp = x_df[time_features].values.astype(np.float32)
        
        # 移除包含 NaN 的行
        valid_mask = ~np.isnan(x).any(axis=1)
        if not valid_mask.any():
            return
            
        x = x[valid_mask]
        x_stamp = x_stamp[valid_mask]
        
        # 如果数据不足，跳过
        if len(x) == 0:
            return
        
        # 预测未来1个10分钟K线（30个样本）
        N_SAMPLES = 30
        PRED_LENGTH = 1
        
        # Normalize
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x_norm = (x - x_mean) / (x_std + 1e-5)
        x_norm = np.clip(x_norm, -5.0, 5.0)
        
        # **关键修复：确保 y_stamp 是数值类型**
        last_timestamp = x_df.index[-1]
        y_stamp_data = np.array([[
            last_timestamp.minute,
            last_timestamp.hour, 
            last_timestamp.weekday(),
            last_timestamp.day,
            last_timestamp.month
        ]], dtype=np.float32)  # ← 直接创建 float32 数组
        
        y_stamp = np.tile(y_stamp_data, (PRED_LENGTH, 1))
        
        # 预测
        preds = []
        for _ in range(N_SAMPLES):
            pred = self.predictor.predict(
                x=x_norm,
                x_stamp=x_stamp,
                y_stamp=y_stamp,  # ← 现在是 float32
                pred_len=PRED_LENGTH,
                T=0.6,
                top_p=0.9,
                top_k=0
            )
            # 反归一化
            pred = pred * (x_std + 1e-5) + x_mean
            preds.append(pred[0])
        
        if not preds:
            return
            
        preds = np.array(preds)
        
        # 计算 High 和 Low 的统计量
        high_mean = np.mean(preds[:, 1])
        high_std = np.std(preds[:, 1])
        low_mean = np.mean(preds[:, 2])
        low_std = np.std(preds[:, 2])
        
        # 计算动态中点
        dynamic_midpoint = (high_mean + high_std + low_mean - low_std) / 2
        
        # 设置动态参数
        self.current_dynamic_params = [
            dynamic_midpoint + self.c_tt/2,
            dynamic_midpoint - self.c_tt/2,
            dynamic_midpoint + self.c_mt/2,
            dynamic_midpoint - self.c_mt/2,
            dynamic_midpoint + self.c_tm/2,
            dynamic_midpoint - self.c_tm/2
        ]

    def get_price_with_slippage(self, price_col, next_price_col, timestamp):
        if timestamp not in self.raw_df.index:
            return np.nan
        current_price = self.raw_df.loc[timestamp, price_col]
        if pd.isna(current_price):
            return np.nan
        try:
            next_price = self.raw_df.loc[timestamp, next_price_col]
            if pd.isna(next_price):
                next_price = current_price
        except:
            next_price = current_price
        return self.beta * current_price + (1 - self.beta) * next_price

    def get_future_price(self, price_col, timestamp, dt_steps=1):
        if timestamp not in self.raw_df.index:
            return np.nan
        current_idx = self.raw_df.index.get_loc(timestamp)
        future_idx = min(current_idx + dt_steps, len(self.raw_df) - 1)
        future_timestamp = self.raw_df.index[future_idx]
        return self.raw_df.loc[future_timestamp, price_col]

    def execute_trade(self, strategy, timestamp, params):
        """执行交易（策略1或策略2）"""
        if timestamp not in self.raw_df.index:
            return False
            
        # 参数解包
        tt_open, tt_close, mt_open, mt_close, tm_open, tm_close = params
        
        row = self.raw_df.loc[timestamp]
        if pd.isna(row['basis1_price']) or pd.isna(row['basis2_price']):
            return False

          
        # print(row['spot_bid0_price'], row['spot_ask0_price'], row['swap_bid0_price'], row['swap_ask0_price'])
        # print(row['basis1_price'], row['basis2_price'])    

        # 选择仓位和本金
        if strategy == 1:
            P, p_swap, p_spot, P_swap = self.P1, self.p1_swap, self.p1_spot, self.P1_swap
        else:
            P, p_swap, p_spot, P_swap = self.P2, self.p2_swap, self.p2_spot, self.P2_swap
            
        # 开仓检查（三种模式）
        if (row['basis1_price']*row['spot_ask0_price'] > tt_open and P > 0):
            trade_type = 'A'
        elif (row['basis1_price']*row['spot_ask0_price'] > mt_open and P > 0):
            trade_type = 'B'
        elif (row['basis1_price']*row['spot_ask0_price'] > tm_open and P > 0):
            trade_type = 'C'
        elif (row['basis2_price']*row['spot_bid0_price'] < tt_close and p_swap < 0):
            trade_type = 'close_A'
        elif (row['basis2_price']*row['spot_bid0_price'] < mt_close and p_swap < 0):
            trade_type = 'close_B'
        elif (row['basis2_price']*row['spot_bid0_price'] < tm_close and p_swap < 0):
            trade_type = 'close_C'
        else:
            return False
            
        # 执行交易
        if trade_type.startswith('close'):
            # 关仓逻辑
            if pd.isna(row['spot_bid0_price']) or pd.isna(row['basis2_volume']) or row['spot_bid0_price'] <= 0:
                return False
            z = self.alpha * min(P / row['spot_bid0_price'], row['basis2_volume'])
            z = min(z, -p_swap)
            if z <= 0:
                return False
                
            if trade_type == 'close_A':
                spot_price = self.get_price_with_slippage('spot_bid0_price', 'spot_bid1_price', timestamp)
                swap_price = self.get_price_with_slippage('swap_ask0_price', 'swap_ask1_price', timestamp)
                if pd.isna(spot_price) or pd.isna(swap_price):
                    return False
                new_p_swap = p_swap + z
                new_p_spot = p_spot - z
                new_P_swap = P_swap - swap_price * z
                new_P = P + (spot_price - self.c_t_swap - self.c_t_spot) * z
            elif trade_type == 'close_B':
                future_spot_bid0 = self.get_future_price('spot_bid0_price', timestamp, 1)
                future_spot_bid1 = self.get_future_price('spot_bid1_price', timestamp, 1)
                if pd.isna(future_spot_bid0) or pd.isna(future_spot_bid1):
                    return False
                spot_price = self.beta * future_spot_bid0 + (1 - self.beta) * future_spot_bid1
                new_p_swap = p_swap + z
                new_p_spot = p_spot - z
                new_P_swap = P_swap - row['swap_ask0_price'] * z
                new_P = P + (spot_price - self.c_m_swap - self.c_t_spot) * z
            else:  # close_C
                future_swap_ask0 = self.get_future_price('swap_ask0_price', timestamp, 1)
                future_swap_ask1 = self.get_future_price('swap_ask1_price', timestamp, 1)
                if pd.isna(future_swap_ask0) or pd.isna(future_swap_ask1):
                    return False
                swap_price = self.beta * future_swap_ask0 + (1 - self.beta) * future_swap_ask1
                new_p_swap = p_swap + z
                new_p_spot = p_spot - z
                new_P_swap = P_swap - swap_price * z
                new_P = P + (row['spot_bid0_price'] - self.c_t_swap - self.c_m_spot) * z
        else:
            # 开仓逻辑
            if pd.isna(row['spot_ask0_price']) or pd.isna(row['basis1_volume']) or row['spot_ask0_price'] <= 0:
                return False
            z = self.alpha * min(P / row['spot_ask0_price'], row['basis1_volume'])
            if z <= 0:
                return False
                
            if trade_type == 'A':
                spot_price = self.get_price_with_slippage('spot_ask0_price', 'spot_ask1_price', timestamp)
                swap_price = self.get_price_with_slippage('swap_bid0_price', 'swap_bid1_price', timestamp)
                if pd.isna(spot_price) or pd.isna(swap_price):
                    return False
                new_p_swap = p_swap - z
                new_p_spot = p_spot + z
                new_P_swap = P_swap + swap_price * z
                new_P = P - (spot_price + self.c_t_swap + self.c_t_spot) * z
            elif trade_type == 'B':
                future_spot_ask0 = self.get_future_price('spot_ask0_price', timestamp, 1)
                future_spot_ask1 = self.get_future_price('spot_ask1_price', timestamp, 1)
                if pd.isna(future_spot_ask0) or pd.isna(future_spot_ask1):
                    return False
                spot_price = self.beta * future_spot_ask0 + (1 - self.beta) * future_spot_ask1
                new_p_swap = p_swap - z
                new_p_spot = p_spot + z
                new_P_swap = P_swap + row['swap_bid0_price'] * z
                new_P = P - (spot_price + self.c_m_swap + self.c_t_spot) * z
            else:  # 'C'
                future_swap_bid0 = self.get_future_price('swap_bid0_price', timestamp, 1)
                future_swap_bid1 = self.get_future_price('swap_bid1_price', timestamp, 1)
                if pd.isna(future_swap_bid0) or pd.isna(future_swap_bid1):
                    return False
                swap_price = self.beta * future_swap_bid0 + (1 - self.beta) * future_swap_bid1
                new_p_swap = p_swap - z
                new_p_spot = p_spot + z
                new_P_swap = P_swap + swap_price * z
                new_P = P - (row['spot_ask0_price'] + self.c_t_swap + self.c_m_spot) * z
        
        # 更新仓位
        if strategy == 1:
            self.p1_swap, self.p1_spot, self.P1_swap, self.P1 = new_p_swap, new_p_spot, new_P_swap, new_P
        else:
            self.p2_swap, self.p2_spot, self.P2_swap, self.P2 = new_p_swap, new_p_spot, new_P_swap, new_P
            
        return True

    def calculate_pnl(self, timestamp, strategy):
        if timestamp not in self.raw_df.index:
            return np.nan
        row = self.raw_df.loc[timestamp]
        spot_price = row['spot_bid0_price']
        swap_price = row['swap_ask0_price']
        if pd.isna(spot_price) or pd.isna(swap_price):
            return np.nan
        if strategy == 1:
            pnl = self.P1 + self.P1_swap + self.p1_spot * spot_price + self.p1_swap * swap_price
        else:
            pnl = self.P2 + self.P2_swap + self.p2_spot * spot_price + self.p2_swap * swap_price
        return pnl

    def run_backtest(self):
        self.load_data()
        self.initialize_predictor()
        
        # 按秒分组（100ms数据）
        df_grouped = self.raw_df.groupby(pd.Grouper(freq='1s'))
        last_update_time = None
        
        for second_timestamp, second_group in df_grouped:
            if second_group.empty:
                continue
                
            # 每10分钟更新动态参数
            if (last_update_time is None or 
                second_timestamp - last_update_time >= pd.Timedelta(minutes=10)):
                self.update_dynamic_params(second_timestamp)
                last_update_time = second_timestamp
            
            # 在每个秒区间内，逐100ms检查交易机会
            trade_executed_1 = trade_executed_2 = False
            for timestamp in second_group.index:
                # 策略1：静态参数
                if not trade_executed_1:
                    if self.execute_trade(1, timestamp, self.static_params):
                        trade_executed_1 = True
                
                # 策略2：动态参数
                if not trade_executed_2:
                    if self.execute_trade(2, timestamp, self.current_dynamic_params):
                        trade_executed_2 = True
                
                # 如果两个策略都执行了，可以提前跳出
                if trade_executed_1 and trade_executed_2:
                    break
            
            # 记录秒结束时的PnL
            last_timestamp = second_group.index[-1]
            pnl1 = self.calculate_pnl(last_timestamp, 1)
            pnl2 = self.calculate_pnl(last_timestamp, 2)
            if not pd.isna(pnl1) and not pd.isna(pnl2):
                self.pnl1_history.append(pnl1)
                self.pnl2_history.append(pnl2)
                self.timestamps.append(last_timestamp)

    def save_results(self):
        results_dir = Path("./backtest/data/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存PnL
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'pnl_strategy1': self.pnl1_history,
            'pnl_strategy2': self.pnl2_history
        })
        df.set_index('timestamp', inplace=True)
        df.to_csv(results_dir / f"dual_strategy_pnl_{self.symbol}_{self.start_time[:10]}.csv")
        
        # 保存基差
        basis_df = self.raw_df.loc[self.timestamps][['basis1_price', 'basis2_price']]
        basis_df.to_csv(results_dir / f"dual_strategy_basis_{self.symbol}_{self.start_time[:10]}.csv")
        
        return df, basis_df

    def plot_results(self):
        results_dir = Path("./backtest/data/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载基差数据
        basis_df = self.raw_df.loc[self.timestamps][['basis1_price', 'basis2_price']]
        
        # 创建子图 (3, 1)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # PnL 对比
        ax1.plot(self.timestamps, self.pnl1_history, label='Strategy 1 (Static)', color='blue')
        ax1.plot(self.timestamps, self.pnl2_history, label='Strategy 2 (Dynamic)', color='red')
        ax1.set_ylabel('PnL')
        ax1.set_title(f'Dual Strategy Backtest - {self.symbol}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 基差
        ax2.plot(basis_df.index, basis_df['basis1_price'], label='Basis1', color='orange')
        ax2.plot(basis_df.index, basis_df['basis2_price'], label='Basis2', color='green')
        ax2.set_ylabel('Basis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # PnL 差异
        pnl_diff = np.array(self.pnl2_history) - np.array(self.pnl1_history)
        ax3.plot(self.timestamps, pnl_diff, label='Strategy2 - Strategy1', color='purple')
        ax3.set_ylabel('PnL Difference')
        ax3.set_xlabel('Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / f"dual_strategy_{self.symbol}_{self.start_time[:10]}.png", dpi=150, bbox_inches='tight')
        plt.close()

def main():
    symbol = "XRP"
    start_time = "2025-10-01 00:00:00"
    end_time = "2025-10-07 23:59:59"
    static_params = [0.01, -0.01, 0.008, -0.008, 0.009, -0.009]
    
    backtest = DualStrategyBacktest100ms(symbol, start_time, end_time, static_params)
    backtest.run_backtest()
    backtest.save_results()
    backtest.plot_results()
    
    print(f"Final PnL - Strategy 1: {backtest.pnl1_history[-1]:.2f}")
    print(f"Final PnL - Strategy 2: {backtest.pnl2_history[-1]:.2f}")

if __name__ == "__main__":
    main()