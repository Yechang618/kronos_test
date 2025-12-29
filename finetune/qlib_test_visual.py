# finetune/qlib_test_visual.py
import os
import sys
import pickle
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Add project root to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from config import Config
from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference

# =================================================================================
# 1. 自定义 Collate 函数（处理 pandas.Timestamp）
# =================================================================================
def rolling_collate_fn(batch):
    x_list, x_stamp_list, y_stamp_list, current_t_list, future_ts_list = zip(*batch)
    x_batch = torch.stack(x_list, dim=0)
    x_stamp_batch = torch.stack(x_stamp_list, dim=0)
    y_stamp_batch = torch.stack(y_stamp_list, dim=0)
    current_t_str = [str(t) for t in current_t_list]
    future_ts_batch = np.stack(future_ts_list, axis=0)
    return x_batch, x_stamp_batch, y_stamp_batch, current_t_str, future_ts_batch

# =================================================================================
# 2. 滚动测试数据集（使用 iloc 精确索引）
# =================================================================================
class RollingTestDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        lookback_window: int,
        predict_window: int,
        start_time: str,
        end_time: str,
        step_minutes: int = 30
    ):
        self.df = df.sort_index()
        self.lookback = lookback_window
        self.pred_horizon = predict_window
        self.step = pd.Timedelta(minutes=step_minutes)
        config = Config()
        self.feature_list = config.feature_list  # ['open', 'high', 'low', 'close', 'vol', 'amt']
        self.time_features = ['minute', 'hour', 'weekday', 'day', 'month']

        start_ts = pd.Timestamp(start_time)
        end_ts = pd.Timestamp(end_time)

        # 使用 get_indexer（兼容旧版 pandas）
        start_idx = self.df.index.get_indexer([start_ts], method='nearest')[0]
        end_idx = self.df.index.get_indexer([end_ts], method='nearest')[0]
        print(f"Dataset indices: start_idx={start_idx}, end_idx={end_idx}")

        self.timestamps = []
        current_idx = start_idx

        min_required = self.lookback + self.pred_horizon
        while current_idx + self.pred_horizon <= len(self.df) and current_idx <= end_idx:
            t = self.df.index[current_idx]
            if t < start_ts:
                current_idx += 1
                continue
            if t > end_ts:
                break

            input_start_idx = current_idx - self.lookback + 1
            if input_start_idx < 0:
                current_idx += 1
                continue

            self.timestamps.append(t)
            # 步长：30 分钟 → 30 个索引位置（因数据是 1 分钟粒度）
            current_idx += 30

        print(f"Rolling windows: {len(self.timestamps)} from {start_time} to {end_time}")

    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.Timestamp, np.ndarray]:
        t = self.timestamps[idx]
        try:
            current_idx = self.df.index.get_loc(t)
        except KeyError:
            current_idx = self.df.index.get_indexer([t], method='nearest')[0]

        # 输入窗口: [current_idx - 239, current_idx] -> 240 points
        input_start = current_idx - self.lookback + 1
        input_end = current_idx + 1
        # 预测窗口: [current_idx + 1, current_idx + 30] -> 30 points
        pred_start = current_idx + 1
        # pred_end = pred_start + 1
        pred_end = current_idx + 1 + self.pred_horizon

        if input_start < 0 or pred_end > len(self.df):
            raise ValueError("Window out of bounds")

        input_df = self.df.iloc[input_start:input_end]
        pred_df = self.df.iloc[pred_start:pred_end]

        x = input_df[self.feature_list].values.astype(np.float32)
        x_stamp = input_df[self.time_features].values.astype(np.float32)
        y_stamp = pred_df[self.time_features].values.astype(np.float32)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -5.0, 5.0)

        return (
            torch.from_numpy(x),
            torch.from_numpy(x_stamp),
            torch.from_numpy(y_stamp),
            t,
            pred_df.index.values
        )

# =================================================================================
# 3. 预测函数
# =================================================================================
def predict_with_uncertainty(
    tokenizer,
    model,
    x: torch.Tensor,
    x_stamp: torch.Tensor,
    y_stamp: torch.Tensor,
    device: torch.device,
    sample_count: int = 1,
    **kwargs
) -> np.ndarray:
    x = x.unsqueeze(0).to(device)
    x_stamp = x_stamp.unsqueeze(0).to(device)
    y_stamp = y_stamp.unsqueeze(0).to(device)

    # 显式指定 pred_len=30，不依赖 kwargs
    preds = auto_regressive_inference(
        tokenizer, model, x, x_stamp, y_stamp,
        pred_len=30,  # ← 关键修复
        sample_count=1,
        max_context=kwargs.get('max_context', 2048),
        clip=kwargs.get('clip', 5.0),
        T=kwargs.get('T', 1.0),
        top_p=kwargs.get('top_p', 0.99),
        top_k=kwargs.get('top_k', 0)
    )
    return preds[-len(y_stamp):, 0, :]  # (pred_horizon, feature_dim)

# =================================================================================
# 4. 主函数
# =================================================================================
def main():
    symbols = ["SOL", "BTC", "ETH"]  # 可替换为全部 14 个 symbol
    config = Config()

    # ===== 时间段设置 =====
    start_time = "2025-10-01 04:00:00"
    end_time = "2025-10-01 05:00:00"
    lookback = 240
    pred_horizon = 30
    pred_horizon_1 = 30
    n_realiz = 13

    # ===== 验证测试数据范围 =====
    test_data_path = f"./data/processed_datasets/SOL/test_data.pkl"
    with open(test_data_path, 'rb') as f:
        data = pickle.load(f)
    df_debug = data["SOL"]
    print(f"✅ Test data time range: {df_debug.index.min()} ~ {df_debug.index.max()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = KronosTokenizer.from_pretrained(
        "./outputs/models/finetune_tokenizer_all/checkpoints/best_model"
    ).to(device).eval()
    model = Kronos.from_pretrained(
        "./outputs/models/finetune_predictor_all/checkpoints/best_model"
    ).to(device).eval()

    os.makedirs("figures/rolling_pred", exist_ok=True)

    for j, sym in enumerate(symbols):
        print(f"Processing {sym}...")
        test_data_path = f"./data/processed_datasets/{sym}/test_data.pkl"
        if not os.path.exists(test_data_path):
            print(f"Skipping {sym}: test data not found")
            continue

        with open(test_data_path, 'rb') as f:
            data = pickle.load(f)
        df = data[sym].copy()
        df['minute'] = df.index.minute
        df['hour'] = df.index.hour
        df['weekday'] = df.index.weekday
        df['day'] = df.index.day
        df['month'] = df.index.month

        dataset = RollingTestDataset(
            df=df,
            lookback_window=lookback,
            predict_window=pred_horizon,
            start_time=start_time,
            end_time=end_time,
            step_minutes=30
        )

        if len(dataset) == 0:
            print(f"No valid windows for {sym}")
            continue

        all_true = []
        all_pred_mean = []
        all_pred_std = []
        all_pred_times = []

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=rolling_collate_fn
        )

        for x, x_stamp, y_stamp, current_t, future_ts in loader:
            # print(f"y_stamp: {y_stamp}")  # Debug 信息
            # print(f"y_stamp shape: {y_stamp.shape}")  # Debug 信息
            x, x_stamp, y_stamp = x[0], x_stamp[0], y_stamp[0]
            future_ts = future_ts[0]  # (30,)
            # print(f"y_stamp shape: {y_stamp.shape} after squeezing")  # Debug 信息
            # 真实值
            # try:
            #     true_close = df.loc[future_ts, 'close'].values  # (30,)
            # except KeyError:
            #     continue
            # all_true.append(true_close)
            all_true.append(df.loc[future_ts].values)  # Close price index is 3
            preds = np.zeros((n_realiz, pred_horizon_1, x.shape[1]), dtype=np.float32)
            # 预测
            for n in range(n_realiz):
                preds[n] = predict_with_uncertainty(
                    tokenizer, model, x, x_stamp, y_stamp, device,
                    max_context=2048,
                    pred_len=pred_horizon_1,
                    clip=5.0,
                    T=0.6,
                    top_p=0.9,
                    top_k=0
                )
            # preds shape: (sample_count, pred_horizon, feature_dim)
            print(f"Preds shape: {preds.shape}") # Preds shape: (n_realiz, {pred_horizon_1}, 6)
            pred_mean = preds.mean(axis=0)  # (pred_horizon_1,6)
            pred_std = preds.std(axis=0)    # (pred_horizon_1,6)

            print(f"Time {current_t[0]} - True: {len(all_true)}, Pred Mean: {pred_mean.shape}, Pred Std: {pred_std.shape}")
            all_pred_mean.append(pred_mean)
            all_pred_std.append(pred_std)
            all_pred_times.append(future_ts)

        print(f"all_true length: {len(all_true)}, all_pred_mean length: {len(all_pred_mean)}, all_pred_std length: {len(all_pred_std)}, all_pred_times length: {len(all_pred_times)}")

        if not all_true:
            continue

        true_full = all_true[0]    # (6 * 30 = 180,)
        pred_mean_full = all_pred_mean[0]  # (180,)
        pred_std_full = all_pred_std[0]    # (180,)
        time_full = all_pred_times[0]      # (180,)

        # 最终维度检查
        print(f"Final lengths for {sym} - time: {len(time_full)}, true: {len(true_full)}, pred_mean: {len(pred_mean_full)}, pred_std: {len(pred_std_full)}")
        assert len(time_full) == len(pred_mean_full), f"time {len(time_full)} vs pred {len(pred_mean_full)}"

        # 可视化
        feature_list = ['Open', 'High', 'Low', 'Close', 'Vol', 'Amt']
        for i in range(6):
            plt.figure(figsize=(12, 6))
            plt.plot(time_full, true_full[:, i], label="Ground Truth", color="black", linewidth=2)
            plt.plot(time_full, pred_mean_full[:, i], label="Prediction Mean", color="red", linewidth=1.5)
            print(f"{time_full.shape}, {pred_std_full.shape}, {pred_mean_full.shape}")
            plt.fill_between(
                time_full,
                pred_mean_full[:, i] - pred_std_full[:, i],
                pred_mean_full[:, i] + pred_std_full[:, i],
                color="orange",
                alpha=0.3,
                label="±1 Std"
            )
            plt.title(f"{sym} Rolling Prediction ({start_time[:10]} {start_time[11:16]}-{end_time[11:16]})")
            plt.xlabel("Time")
            plt.ylabel(f"{feature_list[i]} Price")
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"figures/rolling_pred/{sym}_rolling_pred_{feature_list[i]}.png", dpi=200)
            plt.close()

    print("✅ All figures saved in figures/rolling_pred/")

if __name__ == '__main__':
    main()