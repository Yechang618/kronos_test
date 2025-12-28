# ./finetune/qlib_test.py
import os
import sys

# Add project root to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import pickle
import pandas as pd
from collections import defaultdict
from typing import Any, Dict, List, Tuple
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Add project root to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from config import Config
from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference

# =================================================================================
# 1. 自定义 Collate 函数（关键修复）
# =================================================================================
def collate_fn_for_inference(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[Any]]:
    """
    自定义 collate 函数，避免对 pandas.Timestamp 调用 torch.stack()。
    """
    x_list, x_stamp_list, y_stamp_list, symbols, timestamps = zip(*batch)
    x_batch = torch.stack(x_list, dim=0)
    x_stamp_batch = torch.stack(x_stamp_list, dim=0)
    y_stamp_batch = torch.stack(y_stamp_list, dim=0)
    return x_batch, x_stamp_batch, y_stamp_batch, list(symbols), list(timestamps)

# =================================================================================
# 2. 测试数据集
# =================================================================================
class QlibTestDataset(Dataset):
    def __init__(self, data: Dict[str, pd.DataFrame], config: Config):
        self.data = data
        self.config = config
        self.window_size = config.lookback_window + config.predict_window
        self.symbols = list(data.keys())
        self.feature_list = config.feature_list
        self.time_feature_list = config.time_feature_list
        self.indices = []

        for symbol in self.symbols:
            df = data[symbol].reset_index()
            df['minute'] = df['datetime'].dt.minute
            df['hour'] = df['datetime'].dt.hour
            df['weekday'] = df['datetime'].dt.weekday
            df['day'] = df['datetime'].dt.day
            df['month'] = df['datetime'].dt.month
            self.data[symbol] = df
            num_samples = len(df) - self.window_size + 1
            for i in range(num_samples):
                # 转为 Python datetime（可选，但更安全）
                timestamp = df.iloc[i + config.lookback_window - 1]['datetime'].to_pydatetime()
                self.indices.append((symbol, i, timestamp))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        symbol, start_idx, timestamp = self.indices[idx]
        df = self.data[symbol]
        context_end = start_idx + self.config.lookback_window
        predict_end = context_end + self.config.predict_window
        context_df = df.iloc[start_idx:context_end]
        predict_df = df.iloc[context_end:predict_end]

        x = context_df[self.feature_list].values.astype(np.float32)
        x_stamp = context_df[self.time_feature_list].values.astype(np.float32)
        y_stamp = predict_df[self.time_feature_list].values.astype(np.float32)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        return torch.from_numpy(x), torch.from_numpy(x_stamp), torch.from_numpy(y_stamp), symbol, timestamp

# =================================================================================
# 3. 推理函数
# =================================================================================
def generate_predictions_for_symbol(symbol: str, run_config: dict) -> dict:
    test_data_path = f"./data/processed_datasets/{symbol}/test_data.pkl"
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)

    device = torch.device(run_config['device'])
    tokenizer = KronosTokenizer.from_pretrained(run_config['tokenizer_path']).to(device).eval()
    model = Kronos.from_pretrained(run_config['model_path']).to(device).eval()

    dataset = QlibTestDataset(test_data, Config())
    loader = DataLoader(
        dataset,
        batch_size=run_config['batch_size'] // run_config['sample_count'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_for_inference
    )

    results = defaultdict(list)
    with torch.no_grad():
        for x, x_stamp, y_stamp, symbols, timestamps in loader:
            preds = auto_regressive_inference(
                tokenizer,
                model,
                x.to(device),
                x_stamp.to(device),
                y_stamp.to(device),
                max_context=run_config['max_context'],
                pred_len=run_config['pred_len'],
                clip=run_config['clip'],
                T=run_config['T'],
                top_k=run_config['top_k'],
                top_p=run_config['top_p'],
                sample_count=run_config['sample_count']
            )
            # preds is already a numpy.ndarray (from auto_regressive_inference)
            preds = preds[:, -run_config['pred_len']:, :]  # (B, H, D)

            # last_day_close is numpy from x[:, -1, 3].numpy()
            last_day_close = x[:, -1, 3].cpu().numpy()  # ← 这里已经是 numpy

            signals = {
                'last': preds[:, -1, 3] - last_day_close,          # ✅ numpy - numpy
                'mean': preds[:, :, 3].mean(axis=1) - last_day_close,
            }
            for i in range(len(symbols)):
                for sig_type, sig_vals in signals.items():
                    results[sig_type].append((timestamps[i], symbols[i], sig_vals[i]))
    return results

# =================================================================================
# 4. 主函数
# =================================================================================
def main():
    symbols = ["SOL", "BNB", "ZEC", "KAITO", "DOT", "ETH", "BTC", "LTC", "XRP", "ADA", "DOGE", "AVAX", "ETC", "TAO"]
    config = Config()

    run_config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'tokenizer_path': "./outputs/models/finetune_tokenizer_all/checkpoints/best_model",
        'model_path': "./outputs/models/finetune_predictor_all/checkpoints/best_model",
        'max_context': config.max_context,
        'pred_len': config.predict_window,
        'clip': config.clip,
        'T': 0.6,
        'top_k': 0,
        'top_p': 0.9,
        'sample_count': 5,
        'batch_size': 1000,
    }

    all_signals = defaultdict(list)
    for sym in symbols:
        print(f"Running inference for {sym}...")
        sym_results = generate_predictions_for_symbol(sym, run_config)
        for sig_type, records in sym_results.items():
            all_signals[sig_type].extend(records)

    # 转为 DataFrame
    prediction_dfs = {}
    for sig_type, records in all_signals.items():
        df = pd.DataFrame(records, columns=['datetime', 'instrument', 'score'])
        pivot = df.pivot(index='datetime', columns='instrument', values='score')
        prediction_dfs[sig_type] = pivot.sort_index()

    # 保存
    save_dir = os.path.join(config.backtest_result_path, "task3_multisymbol_backtest")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "predictions.pkl"), 'wb') as f:
        pickle.dump(prediction_dfs, f)

    print("✅ Inference complete. Predictions saved.")

if __name__ == '__main__':
    main()

