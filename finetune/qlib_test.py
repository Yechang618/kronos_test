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
from config import Config
from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class QlibTestDataset(Dataset):
    def __init__(self, data: dict, config: Config):
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
                timestamp = df.iloc[i + config.lookback_window - 1]['datetime']
                self.indices.append((symbol, i, timestamp))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
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

def generate_predictions_for_symbol(symbol, run_config):
    test_data_path = f"./data/processed_datasets/{symbol}/test_data.pkl"
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)

    tokenizer = KronosTokenizer.from_pretrained(run_config['tokenizer_path'])
    model = Kronos.from_pretrained(run_config['model_path'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer.to(device).eval()
    model.to(device).eval()

    dataset = QlibTestDataset(test_data, Config())
    loader = DataLoader(dataset, batch_size=run_config['batch_size'] // run_config['sample_count'], shuffle=False, num_workers=4)

    results = defaultdict(list)
    with torch.no_grad():
        for x, x_stamp, y_stamp, symbols, timestamps in loader:
            preds = auto_regressive_inference(
                tokenizer, model, x.to(device), x_stamp.to(device), y_stamp.to(device),
                max_context=run_config['max_context'], pred_len=run_config['pred_len'], clip=run_config['clip'],
                T=run_config['T'], top_k=run_config['top_k'], top_p=run_config['top_p'], sample_count=run_config['sample_count']
            )
            preds = preds[:, -run_config['pred_len']:, :]
            last_day_close = x[:, -1, 3].numpy()
            signals = {
                'last': preds[:, -1, 3] - last_day_close,
                'mean': np.mean(preds[:, :, 3], axis=1) - last_day_close,
            }
            for i in range(len(symbols)):
                for sig_type, sig_vals in signals.items():
                    results[sig_type].append((timestamps[i], symbols[i], sig_vals[i]))

    return results

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

    prediction_dfs = {}
    for sig_type, records in all_signals.items():
        df = pd.DataFrame(records, columns=['datetime', 'instrument', 'score'])
        pivot = df.pivot(index='datetime', columns='instrument', values='score')
        prediction_dfs[sig_type] = pivot.sort_index()

    save_dir = os.path.join(config.backtest_result_path, config.backtest_save_folder_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "predictions.pkl"), 'wb') as f:
        pickle.dump(prediction_dfs, f)

    print("✅ All done. Predictions saved.")

if __name__ == '__main__':
    main()