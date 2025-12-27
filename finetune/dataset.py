# ./finetune/dataset.py
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from config import Config
from typing import Tuple 

class QlibDataset(Dataset):
    def __init__(self, data_type: str = 'train'):
        self.config = Config()
        if data_type not in ['train', 'val']:
            raise ValueError("data_type must be 'train' or 'val'")
        self.data_type = data_type
        self.py_rng = random.Random(self.config.seed)

        if data_type == 'train':
            self.data_path = f"{self.config.dataset_path}/train_data.pkl"
            self.n_samples = self.config.n_train_iter
        else:
            self.data_path = f"{self.config.dataset_path}/val_data.pkl"
            self.n_samples = self.config.n_val_iter

        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

        self.window = self.config.lookback_window + self.config.predict_window + 1
        self.symbols = list(self.data.keys())
        self.feature_list = self.config.feature_list
        self.time_feature_list = self.config.time_feature_list

        self.indices = []
        print(f"[{data_type.upper()}] Pre-computing sample indices...")
        for symbol in self.symbols:
            df = self.data[symbol].reset_index()
            series_len = len(df)
            num_samples = series_len - self.window + 1
            if num_samples > 0:
                df['minute'] = df['datetime'].dt.minute
                df['hour'] = df['datetime'].dt.hour
                df['weekday'] = df['datetime'].dt.weekday
                df['day'] = df['datetime'].dt.day
                df['month'] = df['datetime'].dt.month
                self.data[symbol] = df[self.feature_list + self.time_feature_list]
                for i in range(num_samples):
                    self.indices.append((symbol, i))

        self.n_samples = min(self.n_samples, len(self.indices))
        print(f"[{data_type.upper()}] Found {len(self.indices)} samples. Using {self.n_samples} per epoch.")

    def set_epoch_seed(self, epoch: int):
        epoch_seed = self.config.seed + epoch
        self.py_rng.seed(epoch_seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: 
        random_idx = self.py_rng.randint(0, len(self.indices) - 1)
        symbol, start_idx = self.indices[random_idx]
        df = self.data[symbol]
        end_idx = start_idx + self.window
        win_df = df.iloc[start_idx:end_idx]

        x = win_df[self.feature_list].values.astype(np.float32)
        x_stamp = win_df[self.time_feature_list].values.astype(np.float32)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        return torch.from_numpy(x), torch.from_numpy(x_stamp)