# ./finetune/qlib_data_preprocess.py
import os
import pickle
import pandas as pd
from config import Config

TASK_NAME = "task4"

def main():
    symbols = ["SOL", "BNB", "ZEC", "KAITO", "DOT", "ETH", "BTC", "LTC", "XRP", "ADA", "DOGE", "AVAX", "ETC", "TAO", # 13
            "CHESS", "COMP", "LINK", "TON", "AIXBT", "BCH", "ETH", "FET", "OM", "ONDO"] # 23
    # config = Config()
    train_val_start = "2025-01-01"
    train_val_end = "2025-09-30"
    test_start = "2025-10-01"
    test_end = "2025-10-29"

    combined_train, combined_val = {}, {}

    for sym in symbols:
        csv_path = f"batch/data/{TASK_NAME}/{sym}USDT_{TASK_NAME}.csv"
        try:
            df = pd.read_csv(csv_path)
            df['datetime'] = pd.to_datetime(df['timestamps'])
            df = df.set_index('datetime').sort_index()
            # df = df.rename(columns={'volume': 'vol', 'amount': 'amt'})[config.feature_list]
        except Exception as e:
            print(f"Skip {sym}: {e}")
            continue

        # 训练+验证段
        train_val_df = df[(df.index >= train_val_start) & (df.index <= train_val_end)]
        n_total = len(train_val_df)
        n_train = int(n_total * 0.9)
        combined_train[sym] = train_val_df.iloc[:n_train]
        combined_val[sym] = train_val_df.iloc[n_train:]

        # 独立测试集（按 symbol）
        test_df = df[(df.index >= test_start) & (df.index <= test_end)]
        os.makedirs(f"./datasets/{TASK_NAME}/processed_datasets/{sym}", exist_ok=True)
        with open(f"./datasets/{TASK_NAME}/processed_datasets/{sym}/test_data.pkl", 'wb') as f:
            pickle.dump({sym: test_df}, f)

    # 保存合并 train/val
    os.makedirs(f"./datasets/{TASK_NAME}/processed_datasets", exist_ok=True)
    with open(f"./datasets/{TASK_NAME}/processed_datasets/train_data.pkl", 'wb') as f:
        pickle.dump(combined_train, f)
    with open(f"./datasets/{TASK_NAME}/processed_datasets/val_data.pkl", 'wb') as f:
        pickle.dump(combined_val, f)

    print("✅ Preprocessing complete.")

if __name__ == '__main__':
    main()