# ./finetune/qlib_data_preprocess.py
import os
import pickle
import pandas as pd
from config import Config

TASK_NAME = "task8"
OUTPUT_NAME = "custom_25d"
def main():
    symbols = ["ADA", "AIXBT", "APT", "AVAX", "BCH", "BNB", "BTC",  # 6
            "CHESS", "COMP", "DOGE", "DOT", "ENA", "ETC","ETH", # 13
            "FET", "FORM", "HBAR", "HFT", "KAITO", "LINK", "LTC", # 20
            "NEAR", "OM", "ONDO", "PNUT", "SOL", "TAO", # 26
            "THE", "TON", "TRX", "TURBO",  # 30
            "UNI", "XLM", "XRP", "ZEC", # 34
            ] # 
    # symbols = ['1000CAT', '1000CHEEMS', '1000SATS', '1MBABYDOGE', 
    #             'AAVE', 'ACH', 'ADA', 'AI', 'AIXBT', 'ALGO', 
    #             'ALICE', 'ALPINE', 'ALT', 'APE', 'API3', 'APT', 
    #             'ARB', 'ARKM', 'ARK', 'ASTR', 'ATOM', 'AUCTION', 
    #             'A', 'AVAX', 'AXL', 
    #             'BANANA', 'BAND', 'BB', 'BCH', 'BIO', 'BMT', "BNB", "BTC", 
    #             'CAKE', 'CELO', 'CETUS', 'CFX', 'CHESS', 'CHZ', 'CKB', 'COMP', 'COTI', 'CRV', 'C', 
    #             'DEXE', 'DIA', 'DOGE', 'DOT', 'DUSK', 'DYDX', 
    #             'EGLD', 'EIGEN', 'ENA', 'ENJ', 'ENS', 'EPIC', 'ETC', 'ETHFI', 'ETH', 
    #             'FET', 'FLUX', 'FORM', 'FXS', 
    #             'GAS', 'GLM', 'GMX', 'GPS', 
    #             'HAEDAL', 'HBAR', 'HIVE', 'HUMA',  "HFT",
    #             'ICP', 'ID', 'ILV', 'INJ', 'IO', 
    #             'JASMY', 'JTO', 'JUP', 'KAITO', 'KAVA', 'KSM', 
    #             'LAYER', 'LDO', 'LINK', 'LPT', 'LQTY', 'LTC', 
    #             'MASK', 'MAV', 'MOVR', 'NEAR', 'NEIRO', 'NEO', 'NXPC', 
    #             'OG', 'OM', 'ONDO', 'OP', 'ORDI', 
    #             'PENDLE', 'PENGU', 'PEOPLE', 'PHA', 'PNUT', 'POL', 'POLYX', 'PROM', 'PROVE', 'PYTH', 
    #             'QNT', 'QTUM', 'RARE', 'RED', 'RENDER', 'RESOLV', 'RONIN', 'RSR', 
    #             'SAGA', 'SAND', 'SANTOS', 'SCRT', 'SCR', 'SFP', 'SOL', 'SOLV', 'SPELL', 
    #             'SUI', 'SUPER', 'S', 'SYRUP', 'SYS', 
    #             'TAO', 'THETA', 'TIA', 'TON', 'TRB', 'TRUMP', 'TRX', 'TST', 'TURBO', 'TWT',  "THE",
    #             'UNI', 'VANA', 'VET', 'VIRTUAL', 'WIF', 'WLD', 'XLM', 'XRP', 'XVG', 
    #             'YFI', 'ZEC', 'ZEN', 'ZK', 'ZRO']
    # config = Config()
    train_val_start = "2025-01-01"
    train_val_end = "2025-09-30"
    test_start = "2025-10-01"
    test_end = "2025-10-29"

    combined_train, combined_val = {}, {}

    print("🔄 Starting preprocessing...")
    print(f"Processing symbols for {TASK_NAME}...")

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
        
        print(f"Processing {sym} with {len(df)} rows, csv path: {csv_path}.")
        print(df.isna().any(axis=None))
        if df.isna().any(axis=None):
            print(f"Warning: {sym} contains NaN values. Filling forward.")
            df = df.fillna(method="ffill").dropna()
        print(f"After NaN handling, {sym} has {len(df)} rows.")
        print(df.isna().any(axis=None))
        # 训练+验证段
        train_val_df = df[(df.index >= train_val_start) & (df.index <= train_val_end)]
        n_total = len(train_val_df)
        n_train = int(n_total * 0.9)
        combined_train[sym] = train_val_df.iloc[:n_train]
        combined_val[sym] = train_val_df.iloc[n_train:]

        # 独立测试集（按 symbol）
        test_df = df[(df.index >= test_start) & (df.index <= test_end)]
        os.makedirs(f"./datasets/{OUTPUT_NAME}/processed_datasets/{sym}", exist_ok=True)
        with open(f"./datasets/{OUTPUT_NAME}/processed_datasets/{sym}/test_data.pkl", 'wb') as f:
            pickle.dump({sym: test_df}, f)

    # 保存合并 train/val
    os.makedirs(f"./datasets/{OUTPUT_NAME}/processed_datasets", exist_ok=True)
    with open(f"./datasets/{OUTPUT_NAME}/processed_datasets/train_data.pkl", 'wb') as f:
        pickle.dump(combined_train, f)
    with open(f"./datasets/{OUTPUT_NAME}/processed_datasets/val_data.pkl", 'wb') as f:
        pickle.dump(combined_val, f)

    print("✅ Preprocessing complete.")

if __name__ == '__main__':
    main()