import pandas as pd
from pathlib import Path

# 配置
# symbols = ["SOL", "BNB", "ZEC", "KAITO", "DOT", "ETH", "BTC", "LTC", "XRP", "ADA", "DOGE", "AVAX", "ETC", "TAO", # 13
#            "CHESS", "COMP", "LINK", "TON", "AIXBT", "BCH", "ETH", "FET", "OM", "ONDO"]
symbols = ["ADA", "AIXBT", "APT", "AVAX", "BCH", "BNB", "BTC",  # 6
           "CHESS", "COMP", "DOGE", "DOT", "ENA", "ETC","ETH", # 13
           "FET", "FORM", "HBAR", "HFT", "KAITO", "LINK", "LTC", # 20
           "NEAR", "OM", "ONDO", "PNUT", "SOL", "TAO", # 26
           "THE", "TON", "TRX", "TURBO",  # 30
           "UNI", "XLM", "XRP", "ZEC", # 34
           ] # 
for i in range(len(symbols)):
    symbol = symbols[i]
    quote = "USDT"
    pair = f"{symbol}{quote}"
    processed_dir, output_dir = Path("datasets/processed/basis_10min_task5"), Path("batch/data/task5")
    output_dir.mkdir(parents=True, exist_ok=True)
    # 收集所有 processed 文件
    files = list(processed_dir.glob(f"{pair}_*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No processed files found for {pair} in {processed_dir}")

    print(f"Found {len(files)} processed file(s).")

    # 按时间顺序合并所有数据
    all_dfs = []
    for f in sorted(files):
        print(f"Loading {f.name}...")
        df = pd.read_csv(f, parse_dates=["timestamps"], index_col="timestamps")
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, axis=0).sort_index()
    print(f"Total rows: {len(df_all)}")

    # 检查必要列是否存在

    print(df_all.info())
    # missing = [c for c in required_cols if c not in df_all.columns]
    # if missing:
    #     raise KeyError(f"Missing required columns: {missing}")

    # # 生成 Kronos OHLCV 字段
    df_kronos = pd.DataFrame(index=df_all.index)
    df_kronos["open"] = df_all["Open"]
    df_kronos["high"] = df_all["Max"]
    df_kronos["low"]  = df_all["Min"]
    df_kronos["close"] = df_all["Close"]
    df_kronos["volume"] = df_all["Volume"]
    df_kronos["amount"] = df_all["Amount"]

    # 可选：移除全 NaN 行（如 funding_rate 初始缺失）
    df_kronos = df_kronos.dropna(how="all")
    df_kronos = df_kronos.clip(-10,10)
    # df_kronos.index.names = ["timestamps"]
    # 保存为 Kronos 格式
    output_file = output_dir / f"{pair}_task5.csv"
    # if "timestamp" in df_kronos.columns:
    #     df_kronos = df_kronos.rename(columns={"timestamp": "timestamps"})
    print(df_kronos.info())
    print(f"Saving Kronos dataset: {output_file}")
    # df_kronos.to_csv(output_file, compression="gzip", date_format="%Y-%m-%d %H:%M:%S")
    df_kronos.to_csv(output_file,  date_format="%Y-%m-%d %H:%M:%S")
    print("✅ Done.")