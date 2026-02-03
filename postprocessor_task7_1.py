import pandas as pd
from pathlib import Path

# 配置
symbols = ['1000CAT', '1000CHEEMS', '1000SATS', '1MBABYDOGE', 
            'AAVE', 'ACH', 'ADA', 'AI', 'AIXBT', 'ALGO', 
            'ALICE', 'ALPINE', 'ALT', 'APE', 'API3', 'APT', 
            'ARB', 'ARKM', 'ARK', 'ASTR', 'ATOM', 'AUCTION', 
            'A', 'AVAX', 'AXL', 
            'BANANA', 'BAND', 'BB', 'BCH', 'BIO', 'BMT', "BNB", "BTC", 
            'CAKE', 'CELO', 'CETUS', 'CFX', 'CHESS', 'CHZ', 'CKB', 'COMP', 'COTI', 'CRV', 'C', 
            'DEXE', 'DIA', 'DOGE', 'DOT', 'DUSK', 'DYDX', 
            'EGLD', 'EIGEN', 'ENA', 'ENJ', 'ENS', 'EPIC', 'ETC', 'ETHFI', 'ETH', 
            'FET', 'FLUX', 'FORM', 'FXS', 
            'GAS', 'GLM', 'GMX', 'GPS', 
            'HAEDAL', 'HBAR', 'HIVE', 'HUMA',  "HFT",
            'ICP', 'ID', 'ILV', 'INJ', 'IO', 
            'JASMY', 'JTO', 'JUP', 'KAITO', 'KAVA', 'KSM', 
            'LAYER', 'LDO', 'LINK', 'LPT', 'LQTY', 'LTC', 
            'MASK', 'MAV', 'MOVR', 'NEAR', 'NEIRO', 'NEO', 'NXPC', 
            'OG', 'OM', 'ONDO', 'OP', 'ORDI', 
            'PENDLE', 'PENGU', 'PEOPLE', 'PHA', 'PNUT', 'POL', 'POLYX', 'PROM', 'PROVE', 'PYTH', 
            'QNT', 'QTUM', 'RARE', 'RED', 'RENDER', 'RESOLV', 'RONIN', 'RSR', 
            'SAGA', 'SAND', 'SANTOS', 'SCRT', 'SCR', 'SFP', 'SOL', 'SOLV', 'SPELL', 
            'SUI', 'SUPER', 'S', 'SYRUP', 'SYS', 
            'TAO', 'THETA', 'TIA', 'TON', 'TRB', 'TRUMP', 'TRX', 'TST', 'TURBO', 'TWT',  "THE",
            'UNI', 'VANA', 'VET', 'VIRTUAL', 'WIF', 'WLD', 'XLM', 'XRP', 'XVG', 
            'YFI', 'ZEC', 'ZEN', 'ZK', 'ZRO']
for i in range(len(symbols)):
    symbol = symbols[i]
    quote = "USDT"
    pair = f"{symbol}{quote}"
    processed_dir, output_dir = Path("datasets/processed/basis_1min_task7"), Path("batch/data/task7")
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
        df = pd.read_csv(f, compression='gzip')  # assuming .csv.gz
        df['timestamps'] = pd.to_datetime(df['timestamps'], format='%m/%d/%Y %I:%M:%S %p')
        df.set_index('timestamps', inplace=True)
        # df = pd.read_csv(f, parse_dates=["timestamps"], index_col="timestamps")
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, axis=0).sort_index()
    print(f"Total rows: {len(df_all)}")

    # 检查必要列是否存在

    print(df_all.info())
    
    # # 生成 Kronos OHLCV 字段
    df_kronos = pd.DataFrame(index=df_all.index)
    if "Open" in df_all.columns:
        df_kronos["open"] = df_all["Open"]
        df_kronos["high"] = df_all["Max"]
        df_kronos["low"]  = df_all["Min"]
        df_kronos["close"] = df_all["Close"]
        df_kronos["volume"] = df_all["Volume"]
        df_kronos["amount"] = df_all["Amount"]
    else:
        df_kronos["open"] = df_all["open"]
        df_kronos["high"] = df_all["high"]
        df_kronos["low"]  = df_all["low"]
        df_kronos["close"] = df_all["close"]
        df_kronos["volume"] = df_all["volume"]
        df_kronos["amount"] = df_all["amount"]

    # 可选：移除全 NaN 行（如 funding_rate 初始缺失）
    df_kronos = df_kronos.dropna(how="all")
    df_kronos = df_kronos.fillna(method="ffill")
    # 可选：裁剪极端值
    df_kronos = df_kronos.clip(-10,10)
    df_kronos = df_kronos.dropna()
    # df_kronos.index.names = ["timestamps"]
    # 保存为 Kronos 格式
    output_file = output_dir / f"{pair}_task7.csv"
    # if "timestamp" in df_kronos.columns:
    #     df_kronos = df_kronos.rename(columns={"timestamp": "timestamps"})
    print(df_kronos.info())
    print(f"Saving Kronos dataset: {output_file}")
    # df_kronos.to_csv(output_file, compression="gzip", date_format="%Y-%m-%d %H:%M:%S")
    df_kronos.to_csv(output_file,  date_format="%Y-%m-%d %H:%M:%S")
    print("✅ Done.")