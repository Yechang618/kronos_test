# rename_timestamp_column.py
import pandas as pd

# input_path = "datasets/kronos/BNBUSDT_kronos.csv"
# input_path = "datasets/kronos/SOLUSDT_kronos.csv"
# input_path = "batch/data/BNBUSDT_kronos.csv"
# input_path = "batch/data/KAITOUSDT_kronos.csv"
# input_path = "batch/data/DOTUSDT_kronos.csv"
input_path = "batch/data/ZECUSDT_kronos.csv"
# input_path = "batch/data/SOLUSDT_kronos.csv"
df = pd.read_csv(input_path)

# 重命名列
if "timestamp" in df.columns:
    df = df.rename(columns={"timestamp": "timestamps"})
elif "timestamps" not in df.columns:
    raise KeyError("CSV must contain 'timestamp' or 'timestamps' column")

# 保存（覆盖或另存）
df.to_csv(input_path, index=False)
print("✅ Renamed 'timestamp' -> 'timestamps'")