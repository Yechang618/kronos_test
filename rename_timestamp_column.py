# rename_timestamp_column.py
import os
import pandas as pd

# input_path = "datasets/kronos/BNBUSDT_kronos.csv"
# input_path = "datasets/kronos/SOLUSDT_kronos.csv"
# input_path = "batch/data/BNBUSDT_kronos.csv"
# input_path = "batch/data/KAITOUSDT_kronos.csv"
# input_path = "batch/data/DOTUSDT_kronos.csv"
# input_path = "batch/data/ZECUSDT_kronos.csv"
# input_path = "batch/data/SOLUSDT_kronos.csv"
input_path = "batch/data/SOLUSDT_kronos.csv"

df = pd.read_csv(input_path)

# Rename column if needed
if "timestamp" in df.columns:
    df = df.rename(columns={"timestamp": "timestamps"})
elif "timestamps" not in df.columns:
    raise KeyError("CSV must contain 'timestamp' or 'timestamps' column")

# Parse `timestamps` into datetimes (try several common formats/units)
ts = df['timestamps']
ts_dt = pd.to_datetime(ts, errors='coerce', infer_datetime_format=True)
if ts_dt.isna().all():
    # try numeric seconds since epoch
    ts_num = pd.to_numeric(ts, errors='coerce')
    ts_dt = pd.to_datetime(ts_num, unit='s', errors='coerce')

if ts_dt.isna().all():
    # try milliseconds since epoch
    ts_num = pd.to_numeric(ts, errors='coerce')
    ts_dt = pd.to_datetime(ts_num, unit='ms', errors='coerce')

if ts_dt.isna().all():
    raise ValueError("Unable to parse 'timestamps' column to datetimes")

# Select a one-day sample: from the earliest calendar day present
start = ts_dt.min().normalize()
end = start + pd.Timedelta(days=1)
mask = (ts_dt >= start) & (ts_dt < end)
sample_df = df[mask].copy()

if sample_df.empty:
    # Fallback: build a contiguous block of rows covering approximately one day
    # We'll use sorted timestamps and accumulate until >= 1 day, else take first 1440 rows
    sorted_idx = ts_dt.sort_values().index
    if len(sorted_idx) <= 1:
        nrows = min(len(df), 1440)
        sample_df = df.iloc[:nrows].copy()
    else:
        acc = pd.Timedelta(0)
        prev = ts_dt.loc[sorted_idx[0]]
        selected_idx = [sorted_idx[0]]
        for idx in sorted_idx[1:]:
            curr = ts_dt.loc[idx]
            acc += (curr - prev)
            prev = curr
            selected_idx.append(idx)
            if acc >= pd.Timedelta(days=1):
                break
        if not selected_idx:
            nrows = min(len(df), 1440)
            sample_df = df.iloc[:nrows].copy()
        else:
            sample_df = df.loc[selected_idx].copy()

# Save sample as *_sample.csv alongside original
base, ext = os.path.splitext(input_path)
output_path = f"{base}_sample{ext}"
sample_df.to_csv(output_path, index=False)
print(f"✅ Saved one-day sample to '{output_path}'")