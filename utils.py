# utils.py (or inline)
import pandas as pd
from datetime import datetime, timezone
import requests
import time
import re

def parse_time_str(series):
    """统一解析 time_str 为 UTC datetime"""
    return pd.to_datetime(series, utc=True)

def clean_columns(df, stream_type):
    """删除不需要的列，并为保留列加前缀"""
    drop_cols = {
        "update_id", "bid_px", "bid_qty", "ask_px", "ask_qty",
        "lag_ms", "ts_from_last_ms", "event_ts", "transaction_ts", "local_ts"
    }
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # 加前缀
    prefix = "spot_" if stream_type == "spot_l5" else "swap_"
    rename_map = {col: f"{prefix}{col}" for col in df.columns if col not in ["time_str", "symbol"]}
    df = df.rename(columns=rename_map)
    return df[["time_str"] + [c for c in df.columns if c != "time_str"]]