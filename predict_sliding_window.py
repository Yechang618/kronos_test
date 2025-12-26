import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Fix OMP warning
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
sys.path.append(str(Path(__file__).parent))

from model import Kronos, KronosTokenizer, KronosPredictor

# ==============================
# 配置
# ==============================
BASE_DIR = Path("trained/sol_1min_10s")
TOKENIZER_PATH = BASE_DIR / "tokenizer" / "best_model"
BASEMODEL_PATH = BASE_DIR / "basemodel" / "best_model"

TEST_DATA_PATH = "batch/kronos_test/SOLUSDT_kronos.csv"

N_SAMPLES = 30
LOOKBACK_WINDOW = 60    # 60 分钟上下文（1 分钟 K 线）
PRED_HORIZON = 120       # 预测未来 10 个点（每点间隔 1 分钟）
TIME_POINT_INDEX = 10000 # 上下文窗口结束位置（倒数第 200 行）

OUTPUT_DIR = Path("prediction_full_context")
OUTPUT_DIR.mkdir(exist_ok=True)

COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'amount']

# ==============================
# 主函数
# ==============================
def main():
    print("🔍 Loading fine-tuned model...")
    tokenizer = KronosTokenizer.from_pretrained(str(TOKENIZER_PATH))
    model = Kronos.from_pretrained(str(BASEMODEL_PATH))
    predictor = KronosPredictor(
        model=model,
        tokenizer=tokenizer,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        max_context=2048
    )

    print("📊 Loading test data...")
    df = pd.read_csv(TEST_DATA_PATH, parse_dates=["timestamps"])
    df = df.set_index("timestamps").sort_index()

    total_needed = LOOKBACK_WINDOW + PRED_HORIZON
    if len(df) < total_needed:
        raise ValueError(f"Test data too short! Need {total_needed}, got {len(df)}")

    start_idx = TIME_POINT_INDEX if TIME_POINT_INDEX >= 0 else len(df) + TIME_POINT_INDEX
    if start_idx < LOOKBACK_WINDOW:
        start_idx = LOOKBACK_WINDOW
    if start_idx + PRED_HORIZON > len(df):
        start_idx = len(df) - PRED_HORIZON

    # 获取完整 x（上下文）
    x_start = start_idx - LOOKBACK_WINDOW
    x_end = start_idx
    x_df = df.iloc[x_start:x_end][COLUMNS]
    x_timestamp = df.index[x_start:x_end]

    # 获取完整 y_true（未来 10 点）
    y_true_df = df.iloc[x_end : x_end + PRED_HORIZON][COLUMNS]
    y_true_timestamp = df.index[x_end : x_end + PRED_HORIZON]

    print(f"📈 Context: {x_timestamp[0]} → {x_timestamp[-1]}")
    print(f"🎯 Target:  {y_true_timestamp[0]} → {y_true_timestamp[-1]}")

    # 存储预测结果
    all_forecasts = np.full((PRED_HORIZON, N_SAMPLES, len(COLUMNS)), np.nan)

    # 滑动预测
    for i in range(PRED_HORIZON):
        context_end = x_end + i
        context_start = context_end - LOOKBACK_WINDOW

        if context_end >= len(df):
            break

        x_input = df.iloc[context_start:context_end][COLUMNS]
        x_ts_input = pd.Series(df.index[context_start:context_end])
        y_ts = pd.Series([df.index[context_end]])

        preds = []
        for _ in range(N_SAMPLES):
            pred_df = predictor.predict(
                df=x_input,
                x_timestamp=x_ts_input,
                y_timestamp=y_ts,
                pred_len=1,
                T=1.0,
                top_p=0.9,
                sample_count=1
            )
            preds.append(pred_df.values[0])
        all_forecasts[i] = np.array(preds)

    # 计算预测统计量
    pred_mean = all_forecasts.mean(axis=1)  # (10, 6)
    pred_std = all_forecasts.std(axis=1)    # (10, 6)

    # 合并完整时间轴：x + y
    full_timestamp = df.index[x_start : x_end + PRED_HORIZON]
    full_values = df.iloc[x_start : x_end + PRED_HORIZON][COLUMNS].values  # (70, 6)

    # 绘图：每个指标一张图
    for col_idx, col in enumerate(COLUMNS):
        plt.figure(figsize=(12, 5))

        # 1. 完整真实信号（x + y）
        plt.plot(full_timestamp, full_values[:, col_idx],
                 color='black', linewidth=1.5, label=f'True {col} (x + y)')

        # 2. 预测均值（y only）
        plt.plot(y_true_timestamp, pred_mean[:, col_idx],
                 'o-', color='red', linewidth=2, label='Predicted mean')

        # 3. 预测不确定性
        plt.fill_between(
            y_true_timestamp,
            pred_mean[:, col_idx] - pred_std[:, col_idx],
            pred_mean[:, col_idx] + pred_std[:, col_idx],
            color='lightcoral', alpha=0.4, label='±1 std'
        )

        # 竖线分隔 x 和 y
        plt.axvline(x=x_timestamp[-1], color='gray', linestyle='--', alpha=0.7, label='Prediction start')

        plt.title(f'Full Context + Prediction: {col} (N={N_SAMPLES})')
        plt.xlabel('Time')
        plt.ylabel(col)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"full_{col}.png", dpi=150)
        plt.close()

        print(f"✅ Saved full context plot for {col}.")

    print(f"\n📁 Plots saved to: {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    main()