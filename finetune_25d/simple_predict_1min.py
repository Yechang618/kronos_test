# finetune/simple_predict_step_by_step.py
import os
import sys
import pickle
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Fix OMP warning
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Add project root
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from config import Config
from model.kronos import Kronos, KronosTokenizer
from model.kronos import sample_from_logits
# ==============================
# 配置
# ==============================
idx_high = 2
idx_low = 3
TASK = "task8"
TEMPERATURE = 100

MODEL_NOTE, LOOKBACK_WINDOW, PRED_LENGTH = "_25d_1min", 480, 60
MODEL_NOTE, LOOKBACK_WINDOW, PRED_LENGTH = "_56d_1min", 10, 10

TOKENIZER_PATH = f"./core/models/model{MODEL_NOTE}/custom_25d_tokenizer/checkpoints/best_model"
PREDICTOR_PATH = f"./core/models/model{MODEL_NOTE}/custom_25d_predictor/checkpoints/best_model"

symbols = ["ADA", "AIXBT", "APT", "AVAX", "BCH", "BNB", "BTC",  # 6
           "CHESS", "COMP", "DOGE", "DOT", "ENA", "ETC","ETH", # 13
           "FET", "FORM", "HBAR", "HFT", "KAITO", "LINK", "LTC", # 20
           "NEAR", "OM", "ONDO", "PNUT", "SOL", "TAO", # 26
           "THE", "TON", "TRX", "TURBO",  # 30
           "UNI", "XLM", "XRP", "ZEC", # 34
           ] # 
SYMBOL = symbols[5]
START_TIME = "2025-10-05 00:00:00"
PRED_HORIZON = 60
N_SAMPLES = 10
note = f"{SYMBOL}_lookback{LOOKBACK_WINDOW}_pred{PRED_HORIZON}_samples{N_SAMPLES}_10min"
OUTPUT_DIR = Path(f"figures/step_pred_{note}")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ==============================
# My tools
# ==============================
# def compute_trends(pred):
#     """
#     计算趋势：1 表示上升，-1 表示下降，0 表示持平
#     data: numpy array, shape (N,)
#     """
#     assert pred.shape[1] == 6, "Input data must have 6 features"
#     close = pred[:, 3]  # 使用收盘价计算趋势

#     assert pred.shape[0] == PRED_LENGTH, "Input data must be PRED_LENGTH-dimensional"
#     if PRED_LENGTH == 1:
#         return close  # 单步预测无法计算趋势
#     K = 1 + PRED_LENGTH//10
#     trend = np.zeros(K)
#     intervels = [1]
#     for i in range(1, K):
#         if i*10 <= PRED_LENGTH:
#             intervels.append(i*5)
#         else:
#             break
#     for k, itv in enumerate(intervels):
#         trend[k] = (sum(close[itv:(2*itv)]) - sum(close[0:itv])) / itv
#     return trend

feature_names = ['Basis Bid', 'Basis Ask', 'High Bid', 'Low Ask', 'funding rate',  # 0 - 4
                    'index_price', 'spot_index_imbalance', 'mark_price',  # 5 - 7
                    'spot_buy_price', 'spot_sell_price', 'spot_buy_amount', 'spot_sell_amount', # 8 - 11
                    'swap_buy_price', 'swap_sell_price', 'swap_buy_amount', 'swap_sell_amount', # 12 - 15
                    'spot_bid0_price', 'spot_bid0_amount', 'spot_bid1_price', 'spot_bid1_amount',  # 16 - 19
                    'spot_bid2_price', 'spot_bid2_amount', 'spot_bid3_price', 'spot_bid3_amount',  # 20 - 23
                    'spot_bid4_price', 'spot_bid4_amount',  # 24 - 25
                    'spot_ask0_price', 'spot_ask0_amount', 'spot_ask1_price', 'spot_ask1_amount', # 26 - 29
                    'spot_ask2_price', 'spot_ask2_amount', 'spot_ask3_price', 'spot_ask3_amount', #     30 - 33
                    'spot_ask4_price', 'spot_ask4_amount', # 34 - 35
                    'swap_bid0_price', 'swap_bid0_amount', 'swap_bid1_price', 'swap_bid1_amount',  # 36 - 39
                    'swap_bid2_price', 'swap_bid2_amount', 'swap_bid3_price', 'swap_bid3_amount',  # 40 - 43
                    'swap_bid4_price', 'swap_bid4_amount',  # 44 - 45
                    'swap_ask0_price', 'swap_ask0_amount','swap_ask1_price', 'swap_ask1_amount', # 46 - 49
                    'swap_ask2_price', 'swap_ask2_amount','swap_ask3_price', 'swap_ask3_amount', # 50 - 53
                    'swap_ask4_price', 'swap_ask4_amount'] # 54 - 55
# ==============================
# 自定义 Predictor（简化版）
# ==============================
class KronosPredictor:
    def __init__(self, model, tokenizer, device, max_context=2048):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_context = max_context

    def predict(self, x, x_stamp, y_stamp, pred_len=1, T=1.0, top_p=0.9, top_k=0):
    # def predict(self, x, x_stamp, y_stamp, ...):
        self.tokenizer = self.tokenizer.to(self.device)
        self.model = self.model.to(self.device)
        """
        预测未来 pred_len 步（这里只用 pred_len=1）
        """
        with torch.no_grad():
            x = torch.from_numpy(x).unsqueeze(0).to(self.device)
            x_stamp = torch.from_numpy(x_stamp).unsqueeze(0).to(self.device)
            y_stamp = torch.from_numpy(y_stamp).unsqueeze(0).to(self.device)
            # print(f"x shape: {x.shape}, x_stamp shape: {x_stamp.shape}, y_stamp shape: {y_stamp.shape}")

            x_token = self.tokenizer.encode(x, half=True)

            initial_seq_len = x.size(1)
            batch_size = x_token[0].size(0)
            total_seq_len = initial_seq_len + pred_len
            full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

            generated_pre = x_token[0].new_empty(batch_size, pred_len)
            generated_post = x_token[1].new_empty(batch_size, pred_len)

            pre_buffer = x_token[0].new_zeros(batch_size, self.max_context)
            post_buffer = x_token[1].new_zeros(batch_size, self.max_context)
            buffer_len = min(initial_seq_len, self.max_context)
            if buffer_len > 0:
                start_idx = max(0, initial_seq_len - self.max_context)
                pre_buffer[:,:buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
                post_buffer[:,:buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]

            for i in range(pred_len):
                current_seq_len = initial_seq_len + i
                window_len = min(current_seq_len, self.max_context)

                if current_seq_len <= self.max_context:
                    input_tokens = [
                        pre_buffer[:,:window_len],
                        post_buffer[:,:window_len]
                    ]
                else:
                    input_tokens = [pre_buffer, post_buffer]

                context_end = current_seq_len
                context_start = max(0, context_end - self.max_context)
                current_stamp = full_stamp[:, context_start:context_end,:].contiguous()

                s1_logits, context = self.model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
                s1_logits = s1_logits[:, -1,:]

                sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

                s2_logits = self.model.decode_s2(context, sample_pre)
                s2_logits = s2_logits[:, -1,:]
                sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

                generated_pre[:, i] = sample_pre.squeeze(-1)
                generated_post[:, i] = sample_post.squeeze(-1)

                if current_seq_len < self.max_context:
                    pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                    post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
                else:
                    pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                    post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                    pre_buffer[:, -1] = sample_pre.squeeze(-1)
                    post_buffer[:, -1] = sample_post.squeeze(-1)

            full_pre = torch.cat([x_token[0], generated_pre], dim=1)
            full_post = torch.cat([x_token[1], generated_post], dim=1)

            context_start = max(0, total_seq_len - self.max_context)
            input_tokens = [
                full_pre[:, context_start:total_seq_len].contiguous(),
                full_post[:, context_start:total_seq_len].contiguous()
            ]
            z = self.tokenizer.decode(input_tokens, half=True)
            # print(f"z shape: {z.shape}")  # Debug 信息
            return z[0, -pred_len:,:].cpu().numpy()  # (pred_len, 6)

# ==============================
# 主函数
# ==============================
def main():
    print("🔍 Loading models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_PATH)
    model = Kronos.from_pretrained(PREDICTOR_PATH)
    predictor = KronosPredictor(model, tokenizer, device, max_context=2048)

    print(f"📊 Loading test data from ./datasets/{TASK}/processed_datasets/{SYMBOL}/test_data.pkl")
    with open(f"./datasets/{TASK}/processed_datasets/{SYMBOL}/test_data.pkl", 'rb') as f:
        data = pickle.load(f)
    df = data[SYMBOL].copy()
    df['minute'] = df.index.minute
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df['day'] = df.index.day
    df['month'] = df.index.month

    config = Config()
    feature_list = config.feature_list  # ['open', 'high', 'low', 'close', 'vol', 'amt']
    time_features = ['minute', 'hour', 'weekday', 'day', 'month']

    start_ts = pd.Timestamp(START_TIME)
    try:
        start_idx = df.index.get_loc(start_ts)
    except KeyError:
        start_idx = df.index.get_indexer([start_ts], method='nearest')[0]

    # 验证数据长度
    total_needed = start_idx + LOOKBACK_WINDOW + PRED_HORIZON
    if total_needed > len(df):
        raise ValueError(f"Not enough data after {START_TIME}")

    # x: [start_idx, start_idx + 240)
    x_start = start_idx
    x_end = start_idx + LOOKBACK_WINDOW
    x_df = df.iloc[x_start:x_end][feature_list]
    x_time = df.index[x_start:x_end]

    # y_true: [x_end, x_end + 30)
    y_true_df = df.iloc[x_end:x_end + PRED_HORIZON][feature_list]
    y_time = df.index[x_end:x_end + PRED_HORIZON]

    print(f"📈 Context: {x_time[0]} → {x_time[-1]}")
    print(f"🎯 Target:  {y_time[0]} → {y_time[-1]}")

    # 存储所有预测 (30, 20, 6)
    all_forecasts = np.full((PRED_HORIZON, N_SAMPLES, len(feature_list)), np.nan)
    all_forecasts_trends = np.full((PRED_HORIZON, N_SAMPLES, 1 + PRED_LENGTH//10), np.nan)
    # 逐步预测
    for i in range(PRED_HORIZON):
        context_end = x_end + i
        context_start = context_end - LOOKBACK_WINDOW

        x_input = df.iloc[context_start:context_end][feature_list].values.astype(np.float32)
        x_stamp_input = df.iloc[context_start:context_end][time_features].values.astype(np.float32)
        y_stamp = df.iloc[context_end:context_end + PRED_LENGTH][time_features].values.astype(np.float32)

        # Normalize x_input
        x_mean, x_std = np.mean(x_input, axis=0), np.std(x_input, axis=0)
        x_input_norm = (x_input - x_mean) / (x_std + 1e-5)
        x_input_norm = np.clip(x_input_norm, -5.0, 5.0)

        preds = []
        # trends = []
        for _ in range(N_SAMPLES):
            pred = predictor.predict(
                x=x_input_norm,
                x_stamp=x_stamp_input,
                y_stamp=y_stamp,
                pred_len=PRED_LENGTH,
                T=TEMPERATURE,
                top_p=0.9,
                top_k=0
            )  # (1, 6)
            # print(f"Step {i+1}/{PRED_HORIZON}, Sample Prediction: {pred}")
            # print(f"pred shape: {pred.shape}") # predict shape: (PRED_LENGTH, 6)
            for j in range(pred.shape[0]):
                pred[j,:] = pred[j,:]* (x_std + 1e-5) + x_mean  # 反归一化
            # preds.append(pred[0,:])  # 反归一化
            preds.append(np.mean(pred, axis = 0))  # 反归一化
        all_forecasts[i] = np.array(preds)
        # all_forecasts_trends[i] = np.array(trends)

    # 计算统计量
    pred_mean = all_forecasts.mean(axis=1)  # (30, 6)
    pred_std = all_forecasts.std(axis=1)    # (30, 6)
    # pred_trend_mean = all_forecasts_trends.mean(axis=1)  # (30, trend_dim)
    # pred_trend_std = all_forecasts_trends.std(axis=1)    # (30, trend_dim)

    # 完整时间轴：x + y
    full_time = df.index[x_start:x_end + PRED_HORIZON]
    full_values = df.iloc[x_start:x_end + PRED_HORIZON][feature_list].values  # (270, 6)
    true_y_values = y_true_df.values  # (30, 6)

    # Figure 1: High and Low with Weighted vs Unweighted Predictions
    fig1, axes1 = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    # (0) High and low
    ax = axes1[0]
    idx1 = 2
    idx2 = 3
    label1 = f'True {feature_names[idx1]}'
    label2 = f'True {feature_names[idx2]}'
    # High
    ax.plot(y_time, true_y_values[:, idx1], color='purple', linewidth=1.5, label=label1)
    ax.plot(y_time, pred_mean[:, idx1], 'o-', color='green', linewidth=2, label='Pred High')
    ax.fill_between(
        y_time,
        pred_mean[:, idx1] - pred_std[:, idx1],
        pred_mean[:, idx1] + pred_std[:, idx1],
        color='lightgreen', alpha=0.3
    )
    # Low
    ax.plot(y_time, true_y_values[:, idx2], color='darkgoldenrod', linewidth=1.5, label=label2)
    ax.plot(y_time, pred_mean[:, idx2], 'o-', color='blue', linewidth=2, label='Pred Low')
    ax.fill_between(
        y_time,
        pred_mean[:, idx2] - pred_std[:, idx2],
        pred_mean[:, idx2] + pred_std[:, idx2],
        color='lightblue', alpha=0.3
    )
    ax.set_ylabel(f'{feature_names[idx1]} and {feature_names[idx2]}')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)

    # (1) Funding rate and index_imbalance
    ax = axes1[1]
    idx1 = 4
    idx2 = 6
    label1 = f'True {feature_names[idx1]}'
    label2 = f'True {feature_names[idx2]}'
    # Idx1
    ax.plot(y_time, true_y_values[:, idx1], color='purple', linewidth=1.5, label=label1)
    ax.plot(y_time, pred_mean[:, idx1], 'o-', color='green', linewidth=2, label=f'Pred {feature_names[idx1]}')
    ax.fill_between(
        y_time,
        pred_mean[:, idx1] - pred_std[:, idx1],
        pred_mean[:, idx1] + pred_std[:, idx1],
        color='lightgreen', alpha=0.3
    )
    # Idx2
    ax.plot(y_time, true_y_values[:, idx2], color='darkgoldenrod', linewidth=1.5, label=label2)
    ax.plot(y_time, pred_mean[:, idx2], 'o-', color='blue', linewidth=2, label=f'Pred {feature_names[idx2]}')
    ax.fill_between(
        y_time,
        pred_mean[:, idx2] - pred_std[:, idx2],
        pred_mean[:, idx2] + pred_std[:, idx2],
        color='lightblue', alpha=0.3
    )
    ax.set_ylabel(f'{feature_names[idx1]} and {feature_names[idx2]}')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)
    plt.xticks(rotation=45)

    fig1.suptitle(f'{SYMBOL} - Price and Volume Prediction (N={N_SAMPLES})')
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig1.savefig(OUTPUT_DIR / f"{SYMBOL}_price_volume.png", dpi=150)
    plt.close(fig1)

    # Figure 2: Order book features 
    fig2, axes2 = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    order_book_features = [16, 26, 36, 46]  # spot_buy_price, spot_sell_price, swap_buy_price, swap_sell_price
    for j, idx in enumerate(order_book_features):
        ax = axes2[j]
        label = f'True {feature_names[idx]}'
        ax.plot(y_time, true_y_values[:, idx], color='purple', linewidth=1.5, label=label)
        ax.plot(y_time, pred_mean[:, idx], 'o-', color='green', linewidth=2, label=f'Pred {feature_names[idx]}')
        ax.fill_between(
            y_time,
            pred_mean[:, idx] - pred_std[:, idx],
            pred_mean[:, idx] + pred_std[:, idx],
            color='lightgreen', alpha=0.3
        )

        ax.set_ylabel(feature_names[idx])
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.7)
    plt.xticks(rotation=45)
    fig2.suptitle(f'{SYMBOL} - Order Book Prediction (N={N_SAMPLES})')
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.savefig(OUTPUT_DIR / f"{SYMBOL}_order_book.png", dpi=150)
    plt.close(fig2)

    # Figure 3: Order book features, amounts only 
    fig3, axes3 = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    order_book_features = [17, 27, 37, 47]  # spot_buy_amount, spot_sell_amount, swap_buy_amount, swap_sell_amount
    for j, idx in enumerate(order_book_features):
        ax = axes3[j]
        label = f'True {feature_names[idx]}'
        ax.plot(y_time, true_y_values[:, idx], color='purple', linewidth=1.5, label=label)
        ax.plot(y_time, pred_mean[:, idx], 'o-', color='green', linewidth=2, label=f'Pred {feature_names[idx]}')
        ax.fill_between(
            y_time,
            pred_mean[:, idx] - pred_std[:, idx],
            pred_mean[:, idx] + pred_std[:, idx],
            color='lightgreen', alpha=0.3
        )

        ax.set_ylabel(feature_names[idx])
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.7)
    plt.xticks(rotation=45)
    fig3.suptitle(f'{SYMBOL} - Order Book Prediction (N={N_SAMPLES})')
    fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig3.savefig(OUTPUT_DIR / f"{SYMBOL}_order_book_amounts.png", dpi=150)
    plt.close(fig3)

    # Figure 4: Order book features, amounts only 
    fig4, axes4 = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    # order_book_features = [10, 11, 14, 15]  # spot_buy_amount, spot_sell_amount, swap_buy_amount, swap_sell_amount
    order_book_features = [8, 9, 12, 13]  # spot_buy_amount, spot_sell_amount, swap_buy_amount, swap_sell_amount
    for j, idx in enumerate(order_book_features):
        ax = axes4[j]
        label = f'True {feature_names[idx]}'
        ax.plot(y_time, pred_mean[:, idx], 'o-', color='green', linewidth=2, label=f'Pred {feature_names[idx]}')
        ax.fill_between(
            y_time,
            pred_mean[:, idx] - pred_std[:, idx],
            pred_mean[:, idx] + pred_std[:, idx],
            color='lightgreen', alpha=0.3
        )

        ax.plot(y_time, true_y_values[:, idx], color='purple', linewidth=1.5, label=label)            
        ax.set_ylabel(feature_names[idx])
        # ax.set_yscale('log')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.7)
    plt.xticks(rotation=45)
    fig4.suptitle(f'{SYMBOL} - Order Book Prediction (N={N_SAMPLES})')
    fig4.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig4.savefig(OUTPUT_DIR / f"{SYMBOL}_order_book_trade.png", dpi=150)
    plt.close(fig4)

    # Figure 5: Basis
    order_book_features = [0, 1]  # Basis Bid, Basis Ask
    n_features = len(order_book_features)
    fig5, axes5 = plt.subplots(n_features, 1, figsize=(12, 12), sharex=True)
    for j, idx in enumerate(order_book_features):
        ax = axes5[j]
        label = f'True {feature_names[idx]}'
        ax.plot(y_time, pred_mean[:, idx], 'o-', color='green', linewidth=2, label=f'Pred {feature_names[idx]}')
        ax.fill_between(
            y_time,
            pred_mean[:, idx] - pred_std[:, idx],
            pred_mean[:, idx] + pred_std[:, idx],
            color='lightgreen', alpha=0.3
        )

        if j == 0:
            ax.plot(y_time, pred_mean[:, 16] - pred_mean[:, 46], '--', color='red', linewidth=2, label=f'Pred {feature_names[idx]} (Bid-Ask)')
        elif j == 1:
            ax.plot(y_time, pred_mean[:, 26] - pred_mean[:, 36], '--', color='red', linewidth=2, label=f'Pred {feature_names[idx]} (Ask-Bid)')
        ax.plot(y_time, true_y_values[:, idx], color='purple', linewidth=1.5, label=label)            
        ax.set_ylabel(feature_names[idx])
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.7)
    plt.xticks(rotation=45)
    fig5.suptitle(f'{SYMBOL} - Order Book Prediction (N={N_SAMPLES})')
    fig5.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig5.savefig(OUTPUT_DIR / f"{SYMBOL}_order_book_basis.png", dpi=150)
    plt.close(fig5)

    # Figure 6: Order book spot bid amount
    order_book_features = [17, 19, 21, 23, 25]  # spot_bid0_amount, spot_bid1_amount, spot_bid2_amount, spot_bid3_amount, spot_bid4_amount
    fig6, axes6 = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    for j, idx in enumerate(order_book_features):
        ax = axes6[j]
        label = f'True {feature_names[idx]}'
        ax.plot(y_time, pred_mean[:, idx], 'o-', color='green', linewidth=2, label=f'Pred {feature_names[idx]}')
        ax.fill_between(
            y_time,
            pred_mean[:, idx] - pred_std[:, idx],
            pred_mean[:, idx] + pred_std[:, idx],
            color='lightgreen', alpha=0.3
        )

        ax.plot(y_time, true_y_values[:, idx], color='purple', linewidth=1.5, label=label)            
        ax.set_ylabel(feature_names[idx])
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.7)
    plt.xticks(rotation=45)
    fig6.suptitle(f'{SYMBOL} - Order Book Prediction (N={N_SAMPLES})')
    fig6.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig6.savefig(OUTPUT_DIR / f"{SYMBOL}_order_book_spot_bid_amount.png", dpi=150)
    plt.close(fig6)

    # Figure 7: Order book spot ask amount
    order_book_features = [27, 29, 31, 33, 35]  # spot_ask0_amount, spot_ask1_amount, spot_ask2_amount, spot_ask3_amount, spot_ask4_amount
    fig7, axes7 = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    for j, idx in enumerate(order_book_features):
        ax = axes7[j]
        label = f'True {feature_names[idx]}'
        ax.plot(y_time, pred_mean[:, idx], 'o-', color='green', linewidth=2, label=f'Pred {feature_names[idx]}')
        ax.fill_between(
            y_time,
            pred_mean[:, idx] - pred_std[:, idx],
            pred_mean[:, idx] + pred_std[:, idx],
            color='lightgreen', alpha=0.3
        )

        ax.plot(y_time, true_y_values[:, idx], color='purple', linewidth=1.5, label=label)            
        ax.set_ylabel(feature_names[idx])
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.7)
    plt.xticks(rotation=45)
    fig7.suptitle(f'{SYMBOL} - Order Book Prediction (N={N_SAMPLES})')
    fig7.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig7.savefig(OUTPUT_DIR / f"{SYMBOL}_order_book_spot_ask_amount.png", dpi=150)
    plt.close(fig7)
    
    # ==============================
    # 绘图：第二张图（K,1）—— Trend Components
    # ==============================
    # n_trends = pred_trend_mean.shape[1]
    # if n_trends > 0:
    #     fig2, axes2 = plt.subplots(n_trends, 1, figsize=(12, 4 * n_trends), sharex=True)
    #     if n_trends == 1:
    #         axes2 = [axes2]  # Ensure list for single subplot

    #     for k in range(n_trends):
    #         ax = axes2[k]
    #         ax.plot(y_time, pred_trend_mean[:, k], 'o-', color='blue', linewidth=2, label=f'Trend {k+1} Mean')
    #         ax.fill_between(
    #             y_time,
    #             pred_trend_mean[:, k] - pred_trend_std[:, k],
    #             pred_trend_mean[:, k] + pred_trend_std[:, k],
    #             color='lightblue', alpha=0.4, label='±1 std'
    #         )
    #         ax.axvline(x=x_time[-1], color='gray', linestyle='--', alpha=0.7)
    #         ax.set_ylabel(f'Trend {k+1}')
    #         ax.legend()
    #         ax.grid(True, linestyle=':', alpha=0.7)

    #     axes2[-1].set_xlabel('Time')
    #     plt.xticks(rotation=45)
    #     fig2.suptitle(f'{SYMBOL} - Trend Components (N={N_SAMPLES})')
    #     fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    #     fig2.savefig(OUTPUT_DIR / f"{SYMBOL}_trend_components.png", dpi=150)
    #     plt.close(fig2)
    # else:
    #     print("⚠️ No trend components to plot.")

    print(f"✅ All plots saved to {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    main()