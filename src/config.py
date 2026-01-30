# ./finetune/config.py
import os

class Config:
    def __init__(self):
        # 数据字段（适配 CSV）
        self.feature_list = ['open', 'high', 'low', 'close', 'volume', 'amount']
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']
        self.symbol_list = ['BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'TAO']
        self.symbol_list_kc = ['BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'TAO']

        # 路径
        self.dataset_path = "./datasets/temp"
        self.tokenizer_10min = f"./core/models/model_10min_144p48/finetune_tokenizer_all/checkpoints/best_model"
        self.predictor_10min = f"./core/models/model_10min_144p48/finetune_predictor_all/checkpoints/best_model"

        # Parameters
        self.lookback = 144
        self.pred_length = 48
        self.n_samples = 30

        # 模型路径
        self.pretrained_tokenizer_path = "core/pretrained/tokenizer/best_model"
        self.pretrained_predictor_path = "core/pretrained/basemodel/best_model"

        self.save_path = "./outputs/models"
        self.tokenizer_save_folder_name = "finetune_tokenizer_all"
        self.predictor_save_folder_name = "finetune_predictor_all"

        # 测试
        self.backtest_result_path = "./outputs/backtest_results"
        self.backtest_save_folder_name = "task6_multisymbol_backtest"
        self.backtest_time_range = ["2025-10-01", "2025-10-29"]

        # Comet（禁用）
        self.use_comet = False