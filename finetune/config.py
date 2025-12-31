# ./finetune/config.py
import os

class Config:
    def __init__(self):
        # 数据字段（适配 CSV）
        self.feature_list = ['open', 'high', 'low', 'close', 'volume', 'amount']
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        # 路径
        self.dataset_path = "./data/processed_datasets"

        # 时间窗口
        self.lookback_window = 144
        self.predict_window = 6
        self.max_context = 2048
        self.clip = 5.0

        # 训练参数
        self.seed = 42
        self.batch_size = 10
        self.log_interval = 100
        self.epochs = 50
        self.n_train_iter = 50000 * self.batch_size
        self.n_val_iter = 8000 * self.batch_size

        self.tokenizer_learning_rate = 5e-4
        self.predictor_learning_rate = 1e-6
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.1
        self.accumulation_steps = 1

        # 模型路径
        self.pretrained_tokenizer_path = "trained/task4/tokenizer/best_model"
        self.pretrained_predictor_path = "trained/task4/basemodel/best_model"

        self.save_path = "./outputs/models"
        self.tokenizer_save_folder_name = "finetune_tokenizer_all"
        self.predictor_save_folder_name = "finetune_predictor_all"

        # 测试
        self.backtest_result_path = "./outputs/backtest_results"
        self.backtest_save_folder_name = "task4_multisymbol_backtest"
        self.backtest_time_range = ["2025-10-01", "2025-10-29"]

        # Comet（禁用）
        self.use_comet = False