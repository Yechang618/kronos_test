# ./finetune_25d/config.py

class Config:
    def __init__(self):
        # 修改 feature_list 为 25 维
        # self.feature_list = [f'x_{i}' for i in range(25)]
        self.feature_list = ['basis_bid', 'basis_ask', 'basis_high', 'basis_low', 'funding_rate', 
                        'index_price', 'spot_index_imbalance', 'mark_price', 
                        'spot_buy_price', 'spot_sell_price', 'spot_buy_amount', 'spot_sell_amount', 
                        'swap_buy_price', 'swap_sell_price', 'swap_buy_amount', 'swap_sell_amount',
                        'spot_bid0_price', 'spot_bid0_amount', 'spot_bid1_price', 'spot_bid1_amount', 
                        'spot_bid2_price', 'spot_bid2_amount', 'spot_bid3_price', 'spot_bid3_amount', 
                        'spot_bid4_price', 'spot_bid4_amount', 
                        'spot_ask0_price', 'spot_ask0_amount', 'spot_ask1_price', 'spot_ask1_amount',
                        'spot_ask2_price', 'spot_ask2_amount', 'spot_ask3_price', 'spot_ask3_amount',
                        'spot_ask4_price', 'spot_ask4_amount',
                        'swap_bid0_price', 'swap_bid0_amount', 'swap_bid1_price', 'swap_bid1_amount', 
                        'swap_bid2_price', 'swap_bid2_amount', 'swap_bid3_price', 'swap_bid3_amount', 
                        'swap_bid4_price', 'swap_bid4_amount', 
                        'swap_ask0_price', 'swap_ask0_amount','swap_ask1_price', 'swap_ask1_amount',
                        'swap_ask2_price', 'swap_ask2_amount','swap_ask3_price', 'swap_ask3_amount',
                        'swap_ask4_price', 'swap_ask4_amount']
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        # 路径配置
        self.dataset_path = "./datasets/custom_25d/processed_datasets"

        # 时间窗口
        self.lookback_window = 30
        self.predict_window = 5
        self.max_context = 2048
        self.clip = 5.0

        # 训练参数
        self.seed = 42
        self.batch_size = 1000
        self.log_interval = 100
        self.epochs = 2000
        self.n_train_iter = 500 * self.batch_size
        self.n_val_iter = 1200 * self.batch_size

        self.tokenizer_learning_rate = 1e-5
        self.predictor_learning_rate = 1e-6
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.1
        self.accumulation_steps = 1

        # 模型路径
        self.pretrained_tokenizer_path = "./core/models/model_25d_1min/custom_25d_tokenizer/best_model"  # 不使用预训练 tokenizer
        # self.pretrained_predictor_path = "./core/pretrained/basemodel/best_model"  # Kronos-mini
        self.pretrained_predictor_path = "./core/pretrained_100K/basemodel/best_model"  # Kronos-base

        self.save_path = "./core/models/model_25d_1min"
        self.tokenizer_save_folder_name = "custom_25d_tokenizer"
        # self.predictor_save_folder_name = "custom_25d_predictor"
        self.predictor_save_folder_name = "custom_100M_predictor"

        # Comet
        self.use_comet = False
