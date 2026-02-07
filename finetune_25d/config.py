# ./finetune_25d/config.py

class Config:
    def __init__(self):
        # 修改 feature_list 为 25 维
        # self.feature_list = [f'x_{i}' for i in range(25)]
        self.feature_list = ['basis_bid', 'basis_ask', 'funding_rate', 'index_price', 'mark_price',
                             'bid_price_0', 'bid_volume_0', 'bid_price_1', 'bid_volume_1', 'bid_price_2',
                             'bid_volume_2', 'bid_price_3', 'bid_volume_3', 'bid_price_4', 'bid_volume_4',
                             'ask_price_0', 'ask_volume_0', 'ask_price_1', 'ask_volume_1', 'ask_price_2', 
                             'ask_volume_2', 'ask_price_3', 'ask_volume_3', 'ask_price_4', 'ask_volume_4'
                             ]
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        # 路径配置
        self.dataset_path = "./datasets/custom_25d/processed_datasets"

        # 时间窗口
        self.lookback_window = 240
        self.predict_window = 60
        self.max_context = 512
        self.clip = 5.0

        # 训练参数
        self.seed = 42
        self.batch_size = 10
        self.log_interval = 100
        self.epochs = 50

        self.tokenizer_learning_rate = 1e-3
        self.predictor_learning_rate = 1e-4
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.1

        # 模型路径
        self.pretrained_tokenizer_path = None  # 不使用预训练 tokenizer
        self.pretrained_predictor_path = "./core/pretrained/basemodel/best_model"  # Kronos-mini

        self.save_path = "./core/models/model_25d_1min"
        self.tokenizer_save_folder_name = "custom_25d_tokenizer"
        self.predictor_save_folder_name = "custom_25d_predictor"

        # Comet
        self.use_comet = False