# backtest/run_dynamic_high_low.py
from pathlib import Path
from KronosPredictor import DynamicSignalGenerator, KronosPredictor
import sys, os

# Add project root
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

# from config import Config
from config import Config

LOOKBACK = Config().lookback
PRED_LENGTH = Config().pred_length
N_SAMPLES = Config().n_samples

TOKENIZER_PATH_10min = Config().tokenizer_10min
PREDICTOR_PATH_10min = Config().predictor_10min

symbol_list = Config().symbol_list

def run_dynamic_strategy_backtest(symbol_list: list[str], 
                                 start_time: str, 
                                 end_time: str):
    # 1. 初始化预测器
    predictor = KronosPredictor(
        tokenizer_path="./outputs/models_144p48/finetune_tokenizer_all/checkpoints/best_model",
        predictor_path="./outputs/models_144p48/finetune_predictor_all/checkpoints/best_model"
    )
    
    # 2. 创建信号生成器
    signal_gen = DynamicSignalGenerator(
        predictor=predictor,
        lookback=144,
        pred_length=48,
        n_samples=100
    )

    

    return results_df

if __name__ == "__main__":
    # 示例：运行TAO币种回测
    run_dynamic_strategy_backtest(
        symbol_list=symbol_list,
        start_time="2025-10-01 00:00:00",
        end_time="2025-10-07 23:59:59"
    )