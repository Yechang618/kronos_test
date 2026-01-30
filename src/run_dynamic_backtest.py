# backtest/run_dynamic_backtest.py
from pathlib import Path
from signal_generator import DynamicSignalGenerator, KronosPredictor
from backtest_engine import BacktestEngine

def run_dynamic_strategy_backtest(symbol: str, 
                                 start_time: str, 
                                 end_time: str,
                                 fditv: int = 8):
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
    
    # 3. 初始化回测引擎
    engine = BacktestEngine(
        symbol=symbol,
        start_time=start_time,
        end_time=end_time,
        initial_capital=1000.0
    )
    
    # 4. 注册Dynamic策略
    engine.register_strategy("dynamic", signal_gen)
    
    # 5. 运行回测
    engine.run(
        signal_update_freq='30min',
        reweight_freq='10min',
        fditv=fditv
    )
    
    # 6. 保存结果
    results_dir = Path("./backtest/data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = engine.get_results()
    results_df.to_csv(results_dir / f"dynamic_strategy_{symbol}_{start_time[:10]}.csv")
    
    return results_df

if __name__ == "__main__":
    # 示例：运行TAO币种回测
    run_dynamic_strategy_backtest(
        symbol="TAO",
        start_time="2025-10-01 00:00:00",
        end_time="2025-10-07 23:59:59",
        fditv=4
    )