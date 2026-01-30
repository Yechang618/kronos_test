# backtest/backtest_engine.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Callable, Dict

class BacktestEngine:
    """
    通用回测引擎
    支持多策略并行回测
    """
    def __init__(self, 
                 symbol: str,
                 start_time: str,
                 end_time: str,
                 initial_capital: float = 1000.0,
                 alpha: float = 0.1,
                 beta: float = 0.9):
        self.symbol = symbol
        self.start_time = pd.Timestamp(start_time)
        self.end_time = pd.Timestamp(end_time)
        self.initial_capital = initial_capital
        self.alpha = alpha
        self.beta = beta
        
        # 交易成本配置
        self.costs = {
            'c_t_swap': 0.000153,
            'c_t_spot': 0.0001725,
            'c_m_swap': 0.0,
            'c_m_spot': 0.0000825,
            'premium_tt': 0.0002,
            'premium_mt': 0.0006
        }
        
        # 策略状态
        self.strategies = {}  # {strategy_id: StrategyState}
        self.results = {
            'timestamps': [],
            'pnl': {},
            'positions': {},
            'params': {}
        }
        self.raw_df = None

    def register_strategy(self, strategy_id: str, signal_generator: Callable):
        """注册策略及其信号生成器"""
        self.strategies[strategy_id] = {
            'signal_generator': signal_generator,
            'position_swap': 0.0,
            'position_spot': 0.0,
            'capital_swap': 0.0,
            'capital': self.initial_capital,
            'last_signal_update': None,
            'last_reweight_update': None
        }
        self.results['pnl'][strategy_id] = []
        self.results['params'][strategy_id] = []

    def load_data(self, data_dir: str = "./backtest/data"):
        """加载100ms粒度市场数据"""
        # ... (保留原load_data逻辑)
        pass

    def execute_trade(self, strategy_id: str, timestamp: pd.Timestamp, 
                     thresholds: Dict[str, float], funding_rate: float):
        """执行单次交易（简化版，保留核心逻辑）"""
        # ... (从原execute_trade提取策略2逻辑，移除self引用)
        pass

    def funding_settlement(self, timestamp: pd.Timestamp, funding_rate: float, 
                          mid_swap_price: float, fditv: int = 8):
        """资金费率结算"""
        # ... (保留原funding_fee_settlement逻辑)
        pass

    def calculate_pnl(self, strategy_id: str, timestamp: pd.Timestamp) -> float:
        """计算策略PnL"""
        # ... (保留原calculate_pnl逻辑)
        pass

    def run(self, 
            signal_update_freq: str = '30min',   # 信号生成频率
            reweight_freq: str = '10min',        # 重加权频率
            fditv: int = 8):                     # 资金费率间隔(4/8小时)
        """
        执行回测主循环
        """
        # 1. 数据准备
        self.load_data()
        df_grouped = self.raw_df.groupby(pd.Grouper(freq='1s'))
        
        # 2. 按秒执行回测
        for second_timestamp, second_group in df_grouped:
            if second_group.empty:
                continue
            
            # 获取资金费率
            funding_rates = second_group['funding_rate'].dropna().values
            funding_rate = funding_rates[0] if len(funding_rates) > 0 else 0.0
            
            # 3. 更新各策略信号
            for sid, state in self.strategies.items():
                # 初始信号生成（每30分钟）
                if (state['last_signal_update'] is None or 
                    second_timestamp - state['last_signal_update'] >= pd.Timedelta(signal_update_freq)):
                    df_window = self.raw_df[
                        second_timestamp - pd.Timedelta(hours=24):
                        second_timestamp + pd.Timedelta(minutes=10 * 48)
                    ]
                    thresholds = state['signal_generator'].generate_initial_signal(
                        df_window, second_timestamp, 
                        feature_list=['open','high','low','close','volume','amount'],
                        time_features=['minute','hour','weekday','day','month']
                    )
                    state['last_signal_update'] = second_timestamp
                    state['current_thresholds'] = thresholds
                
                # 重加权更新（每10分钟）
                elif (state['last_reweight_update'] is None or 
                      second_timestamp - state['last_reweight_update'] >= pd.Timedelta(reweight_freq)):
                    if 'pred_sequences' in state and state['pred_sequences'] is not None:
                        # 获取最新观测价格
                        obs_price = second_group['close'].dropna().values[-1] if 'close' in second_group else None
                        if obs_price is not None:
                            thresholds = state['signal_generator'].update_signal_with_observation(
                                obs_price, second_timestamp
                            )
                            state['last_reweight_update'] = second_timestamp
                            state['current_thresholds'] = thresholds
            
            # 4. 执行交易（每秒最多一次）
            for timestamp in second_group.index:
                for sid, state in self.strategies.items():
                    if state.get('current_thresholds'):
                        self.execute_trade(sid, timestamp, state['current_thresholds'], funding_rate)
            
            # 5. 资金费率结算（特定时间点）
            funding_hours = [0, 4, 8, 12, 16, 20] if fditv == 4 else [0, 8, 16]
            if (second_timestamp.hour in funding_hours and 
                second_timestamp.minute == 1 and 
                second_timestamp.second == 0):
                mid_swap = (second_group['swap_bid0_price'].mean() + 
                           second_group['swap_ask0_price'].mean()) / 2
                for sid in self.strategies:
                    self.funding_settlement(second_timestamp, funding_rate, mid_swap, fditv)
            
            # 6. 记录结果
            pnl_snapshot = {}
            for sid in self.strategies:
                pnl = self.calculate_pnl(sid, second_group.index[-1])
                pnl_snapshot[sid] = pnl
                self.results['pnl'][sid].append(pnl)
                if 'current_thresholds' in self.strategies[sid]:
                    mid_param = (self.strategies[sid]['current_thresholds'][0] + 
                               self.strategies[sid]['current_thresholds'][1]) / 2
                    self.results['params'][sid].append(mid_param)
            
            self.results['timestamps'].append(second_group.index[-1])

    def get_results(self) -> pd.DataFrame:
        """获取回测结果"""
        df = pd.DataFrame({'timestamp': self.results['timestamps']})
        for sid in self.strategies:
            df[f'pnl_{sid}'] = self.results['pnl'][sid]
            df[f'param_{sid}'] = self.results['params'][sid]
        return df.set_index('timestamp')