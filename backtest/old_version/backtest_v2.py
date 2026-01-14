# backtest/strategy_backtest_dynamic.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dynamic_params import calculate_dynamic_trading_params

class DynamicBasisArbitrageBacktest:
    def __init__(self, symbol, start_time, end_time, a=0.001, b=0.0008):
        """
        初始化动态参数回测
        
        Parameters:
        -----------
        symbol : str
            交易对符号
        start_time, end_time : str
            回测时间范围
        a, b : float
            动态参数控制因子
        """
        self.symbol = symbol
        self.start_time = start_time
        self.end_time = end_time
        self.a = a
        self.b = b
        
        # 回测参数
        self.P = 1000.0
        self.alpha = 0.1
        self.beta = 0.5
        self.dt = 0.01
        
        # 交易成本（与策略函数保持一致）
        self.c_t_swap = 0.000153
        self.c_t_spot = 0.0001725
        self.c_m_swap = 0.0
        self.c_m_spot = 0.0000825
        
        # 仓位初始化
        self.p_swap = 0.0
        self.p_spot = 0.0
        self.P_swap = 0.0
        
        # 动态参数
        self.current_params = [0.01, -0.01, 0.008, -0.008, 0.009, -0.009]
        self.param_update_interval = pd.Timedelta(hours=1)  # 每小时更新
        self.last_param_update = None
        
        # 结果记录
        self.pnl_history = []
        self.timestamps = []
        self.position_history = []
        self.param_history = []  # 记录参数变化
        
        # 原始数据
        self.raw_df = None

    def load_data(self):
        data_dir = Path("./backtest/data")
        filename = f"backtest_{self.symbol}_{self.start_time}_{self.end_time}.csv"
        filepath = data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        self.raw_df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        self.raw_df = self.raw_df.sort_index()
        
        # 确保所有需要的列都存在
        required_columns = [
            'spot_bid0_price', 'spot_ask0_price', 'spot_bid0_amount', 'spot_ask0_amount',
            'swap_bid0_price', 'swap_ask0_price', 'swap_bid0_amount', 'swap_ask0_amount',
            'basis1_price', 'basis2_price', 'basis1_volume', 'basis2_volume'
        ]
        
        for col in required_columns:
            if col not in self.raw_df.columns:
                self.raw_df[col] = np.nan

    def update_trading_params(self, current_time):
        """每小时更新交易参数"""
        if (self.last_param_update is None or 
            current_time - self.last_param_update >= self.param_update_interval):
            
            # 获取过去1小时的数据
            start_time = current_time - pd.Timedelta(hours=1)
            historical_data = self.raw_df[start_time:current_time]
            
            if len(historical_data) > 0:
                new_params = calculate_dynamic_trading_params(
                    historical_data, self.a, self.b,
                    self.c_t_swap, self.c_t_spot, self.c_m_swap, self.c_m_spot
                )
                self.current_params = new_params
                self.last_param_update = current_time
                
                # 记录参数变化
                self.param_history.append({
                    'timestamp': current_time,
                    'params': new_params.copy()
                })
                
                # print(f"Updated params at {current_time}: {new_params}")

    def get_current_params(self):
        """获取当前交易参数"""
        return self.current_params

    def get_price_with_slippage(self, price_col, next_price_col, timestamp):
        if timestamp not in self.raw_df.index:
            return np.nan
        
        current_price = self.raw_df.loc[timestamp, price_col]
        if pd.isna(current_price):
            return np.nan
            
        try:
            next_price = self.raw_df.loc[timestamp, next_price_col]
            if pd.isna(next_price):
                next_price = current_price
        except:
            next_price = current_price
        
        return self.beta * current_price + (1 - self.beta) * next_price

    def get_future_price(self, price_col, timestamp):
        if timestamp not in self.raw_df.index:
            return np.nan
        
        current_idx = self.raw_df.index.get_loc(timestamp)
        future_idx = min(current_idx + 1, len(self.raw_df) - 1)
        future_timestamp = self.raw_df.index[future_idx]
        
        return self.raw_df.loc[future_timestamp, price_col]

    def execute_trade_mode_a(self, timestamp):
        """市价单交易（使用动态参数）"""
        if timestamp not in self.raw_df.index:
            return False
        
        # 更新参数
        self.update_trading_params(timestamp)
        tt_open, tt_close, _, _, _, _ = self.get_current_params()
        
        row = self.raw_df.loc[timestamp]
        if pd.isna(row['basis1_price']) or pd.isna(row['basis2_price']):
            return False
        
        # 开仓
        if row['basis1_price'] > tt_open and self.P > 0:
            if pd.isna(row['spot_ask0_price']) or pd.isna(row['basis1_volume']) or row['spot_ask0_price'] <= 0:
                return False
                
            z = self.alpha * min(self.P / row['spot_ask0_price'], row['basis1_volume'])
            if z <= 0:
                return False
            
            spot_price = self.get_price_with_slippage('spot_ask0_price', 'spot_ask1_price', timestamp)
            swap_price = self.get_price_with_slippage('swap_bid0_price', 'swap_bid1_price', timestamp)
            
            if pd.isna(spot_price) or pd.isna(swap_price):
                return False
            
            self.p_swap -= z
            self.p_spot += z
            self.P_swap += swap_price * z
            self.P -= (spot_price + self.c_t_swap + self.c_t_spot) * z
            
            return True
        
        # 关仓
        elif row['basis2_price'] < tt_close and self.p_swap < 0:
            if pd.isna(row['spot_bid0_price']) or pd.isna(row['basis2_volume']) or row['spot_bid0_price'] <= 0:
                return False
                
            z = self.alpha * min(self.P / row['spot_bid0_price'], row['basis2_volume'])
            z = min(z, -self.p_swap)
            if z <= 0:
                return False
            
            spot_price = self.get_price_with_slippage('spot_bid0_price', 'spot_bid1_price', timestamp)
            swap_price = self.get_price_with_slippage('swap_ask0_price', 'swap_ask1_price', timestamp)
            
            if pd.isna(spot_price) or pd.isna(swap_price):
                return False
            
            self.p_swap += z
            self.p_spot -= z
            self.P_swap -= swap_price * z
            self.P += (spot_price - self.c_t_swap - self.c_t_spot) * z
            
            return True
        
        return False

    def execute_trade_mode_b(self, timestamp):
        """限价市价单交易（使用动态参数）"""
        if timestamp not in self.raw_df.index:
            return False
        
        self.update_trading_params(timestamp)
        _, _, mt_open, mt_close, _, _ = self.get_current_params()
        
        row = self.raw_df.loc[timestamp]
        if pd.isna(row['basis1_price']) or pd.isna(row['basis2_price']):
            return False
        
        # 开仓
        if row['basis1_price'] > mt_open and self.P > 0:
            if pd.isna(row['swap_ask0_price']) or pd.isna(row['basis1_volume']) or row['swap_ask0_price'] <= 0:
                return False
                
            z = self.alpha * min(self.P / row['swap_ask0_price'], row['basis1_volume'])
            if z <= 0:
                return False
            
            future_spot_ask0 = self.get_future_price('spot_ask0_price', timestamp)
            future_spot_ask1 = self.get_future_price('spot_ask1_price', timestamp)
            
            if pd.isna(future_spot_ask0) or pd.isna(future_spot_ask1):
                return False
            
            spot_price = self.beta * future_spot_ask0 + (1 - self.beta) * future_spot_ask1
            
            self.p_swap -= z
            self.p_spot += z
            self.P_swap += row['swap_bid0_price'] * z
            self.P -= (spot_price + self.c_m_swap + self.c_t_spot) * z
            
            return True
        
        # 关仓
        elif row['basis2_price'] < mt_close and self.p_swap < 0:
            if pd.isna(row['spot_bid0_price']) or pd.isna(row['basis2_volume']) or row['spot_bid0_price'] <= 0:
                return False
                
            z = self.alpha * min(self.P / row['spot_bid0_price'], row['basis2_volume'])
            z = min(z, -self.p_swap)
            if z <= 0:
                return False
            
            future_spot_bid0 = self.get_future_price('spot_bid0_price', timestamp)
            future_spot_bid1 = self.get_future_price('spot_bid1_price', timestamp)
            
            if pd.isna(future_spot_bid0) or pd.isna(future_spot_bid1):
                return False
            
            spot_price = self.beta * future_spot_bid0 + (1 - self.beta) * future_spot_bid1
            
            self.p_swap += z
            self.p_spot -= z
            self.P_swap -= row['swap_ask0_price'] * z
            self.P += (spot_price - self.c_m_swap - self.c_t_spot) * z
            
            return True
        
        return False

    def execute_trade_mode_c(self, timestamp):
        """市价限价单交易（使用动态参数）"""
        if timestamp not in self.raw_df.index:
            return False
        
        self.update_trading_params(timestamp)
        _, _, _, _, tm_open, tm_close = self.get_current_params()
        
        row = self.raw_df.loc[timestamp]
        if pd.isna(row['basis1_price']) or pd.isna(row['basis2_price']):
            return False
        
        # 开仓
        if row['basis1_price'] > tm_open and self.P > 0:
            if pd.isna(row['spot_ask0_price']) or pd.isna(row['basis1_volume']) or row['spot_ask0_price'] <= 0:
                return False
                
            z = self.alpha * min(self.P / row['spot_ask0_price'], row['basis1_volume'])
            if z <= 0:
                return False
            
            future_swap_bid0 = self.get_future_price('swap_bid0_price', timestamp)
            future_swap_bid1 = self.get_future_price('swap_bid1_price', timestamp)
            
            if pd.isna(future_swap_bid0) or pd.isna(future_swap_bid1):
                return False
            
            swap_price = self.beta * future_swap_bid0 + (1 - self.beta) * future_swap_bid1
            
            self.p_swap -= z
            self.p_spot += z
            self.P_swap += swap_price * z
            self.P -= (row['spot_ask0_price'] + self.c_t_swap + self.c_m_spot) * z
            
            return True
        
        # 关仓
        elif row['basis2_price'] < tm_close and self.p_swap < 0:
            if pd.isna(row['spot_bid0_price']) or pd.isna(row['basis2_volume']) or row['spot_bid0_price'] <= 0:
                return False
                
            z = self.alpha * min(self.P / row['spot_bid0_price'], row['basis2_volume'])
            z = min(z, -self.p_swap)
            if z <= 0:
                return False
            
            future_swap_ask0 = self.get_future_price('swap_ask0_price', timestamp)
            future_swap_ask1 = self.get_future_price('swap_ask1_price', timestamp)
            
            if pd.isna(future_swap_ask0) or pd.isna(future_swap_ask1):
                return False
            
            swap_price = self.beta * future_swap_ask0 + (1 - self.beta) * future_swap_ask1
            
            self.p_swap += z
            self.p_spot -= z
            self.P_swap -= swap_price * z
            self.P += (row['spot_bid0_price'] - self.c_t_swap - self.c_m_spot) * z
            
            return True
        
        return False

    def calculate_pnl(self, timestamp):
        if timestamp not in self.raw_df.index:
            return np.nan
        
        row = self.raw_df.loc[timestamp]
        spot_price = row['spot_bid0_price']
        swap_price = row['swap_ask0_price']
        
        if pd.isna(spot_price) or pd.isna(swap_price):
            return np.nan
        
        pnl = self.P + self.P_swap + self.p_spot * spot_price + self.p_swap * swap_price
        return pnl

    def run_backtest(self):
        self.load_data()
        
        for timestamp in self.raw_df.index:
            trade_executed = False
            
            # 每小时更新参数（在交易前）
            self.update_trading_params(timestamp)
            
            # 按 C, B, A 顺序检查交易
            if self.execute_trade_mode_c(timestamp):
                trade_executed = True
            elif self.execute_trade_mode_b(timestamp):
                trade_executed = True
            elif self.execute_trade_mode_a(timestamp):
                trade_executed = True
            
            # 记录结果
            pnl = self.calculate_pnl(timestamp)
            if not pd.isna(pnl):
                self.pnl_history.append(pnl)
                self.timestamps.append(timestamp)
                self.position_history.append({
                    'p_swap': self.p_swap,
                    'p_spot': self.p_spot,
                    'P': self.P
                })
        
        return self.pnl_history, self.timestamps

    def save_results(self):
        results_dir = Path("./backtest/data/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存 PnL
        pnl_df = pd.DataFrame({
            'timestamp': self.timestamps,
            'pnl': self.pnl_history
        })
        pnl_df.set_index('timestamp', inplace=True)
        pnl_filename = f"dynamic_pnl_{self.symbol}_{self.start_time}_{self.end_time}.csv"
        pnl_df.to_csv(results_dir / pnl_filename)
        
        # 保存参数历史
        if self.param_history:
            param_df = pd.DataFrame(self.param_history)
            param_df.set_index('timestamp', inplace=True)
            param_filename = f"dynamic_params_{self.symbol}_{self.start_time}_{self.end_time}.csv"
            param_df.to_csv(results_dir / param_filename)
        
        return pnl_df

    def plot_results_with_basis_and_params(self):
        """绘制 PnL、基差和动态参数"""
        results_dir = Path("./backtest/data/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备数据
        pnl_series = pd.Series(self.pnl_history, index=self.timestamps)
        basis_df = self.raw_df[['basis1_price', 'basis2_price']].loc[self.timestamps]
        
        # 创建子图 (3, 1)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # PnL 图
        ax1.plot(pnl_series.index, pnl_series.values, label=f'{self.symbol} PnL', color='blue', linewidth=2)
        ax1.set_ylabel('PnL')
        ax1.set_title(f'Dynamic Parameters Backtest - {self.symbol}', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 基差图
        ax2.plot(basis_df.index, basis_df['basis1_price'], 
                label='Basis1 (log(swap_bid) - log(spot_ask))', 
                color='red', alpha=0.8)
        ax2.plot(basis_df.index, basis_df['basis2_price'], 
                label='Basis2 (log(swap_ask) - log(spot_bid))', 
                color='orange', alpha=0.8)
        ax2.set_ylabel('Basis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 参数变化图（只显示 tt_open 和 tt_close 作为示例）
        if self.param_history:
            param_df = pd.DataFrame(self.param_history)
            param_df.set_index('timestamp', inplace=True)
            
            # 只绘制有数据的时间点
            valid_params = param_df[param_df.index.isin(self.timestamps)]
            if len(valid_params) > 0:
                ax3.plot(valid_params.index, [p[0] for p in valid_params['params']], 
                        label='tt_open', marker='o', markersize=3)
                ax3.plot(valid_params.index, [p[1] for p in valid_params['params']], 
                        label='tt_close', marker='o', markersize=3)
                ax3.set_ylabel('Trading Parameters')
                ax3.set_xlabel('Time')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_filename = f"dynamic_pnl_basis_params_{self.symbol}_{self.start_time}_{self.end_time}.png"
        plt.savefig(results_dir / plot_filename, dpi=150, bbox_inches='tight')
        plt.close()

def run_dynamic_backtest(symbol, start_time, end_time, a=0.001, b=0.0008):
    """运行动态参数回测"""
    print(f"Running dynamic backtest for {symbol} from {start_time} to {end_time}")
    
    backtest = DynamicBasisArbitrageBacktest(symbol, start_time, end_time, a, b)
    pnl_history, timestamps = backtest.run_backtest()
    backtest.save_results()
    backtest.plot_results_with_basis_and_params()
    
    print(f"Final PnL: {pnl_history[-1]:.2f}")
    return pnl_history

def main():
    symbol = "AVAX"
    a, b = 0.001, 0.0008  # 可调整的控制参数
    
    # time_periods = [
    #     ("20251001", "20251007"),
    #     ("20251014", "20251020"), 
    #     ("20251022", "20251028")
    # ]

    time_periods = [
        ("20251001", "20251007")
    ]    
    
    for start_time, end_time in time_periods:
        try:
            run_dynamic_backtest(symbol, start_time, end_time, a, b)
        except Exception as e:
            print(f"Error in dynamic backtest {start_time}-{end_time}: {e}")
            continue

if __name__ == "__main__":
    main()