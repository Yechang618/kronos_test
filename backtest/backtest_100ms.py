# backtest/strategy_backtest_100ms.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class BasisArbitrageBacktest100ms:
    def __init__(self, symbol, start_time, end_time, params):
        """
        初始化回测参数（100ms 版本）
        
        params: [tt_open, tt_close, mt_open, mt_close, tm_open, tm_close]
        """
        self.symbol = symbol
        self.start_time = start_time
        self.end_time = end_time
        self.params = params
        
        # 回测参数
        self.P = 1000.0
        self.alpha = 0.1
        self.beta = 0.5
        self.dt = 0.01  # 10ms
        
        # 交易成本
        self.c_t_swap = 0.000153
        self.c_t_spot = 0.0001725
        self.c_m_swap = 0.0
        self.c_m_spot = 0.0000825
        
        # 仓位初始化
        self.p_swap = 0.0
        self.p_spot = 0.0
        self.P_swap = 0.0
        
        # 参数解包
        self.tt_open, self.tt_close, self.mt_open, self.mt_close, self.tm_open, self.tm_close = params
        
        # 结果记录
        self.pnl_history = []
        self.timestamps = []
        
        # 数据存储
        self.raw_df = None
        
    def load_data(self):
        """加载按日保存的100ms数据"""
        data_dir = Path("./backtest/data")
        
        # 解析日期范围
        start_dt = pd.Timestamp(self.start_time)
        end_dt = pd.Timestamp(self.end_time)
        date_range = pd.date_range(start_dt.date(), end_dt.date(), freq="D")
        
        daily_dfs = []
        for date in date_range:
            date_str = date.strftime("%Y-%m-%d")
            filename = f"backtest_{self.symbol}_{date_str}.csv"
            filepath = data_dir / filename
            
            if filepath.exists():
                df_daily = pd.read_csv(filepath, index_col=0, parse_dates=True)
                daily_dfs.append(df_daily)
            else:
                print(f"Warning: {filename} not found")
        
        if not daily_dfs:
            raise ValueError("No data files found")
        
        # 合并所有日期数据
        self.raw_df = pd.concat(daily_dfs, axis=0)
        self.raw_df = self.raw_df.sort_index()
        
        # 切片到指定时间范围
        self.raw_df = self.raw_df[self.start_time:self.end_time]
        
        # 确保所有需要的列都存在
        required_columns = [
            'spot_bid0_price', 'spot_ask0_price', 'spot_bid0_amount', 'spot_ask0_amount',
            'swap_bid0_price', 'swap_ask0_price', 'swap_bid0_amount', 'swap_ask0_amount'
        ]
        
        for col in required_columns:
            if col not in self.raw_df.columns:
                self.raw_df[col] = np.nan

    def get_price_with_slippage(self, price_col, next_price_col, timestamp):
        """获取带滑点的价格"""
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

    def get_future_price(self, price_col, timestamp, dt_steps=1):
        """获取未来dt秒的价格（100ms粒度）"""
        if timestamp not in self.raw_df.index:
            return np.nan
            
        current_idx = self.raw_df.index.get_loc(timestamp)
        future_idx = min(current_idx + dt_steps, len(self.raw_df) - 1)
        future_timestamp = self.raw_df.index[future_idx]
        
        return self.raw_df.loc[future_timestamp, price_col]

    def execute_trade_mode_a(self, timestamp):
        """市价单交易（100ms版本）"""
        if timestamp not in self.raw_df.index:
            return False
            
        row = self.raw_df.loc[timestamp]
        if pd.isna(row['basis1_price']) or pd.isna(row['basis2_price']):
            return False
            
        # 开仓
        if row['basis1_price'] > self.tt_open and self.P > 0:
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
        elif row['basis2_price'] < self.tt_close and self.p_swap < 0:
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
        """限价市价单交易（100ms版本）"""
        if timestamp not in self.raw_df.index:
            return False
            
        row = self.raw_df.loc[timestamp]
        if pd.isna(row['basis1_price']) or pd.isna(row['basis2_price']):
            return False
            
        # 开仓
        if row['basis1_price'] > self.mt_open and self.P > 0:
            if pd.isna(row['swap_ask0_price']) or pd.isna(row['basis1_volume']) or row['swap_ask0_price'] <= 0:
                return False
                
            z = self.alpha * min(self.P / row['swap_ask0_price'], row['basis1_volume'])
            if z <= 0:
                return False
                
            # 未来现货价格（100ms后）
            future_spot_ask0 = self.get_future_price('spot_ask0_price', timestamp, dt_steps=1)
            future_spot_ask1 = self.get_future_price('spot_ask1_price', timestamp, dt_steps=1)
            
            if pd.isna(future_spot_ask0) or pd.isna(future_spot_ask1):
                return False
                
            spot_price = self.beta * future_spot_ask0 + (1 - self.beta) * future_spot_ask1
            self.p_swap -= z
            self.p_spot += z
            self.P_swap += row['swap_bid0_price'] * z
            self.P -= (spot_price + self.c_m_swap + self.c_t_spot) * z
            return True
            
        # 关仓
        elif row['basis2_price'] < self.mt_close and self.p_swap < 0:
            if pd.isna(row['spot_bid0_price']) or pd.isna(row['basis2_volume']) or row['spot_bid0_price'] <= 0:
                return False
                
            z = self.alpha * min(self.P / row['spot_bid0_price'], row['basis2_volume'])
            z = min(z, -self.p_swap)
            if z <= 0:
                return False
                
            future_spot_bid0 = self.get_future_price('spot_bid0_price', timestamp, dt_steps=1)
            future_spot_bid1 = self.get_future_price('spot_bid1_price', timestamp, dt_steps=1)
            
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
        """市价限价单交易（100ms版本）"""
        if timestamp not in self.raw_df.index:
            return False
            
        row = self.raw_df.loc[timestamp]
        if pd.isna(row['basis1_price']) or pd.isna(row['basis2_price']):
            return False
            
        # 开仓
        if row['basis1_price'] > self.tm_open and self.P > 0:
            if pd.isna(row['spot_ask0_price']) or pd.isna(row['basis1_volume']) or row['spot_ask0_price'] <= 0:
                return False
                
            z = self.alpha * min(self.P / row['spot_ask0_price'], row['basis1_volume'])
            if z <= 0:
                return False
                
            future_swap_bid0 = self.get_future_price('swap_bid0_price', timestamp, dt_steps=1)
            future_swap_bid1 = self.get_future_price('swap_bid1_price', timestamp, dt_steps=1)
            
            if pd.isna(future_swap_bid0) or pd.isna(future_swap_bid1):
                return False
                
            swap_price = self.beta * future_swap_bid0 + (1 - self.beta) * future_swap_bid1
            self.p_swap -= z
            self.p_spot += z
            self.P_swap += swap_price * z
            self.P -= (row['spot_ask0_price'] + self.c_t_swap + self.c_m_spot) * z
            return True
            
        # 关仓
        elif row['basis2_price'] < self.tm_close and self.p_swap < 0:
            if pd.isna(row['spot_bid0_price']) or pd.isna(row['basis2_volume']) or row['spot_bid0_price'] <= 0:
                return False
                
            z = self.alpha * min(self.P / row['spot_bid0_price'], row['basis2_volume'])
            z = min(z, -self.p_swap)
            if z <= 0:
                return False
                
            future_swap_ask0 = self.get_future_price('swap_ask0_price', timestamp, dt_steps=1)
            future_swap_ask1 = self.get_future_price('swap_ask1_price', timestamp, dt_steps=1)
            
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
        """计算PnL"""
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
        """运行100ms回测"""
        self.load_data()
        
        # 按秒分组
        df_grouped = self.raw_df.groupby(pd.Grouper(freq='1s'))
        
        for second_timestamp, second_group in df_grouped:
            if second_group.empty:
                continue
                
            # 在每个秒区间内，逐100ms检查交易机会
            trade_executed = False
            for timestamp in second_group.index:
                # 按 C, B, A 顺序检查
                if self.execute_trade_mode_c(timestamp):
                    trade_executed = True
                    break  # 执行一次后跳出100ms循环
                elif self.execute_trade_mode_b(timestamp):
                    trade_executed = True
                    break
                elif self.execute_trade_mode_a(timestamp):
                    trade_executed = True
                    break
            
            # 记录秒结束时的PnL
            last_timestamp = second_group.index[-1]
            pnl = self.calculate_pnl(last_timestamp)
            if not pd.isna(pnl):
                self.pnl_history.append(pnl)
                self.timestamps.append(last_timestamp)

    def save_results(self):
        """保存结果"""
        results_dir = Path("./backtest/data/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        pnl_df = pd.DataFrame({
            'timestamp': self.timestamps,
            'pnl': self.pnl_history
        })
        pnl_df.set_index('timestamp', inplace=True)
        
        filename = f"pnl_100ms_{self.symbol}_{self.start_time}_{self.end_time}.csv"
        pnl_df.to_csv(results_dir / filename)
        return pnl_df

    def plot_results(self):
        """绘制结果"""
        results_dir = Path("./backtest/data/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.timestamps, self.pnl_history, label=f'{self.symbol} PnL (100ms)')
        plt.title(f'100ms Basis Arbitrage Backtest - {self.symbol}')
        plt.xlabel('Time')
        plt.ylabel('PnL')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(results_dir / f"pnl_100ms_{self.symbol}_{self.start_time}_{self.end_time}.png", dpi=150)
        plt.close()

def run_100ms_backtest(symbol, start_time, end_time, params):
    """运行100ms回测"""
    print(f"Running 100ms backtest for {symbol} from {start_time} to {end_time}")
    
    backtest = BasisArbitrageBacktest100ms(symbol, start_time, end_time, params)
    backtest.run_backtest()
    backtest.save_results()
    backtest.plot_results()
    
    print(f"Final PnL: {backtest.pnl_history[-1]:.2f}")
    return backtest.pnl_history

def main():
    symbol = "XRP"
    params = [0.01, -0.01, 0.008, -0.008, 0.009, -0.009]
    
    # 测试一个完整日期（100ms数据）
    start_time = "2025-10-01 00:00:00"
    end_time = "2025-10-01 23:59:59"
    
    run_100ms_backtest(symbol, start_time, end_time, params)

if __name__ == "__main__":
    main()