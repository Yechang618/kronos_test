# backtest/strategy_backtest.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

class BasisArbitrageBacktest:
    def __init__(self, symbol, start_time, end_time, params):
        """
        Initialize backtest parameters
        
        params: [tt_open, tt_close, mt_open, mt_close, tm_open, tm_close]
        """
        self.symbol = symbol
        self.start_time = start_time
        self.end_time = end_time
        self.params = params
        
        # Backtest parameters
        self.P = 1000.0  # Initial capital
        self.alpha = 0.1  # Trading ratio
        self.beta = 0.5   # Slippage ratio
        self.dt = 0.01    # Trading delay (seconds)
        
        # Trading costs
        self.c_t_swap = 0.000153    # Market order futures cost
        self.c_t_spot = 0.0001725   # Market order spot cost
        self.c_m_swap = 0.0         # Limit order futures cost
        self.c_m_spot = 0.0000825   # Limit order spot cost
        
        # Position initialization
        self.p_swap = 0.0    # Futures position (negative means short)
        self.p_spot = 0.0    # Spot position
        self.P_swap = 0.0    # Futures cost
        
        # Unpack parameters
        self.tt_open, self.tt_close, self.mt_open, self.mt_close, self.tm_open, self.tm_close = params
        
        # Results tracking
        self.pnl_history = []
        self.timestamps = []
        self.position_history = []
        
    def load_data(self):
        """Load data"""
        data_dir = Path("./backtest/data")
        filename = f"backtest_{self.symbol}_{self.start_time}_{self.end_time}.csv"
        filepath = data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        self.df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        self.df = self.df.sort_index()
        
        # Ensure all required columns exist
        required_columns = [
            'spot_bid0_price', 'spot_ask0_price', 'spot_bid0_amount', 'spot_ask0_amount',
            'swap_bid0_price', 'swap_ask0_price', 'swap_bid0_amount', 'swap_ask0_amount',
            'basis1_price', 'basis2_price', 'basis1_volume', 'basis2_volume'
        ]
        
        for col in required_columns:
            if col not in self.df.columns:
                self.df[col] = np.nan
    
    def get_price_with_slippage(self, price_col, next_price_col, timestamp):
        """Get price with slippage"""
        if timestamp not in self.df.index:
            return np.nan
        
        current_price = self.df.loc[timestamp, price_col]
        if pd.isna(current_price):
            return np.nan
            
        # Get next price level
        try:
            next_price = self.df.loc[timestamp, next_price_col]
            if pd.isna(next_price):
                next_price = current_price
        except:
            next_price = current_price
        
        return self.beta * current_price + (1 - self.beta) * next_price
    
    def get_future_price(self, price_col, timestamp):
        """Get price at future dt seconds"""
        if timestamp not in self.df.index:
            return np.nan
        
        current_idx = self.df.index.get_loc(timestamp)
        future_idx = min(current_idx + 1, len(self.df) - 1)
        future_timestamp = self.df.index[future_idx]
        
        return self.df.loc[future_timestamp, price_col]
    
    def execute_trade_mode_a(self, timestamp):
        """Execute market order trading (Mode A)"""
        if timestamp not in self.df.index:
            return False
        
        row = self.df.loc[timestamp]
        if pd.isna(row['basis1_price']) or pd.isna(row['basis2_price']):
            return False
        
        # Open position condition
        if row['basis1_price'] > self.tt_open and self.P > 0:
            if pd.isna(row['spot_ask0_price']) or pd.isna(row['basis1_volume']) or row['spot_ask0_price'] <= 0:
                return False
                
            z = self.alpha * min(self.P / row['spot_ask0_price'], row['basis1_volume'])
            if z <= 0:
                return False
            
            # Calculate prices with slippage
            spot_price = self.get_price_with_slippage('spot_ask0_price', 'spot_ask1_price', timestamp)
            swap_price = self.get_price_with_slippage('swap_bid0_price', 'swap_bid1_price', timestamp)
            
            if pd.isna(spot_price) or pd.isna(swap_price):
                return False
            
            # Execute opening
            self.p_swap -= z
            self.p_spot += z
            self.P_swap += swap_price * z
            self.P -= (spot_price + self.c_t_swap + self.c_t_spot) * z
            
            return True
        
        # Close position condition
        elif row['basis2_price'] < self.tt_close and self.p_swap < 0:
            if pd.isna(row['spot_bid0_price']) or pd.isna(row['basis2_volume']) or row['spot_bid0_price'] <= 0:
                return False
                
            z = self.alpha * min(self.P / row['spot_bid0_price'], row['basis2_volume'])
            z = min(z, -self.p_swap)  # Cannot close more than current position
            if z <= 0:
                return False
            
            # Calculate prices with slippage
            spot_price = self.get_price_with_slippage('spot_bid0_price', 'spot_bid1_price', timestamp)
            swap_price = self.get_price_with_slippage('swap_ask0_price', 'swap_ask1_price', timestamp)
            
            if pd.isna(spot_price) or pd.isna(swap_price):
                return False
            
            # Execute closing
            self.p_swap += z
            self.p_spot -= z
            self.P_swap -= swap_price * z
            self.P += (spot_price - self.c_t_swap - self.c_t_spot) * z
            
            return True
        
        return False
    
    def execute_trade_mode_b(self, timestamp):
        """Execute limit-market order trading (Mode B)"""
        if timestamp not in self.df.index:
            return False
        
        row = self.df.loc[timestamp]
        if pd.isna(row['basis1_price']) or pd.isna(row['basis2_price']):
            return False
        
        # Open position condition
        if row['basis1_price'] > self.mt_open and self.P > 0:
            if pd.isna(row['swap_ask0_price']) or pd.isna(row['basis1_volume']) or row['swap_ask0_price'] <= 0:
                return False
                
            z = self.alpha * min(self.P / row['swap_ask0_price'], row['basis1_volume'])
            if z <= 0:
                return False
            
            # Future spot prices
            future_spot_ask0 = self.get_future_price('spot_ask0_price', timestamp)
            future_spot_ask1 = self.get_future_price('spot_ask1_price', timestamp)
            
            if pd.isna(future_spot_ask0) or pd.isna(future_spot_ask1):
                return False
            
            spot_price = self.beta * future_spot_ask0 + (1 - self.beta) * future_spot_ask1
            
            # Execute opening
            self.p_swap -= z
            self.p_spot += z
            self.P_swap += row['swap_bid0_price'] * z
            self.P -= (spot_price + self.c_m_swap + self.c_t_spot) * z
            
            return True
        
        # Close position condition
        elif row['basis2_price'] < self.mt_close and self.p_swap < 0:
            if pd.isna(row['spot_bid0_price']) or pd.isna(row['basis2_volume']) or row['spot_bid0_price'] <= 0:
                return False
                
            z = self.alpha * min(self.P / row['spot_bid0_price'], row['basis2_volume'])
            z = min(z, -self.p_swap)
            if z <= 0:
                return False
            
            # Future spot prices
            future_spot_bid0 = self.get_future_price('spot_bid0_price', timestamp)
            future_spot_bid1 = self.get_future_price('spot_bid1_price', timestamp)
            
            if pd.isna(future_spot_bid0) or pd.isna(future_spot_bid1):
                return False
            
            spot_price = self.beta * future_spot_bid0 + (1 - self.beta) * future_spot_bid1
            
            # Execute closing
            self.p_swap += z
            self.p_spot -= z
            self.P_swap -= row['swap_ask0_price'] * z
            self.P += (spot_price - self.c_m_swap - self.c_t_spot) * z
            
            return True
        
        return False
    
    def execute_trade_mode_c(self, timestamp):
        """Execute market-limit order trading (Mode C)"""
        if timestamp not in self.df.index:
            return False
        
        row = self.df.loc[timestamp]
        if pd.isna(row['basis1_price']) or pd.isna(row['basis2_price']):
            return False
        
        # Open position condition
        if row['basis1_price'] > self.tm_open and self.P > 0:
            if pd.isna(row['spot_ask0_price']) or pd.isna(row['basis1_volume']) or row['spot_ask0_price'] <= 0:
                return False
                
            z = self.alpha * min(self.P / row['spot_ask0_price'], row['basis1_volume'])
            if z <= 0:
                return False
            
            # Future futures prices
            future_swap_bid0 = self.get_future_price('swap_bid0_price', timestamp)
            future_swap_bid1 = self.get_future_price('swap_bid1_price', timestamp)
            
            if pd.isna(future_swap_bid0) or pd.isna(future_swap_bid1):
                return False
            
            swap_price = self.beta * future_swap_bid0 + (1 - self.beta) * future_swap_bid1
            
            # Execute opening
            self.p_swap -= z
            self.p_spot += z
            self.P_swap += swap_price * z
            self.P -= (row['spot_ask0_price'] + self.c_t_swap + self.c_m_spot) * z
            
            return True
        
        # Close position condition
        elif row['basis2_price'] < self.tm_close and self.p_swap < 0:
            if pd.isna(row['spot_bid0_price']) or pd.isna(row['basis2_volume']) or row['spot_bid0_price'] <= 0:
                return False
                
            z = self.alpha * min(self.P / row['spot_bid0_price'], row['basis2_volume'])
            z = min(z, -self.p_swap)
            if z <= 0:
                return False
            
            # Future futures prices
            future_swap_ask0 = self.get_future_price('swap_ask0_price', timestamp)
            future_swap_ask1 = self.get_future_price('swap_ask1_price', timestamp)
            
            if pd.isna(future_swap_ask0) or pd.isna(future_swap_ask1):
                return False
            
            swap_price = self.beta * future_swap_ask0 + (1 - self.beta) * future_swap_ask1
            
            # Execute closing
            self.p_swap += z
            self.p_spot -= z
            self.P_swap -= swap_price * z
            self.P += (row['spot_bid0_price'] - self.c_t_swap - self.c_m_spot) * z
            
            return True
        
        return False
    
    def calculate_pnl(self, timestamp):
        """Calculate current PnL"""
        if timestamp not in self.df.index:
            return np.nan
        
        row = self.df.loc[timestamp]
        spot_price = row['spot_bid0_price']
        swap_price = row['swap_ask0_price']
        
        if pd.isna(spot_price) or pd.isna(swap_price):
            return np.nan
        
        # PnL = Cash + Futures value + Spot value
        pnl = self.P + self.P_swap + self.p_spot * spot_price + self.p_swap * swap_price
        return pnl
    
    def run_backtest(self):
        """Run backtest"""
        self.load_data()
        
        # Iterate through each timestamp
        for timestamp in self.df.index:
            trade_executed = False
            
            # Check trading opportunities in order C, B, A
            if self.execute_trade_mode_c(timestamp):
                trade_executed = True
            elif self.execute_trade_mode_b(timestamp):
                trade_executed = True
            elif self.execute_trade_mode_a(timestamp):
                trade_executed = True
            
            # Record PnL
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
        """Save results"""
        results_dir = Path("./backtest/data/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save PnL history
        pnl_df = pd.DataFrame({
            'timestamp': self.timestamps,
            'pnl': self.pnl_history
        })
        pnl_df.set_index('timestamp', inplace=True)
        
        filename = f"pnl_{self.symbol}_{self.start_time}_{self.end_time}.csv"
        pnl_df.to_csv(results_dir / filename)
        
        # Save trade records
        trade_df = pd.DataFrame(self.position_history)
        trade_df['timestamp'] = self.timestamps
        trade_df = trade_df.set_index('timestamp')
        
        trade_filename = f"trades_{self.symbol}_{self.start_time}_{self.end_time}.csv"
        trade_df.to_csv(results_dir / trade_filename)
        
        return pnl_df, trade_df
    
    def plot_results(self):
        """Visualize results"""
        results_dir = Path("./backtest/data/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.timestamps, self.pnl_history, label=f'{self.symbol} PnL')
        plt.title(f'Basis Arbitrage Backtest Results - {self.symbol}')
        plt.xlabel('Time')
        plt.ylabel('PnL')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plot_filename = f"pnl_plot_{self.symbol}_{self.start_time}_{self.end_time}.png"
        plt.savefig(results_dir / plot_filename, dpi=150)
        plt.close()

def run_single_backtest(symbol, start_time, end_time, params):
    """Run single backtest"""
    print(f"Running backtest for {symbol} from {start_time} to {end_time}")
    
    backtest = BasisArbitrageBacktest(symbol, start_time, end_time, params)
    pnl_history, timestamps = backtest.run_backtest()
    pnl_df, trade_df = backtest.save_results()
    backtest.plot_results()
    
    print(f"Final PnL: {pnl_history[-1]:.2f}")
    print(f"Max PnL: {max(pnl_history):.2f}")
    print(f"Min PnL: {min(pnl_history):.2f}")
    
    return pnl_df, trade_df

def main():
    # Parameters setup
    symbol = "AVAX"
    params = [0.01, -0.01, 0.008, -0.008, 0.009, -0.009]  # [tt_open, tt_close, mt_open, mt_close, tm_open, tm_close]
    
    # Backtest time periods
    time_periods = [
        ("20251001", "20251007"),
        ("20251014", "20251020"), 
        ("20251022", "20251028")
    ]
    
    all_results = []
    
    for start_time, end_time in time_periods:
        try:
            pnl_df, trade_df = run_single_backtest(symbol, start_time, end_time, params)
            all_results.append(pnl_df)
        except Exception as e:
            print(f"Error in backtest {start_time}-{end_time}: {e}")
            continue
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, axis=0)
        combined_df = combined_df.sort_index()
        
        # Save combined results
        results_dir = Path("./backtest/data/results")
        combined_df.to_csv(results_dir / f"combined_pnl_{symbol}.csv")
        
        # Plot combined results
        plt.figure(figsize=(12, 6))
        plt.plot(combined_df.index, combined_df['pnl'], label=f'{symbol} Combined PnL')
        plt.title(f'Basis Arbitrage Backtest Results - {symbol} (All Periods)')
        plt.xlabel('Time')
        plt.ylabel('PnL')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(results_dir / f"combined_pnl_plot_{symbol}.png", dpi=150)
        plt.close()

if __name__ == "__main__":
    main()