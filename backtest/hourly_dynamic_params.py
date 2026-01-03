# backtest/hourly_dynamic_params.py
import pandas as pd
import numpy as np

def calculate_hourly_trading_params(df, a, b, 
                                  c_t_swap=0.000153, 
                                  c_t_spot=0.0001725, 
                                  c_m_swap=0.0, 
                                  c_m_spot=0.0000825):
    """
    基于过去1小时的订单簿数据计算动态交易参数
    
    Parameters:
    -----------
    df : pandas.DataFrame
        包含订单簿数据的DataFrame，必须包含以下列：
        - basis1_price, basis2_price
        - spot_bid0_price, spot_ask0_price
    a : float
        市价单控制参数
    b : float  
        限价单控制参数
    c_t_swap, c_t_spot, c_m_swap, c_m_spot : float
        交易成本参数
        
    Returns:
    --------
    list: [tt_open, tt_close, mt_open, mt_close, tm_open, tm_close]
    """
    # 计算基差均值
    avg_basis1 = df['basis1_price'].mean()
    avg_basis2 = df['basis2_price'].mean()
    
    # 计算现货价格均值
    avg_spot_bid0 = df['spot_bid0_price'].mean()
    avg_spot_ask0 = df['spot_ask0_price'].mean()
    
    # 避免除零错误和无效价格
    if (avg_spot_ask0 <= 0 or avg_spot_bid0 <= 0 or 
        pd.isna(avg_basis1) or pd.isna(avg_basis2)):
        # 返回默认保守参数
        return [0.01, -0.01, 0.008, -0.008, 0.009, -0.009]
    
    # 计算交易参数
    tt_open = avg_basis1 + a + (c_t_swap + c_t_spot) / avg_spot_ask0
    tt_close = avg_basis2 - a - (c_t_swap - c_t_spot) / avg_spot_bid0
    
    mt_open = avg_basis1 + b + (c_m_swap + c_t_spot) / avg_spot_ask0
    mt_close = avg_basis2 - b - (c_m_swap - c_t_spot) / avg_spot_bid0
    
    tm_open = avg_basis1 + b + (c_t_swap + c_m_spot) / avg_spot_ask0
    tm_close = avg_basis2 - b - (c_t_swap - c_m_spot) / avg_spot_bid0
    
    return [tt_open, tt_close, mt_open, mt_close, tm_open, tm_close]