# realtime_prediction_v4.py
"""
实时Kronos预测系统 v2.3 (高频采集版)
- 每秒采集1次订单簿和资金费率数据（原为每分钟1次）
- 每10分钟计算1次K线（open=首basis_mid, close=尾basis_mid, high=max(basis_bid), low=min(basis_ask)）
- 冷启动自动加载历史K线，每10分钟自动保存
- 新增：API速率限制保护、交错采集、日志聚合
"""

import os
import sys
import time
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union
from abc import ABC, abstractmethod
import random

# 设置环境变量避免OMP警告
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 添加项目路径
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / 'core'))
sys.path.insert(0, str(root_dir / 'src'))

from config import Config
from KronosPredictor import KronosPredictor, DynamicSignalGenerator
from redis import Redis
import yaml
from util import DecimalEncoder
import message_bot as mb

config_path = "./core/config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    redis_config = yaml.safe_load(f)

# 配置参数
CONFIG = Config()
MAX_ROWS = 144
PRED_LENGTH = 6
MIN_KLINES_FOR_PREDICTION = 1
KLINE_SAVE_DIR = Path("./datasets/temp")
N_SAMPLES = CONFIG.n_samples
TEMPERATURE = 100

# 高频采集专用配置
COLLECTION_INTERVAL_SEC = 1  # 每秒采集1次
BUFFER_MAX_POINTS = 720  # 12分钟缓冲区（10分钟窗口+2分钟安全边际）
LOG_AGGREGATION_INTERVAL = 30  # 每30秒聚合输出1次日志

class BaseDataFetcher(ABC):
    """数据采集器抽象基类"""
    
    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode
        self.exchange_name = "Unknown"
        self.last_request_time = {}  # 用于API速率限制
    
    @abstractmethod
    def _standardize_symbol(self, symbol: str) -> Tuple[str, str]:
        """
        标准化符号格式
        返回: (现货符号, 合约符号)
        """
        pass
    
    @abstractmethod
    def fetch_orderbook(self, symbol: str, is_swap: bool = False) -> Optional[Dict]:
        """获取订单簿数据"""
        pass
    
    @abstractmethod
    def fetch_funding_rate(self, symbol: str) -> float:
        """获取资金费率"""
        pass
    
    @abstractmethod
    def fetch_index_price(self, symbol: str) -> float:
        """获取指数/标记价格"""
        pass
    
    def _mock_orderbook(self, symbol: str, is_swap: bool) -> Dict:
        """模拟订单簿数据（测试模式通用实现）"""
        base_price = {
            "BTCUSDT": 60000,
            "ETHUSDT": 3000,
            "TAOUSDT": 200,
            "TRXUSDT": 0.15,
            "XRPUSDT": 0.6,
            "ZECUSDT": 50,
            "LTCUSDT": 80,
            "BCHUSDT": 250,
            "EOSUSDT": 0.7
        }.get(symbol.replace("USDT", ""), 10000)
        
        spot_mid = base_price * (1 + np.random.randn() * 0.001)
        spread = base_price * 0.0005
        
        if is_swap:
            swap_mid = spot_mid * (1 + np.random.randn() * 0.0002)
            return {
                'bid0_price': swap_mid - spread/2,
                'bid0_amount': np.random.rand() * 100,
                'ask0_price': swap_mid + spread/2,
                'ask0_amount': np.random.rand() * 100
            }
        else:
            return {
                'bid0_price': spot_mid - spread/2,
                'bid0_amount': np.random.rand() * 100,
                'ask0_price': spot_mid + spread/2,
                'ask0_amount': np.random.rand() * 100
            }
    
    def _rate_limit_protection(self, symbol: str, min_interval: float = 0.1):
        """API速率限制保护（避免触发交易所限流）"""
        now = time.time()
        last_time = self.last_request_time.get(symbol, 0)
        elapsed = now - last_time
        
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            time.sleep(sleep_time)
        
        self.last_request_time[symbol] = time.time()

class BinanceDataFetcher(BaseDataFetcher):
    """Binance数据采集器（支持高频采集）"""
    
    def __init__(self, test_mode: bool = False):
        super().__init__(test_mode)
        self.exchange_name = "Binance"
        self.base_url_spot = "https://api.binance.com"
        self.base_url_swap = "https://fapi.binance.com"
        # Binance速率限制：现货1200次/分钟，合约2400次/分钟
        # 每秒采集7币种×2API=14次，远低于限制，但需交错采集
        self.symbol_order = CONFIG.symbol_list.copy()
        random.shuffle(self.symbol_order)  # 随机化初始顺序
    
    def _standardize_symbol(self, symbol: str) -> Tuple[str, str]:
        """标准化为Binance格式"""
        symbol = symbol.upper().strip()
        if not symbol.endswith('USDT'):
            if len(symbol) > 4 and any(suffix in symbol for suffix in ['BUSD', 'USDC', 'DAI']):
                spot_symbol = symbol
                swap_symbol = symbol
            else:
                spot_symbol = symbol + 'USDT'
                swap_symbol = symbol + 'USDT'
        else:
            spot_symbol = symbol
            swap_symbol = symbol
        return spot_symbol, swap_symbol
    
    def fetch_orderbook(self, symbol: str, is_swap: bool = False) -> Optional[Dict]:
        """获取Binance订单簿（带速率限制保护）"""
        try:
            # 速率限制保护：每符号至少间隔100ms
            self._rate_limit_protection(symbol, min_interval=0.1)
            
            spot_symbol, swap_symbol = self._standardize_symbol(symbol)
            symbol_std = swap_symbol if is_swap else spot_symbol
            
            if self.test_mode:
                return self._mock_orderbook(symbol_std, is_swap)
            
            # 关键修复：合约使用/fapi/v1/depth，现货使用/api/v3/depth
            if is_swap:
                url = f"{self.base_url_swap}/fapi/v1/depth"
            else:
                url = f"{self.base_url_spot}/api/v3/depth"
            
            params = {"symbol": symbol_std, "limit": 5}
            
            # 指数退避重试（最多3次）
            for attempt in range(3):
                try:
                    response = requests.get(url, params=params, timeout=3)  # 缩短超时时间
                    response.raise_for_status()
                    data = response.json()
                    
                    if 'bids' not in data or 'asks' not in data or len(data['bids']) == 0 or len(data['asks']) == 0:
                        raise ValueError(f"无效的订单簿响应: {data}")
                    
                    return {
                        'bid0_price': float(data['bids'][0][0]),
                        'bid0_amount': float(data['bids'][0][1]),
                        'ask0_price': float(data['asks'][0][0]),
                        'ask0_amount': float(data['asks'][0][1])
                    }
                except requests.exceptions.RequestException as e:
                    if attempt < 2:
                        time.sleep(0.2 * (2 ** attempt))  # 指数退避
                        continue
                    raise
            
        except Exception as e:
            symbol_std = self._standardize_symbol(symbol)[0]
            api_type = "合约" if is_swap else "现货"
            if "400" in str(e):
                print(f"[ERROR] {self.exchange_name} {api_type}订单簿获取失败 {symbol} → {symbol_std}: 无效符号")
            elif "429" in str(e) or "too many requests" in str(e).lower():
                print(f"[WARN] {self.exchange_name} API限流，自动退避中...")
                time.sleep(1.0)
            return None
    
    def fetch_funding_rate(self, symbol: str) -> float:
        """获取Binance资金费率（每10秒更新1次，避免频繁请求）"""
        try:
            # 资金费率每8小时更新1次，无需每秒请求
            # 使用缓存机制：每10秒更新1次
            now = time.time()
            cache_key = f"{symbol}_funding"
            if hasattr(self, '_funding_cache') and cache_key in self._funding_cache:
                cached_time, cached_value = self._funding_cache[cache_key]
                if now - cached_time < 10.0:  # 10秒内使用缓存
                    return cached_value
            
            _, swap_symbol = self._standardize_symbol(symbol)
            
            if self.test_mode:
                value = np.random.randn() * 0.0001
            else:
                url = f"{self.base_url_swap}/fapi/v1/fundingRate"
                params = {"symbol": swap_symbol, "limit": 1}
                
                response = requests.get(url, params=params, timeout=3)
                response.raise_for_status()
                data = response.json()
                value = float(data[0]['fundingRate']) if data else 0.0
            
            # 更新缓存
            if not hasattr(self, '_funding_cache'):
                self._funding_cache = {}
            self._funding_cache[cache_key] = (now, value)
            
            return value
            
        except Exception as e:
            symbol_std = self._standardize_symbol(symbol)[1]
            print(f"[ERROR] {self.exchange_name} 资金费率获取失败 {symbol} → {symbol_std}: {str(e)}")
            return 0.0
    
    def fetch_index_price(self, symbol: str) -> float:
        """获取Binance指数价格（使用标记价格）"""
        try:
            # 指数价格变化较慢，每5秒更新1次
            now = time.time()
            cache_key = f"{symbol}_index"
            if hasattr(self, '_index_cache') and cache_key in self._index_cache:
                cached_time, cached_value = self._index_cache[cache_key]
                if now - cached_time < 5.0:
                    return cached_value
            
            _, swap_symbol = self._standardize_symbol(symbol)
            
            if self.test_mode:
                spot_ob = self.fetch_orderbook(symbol, is_swap=False)
                if spot_ob:
                    return (spot_ob['bid0_price'] + spot_ob['ask0_price']) / 2
                return 10000.0
            
            url = f"{self.base_url_swap}/fapi/v1/premiumIndex"
            params = {"symbol": swap_symbol}
            
            response = requests.get(url, params=params, timeout=3)
            response.raise_for_status()
            data = response.json()
            value = float(data['markPrice'])
            
            # 更新缓存
            if not hasattr(self, '_index_cache'):
                self._index_cache = {}
            self._index_cache[cache_key] = (now, value)
            
            return value
            
        except Exception as e:
            symbol_std = self._standardize_symbol(symbol)[1]
            print(f"[WARNING] {self.exchange_name} 指数价格获取失败 {symbol} → {symbol_std}, 使用现货中间价替代: {str(e)}")
            spot_ob = self.fetch_orderbook(symbol, is_swap=False)
            if spot_ob:
                return (spot_ob['bid0_price'] + spot_ob['ask0_price']) / 2
            return 10000.0

class KuCoinDataFetcher(BaseDataFetcher):
    """KuCoin数据采集器（支持高频采集）"""
    
    def __init__(self, test_mode: bool = False):
        super().__init__(test_mode)
        self.exchange_name = "KuCoin"
        self.base_url_spot = "https://api.kucoin.com"
        self.base_url_swap = "https://api-futures.kucoin.com"
        self.symbol_order = CONFIG.symbol_list.copy()
        random.shuffle(self.symbol_order)
    
    def _standardize_symbol(self, symbol: str) -> Tuple[str, str]:
        """
        KuCoin符号映射规则:
        - 现货: BTC-USDT (带连字符)
        - 合约: BTCUSDTM (USDT-margined perpetual, 末尾加M)
        """
        symbol = symbol.upper().strip().replace("USDT", "")
        contract_base = "XBT" if symbol == "BTC" else symbol
        spot_symbol = f"{symbol}-USDT"
        swap_symbol = f"{contract_base}USDTM"
        return spot_symbol, swap_symbol
    
    def fetch_orderbook(self, symbol: str, is_swap: bool = False) -> Optional[Dict]:
        """获取KuCoin订单簿（带速率限制保护）"""
        try:
            self._rate_limit_protection(symbol, min_interval=0.15)  # KuCoin限制更严格
            
            spot_symbol, swap_symbol = self._standardize_symbol(symbol)
            symbol_std = swap_symbol if is_swap else spot_symbol
            
            if self.test_mode:
                mock_symbol = symbol + "USDT" if not symbol.endswith("USDT") else symbol
                return self._mock_orderbook(mock_symbol, is_swap)
            
            if is_swap:
                url = f"{self.base_url_swap}/api/v1/level2/snapshot"
                params = {"symbol": symbol_std}
            else:
                url = f"{self.base_url_spot}/api/v1/market/orderbook/level2_20"
                params = {"symbol": symbol_std}
            
            for attempt in range(3):
                try:
                    response = requests.get(url, params=params, timeout=3)
                    response.raise_for_status()
                    data = response.json()
                    
                    if data.get('code') != '200000':
                        raise ValueError(f"KuCoin API error: {data.get('msg', 'Unknown error')}")
                    
                    bids = data['data']['bids'] if is_swap else data['data']['bids']
                    asks = data['data']['asks'] if is_swap else data['data']['asks']
                    
                    if not bids or not asks:
                        raise ValueError(f"空订单簿: {symbol_std}")
                    
                    return {
                        'bid0_price': float(bids[0][0]),
                        'bid0_amount': float(bids[0][1]),
                        'ask0_price': float(asks[0][0]),
                        'ask0_amount': float(asks[0][1])
                    }
                except requests.exceptions.RequestException as e:
                    if attempt < 2:
                        time.sleep(0.3 * (2 ** attempt))
                        continue
                    raise
            
        except Exception as e:
            symbol_std = self._standardize_symbol(symbol)[0 if not is_swap else 1]
            api_type = "合约" if is_swap else "现货"
            if "429" in str(e) or "rate limit" in str(e).lower():
                print(f"[WARN] {self.exchange_name} API限流，自动退避中...")
                time.sleep(2.0)
            return None
    
    def fetch_funding_rate(self, symbol: str) -> float:
        """获取KuCoin资金费率（缓存机制）"""
        try:
            now = time.time()
            cache_key = f"{symbol}_funding_kc"
            if hasattr(self, '_funding_cache') and cache_key in self._funding_cache:
                cached_time, cached_value = self._funding_cache[cache_key]
                if now - cached_time < 10.0:
                    return cached_value
            
            _, swap_symbol = self._standardize_symbol(symbol)
            
            if self.test_mode:
                value = np.random.randn() * 0.0001
            else:
                url = f"{self.base_url_swap}/api/v1/funding-rate/{swap_symbol}/current"
                response = requests.get(url, timeout=3)
                response.raise_for_status()
                data = response.json()
                value = float(data['data']['value']) if data.get('code') == '200000' else 0.0
            
            if not hasattr(self, '_funding_cache'):
                self._funding_cache = {}
            self._funding_cache[cache_key] = (now, value)
            
            return value
            
        except Exception as e:
            symbol_std = self._standardize_symbol(symbol)[1]
            print(f"[ERROR] {self.exchange_name} 资金费率获取失败 {symbol} → {symbol_std}: {str(e)}")
            return 0.0
    
    def fetch_index_price(self, symbol: str) -> float:
        """获取KuCoin标记价格（缓存机制）"""
        try:
            now = time.time()
            cache_key = f"{symbol}_index_kc"
            if hasattr(self, '_index_cache') and cache_key in self._index_cache:
                cached_time, cached_value = self._index_cache[cache_key]
                if now - cached_time < 5.0:
                    return cached_value
            
            _, swap_symbol = self._standardize_symbol(symbol)
            
            if self.test_mode:
                spot_ob = self.fetch_orderbook(symbol, is_swap=False)
                if spot_ob:
                    return (spot_ob['bid0_price'] + spot_ob['ask0_price']) / 2
                return 10000.0
            
            url = f"{self.base_url_swap}/api/v1/mark-price/{swap_symbol}/current"
            response = requests.get(url, timeout=3)
            response.raise_for_status()
            data = response.json()
            value = float(data['data']['value']) if data.get('code') == '200000' else 10000.0
            
            if not hasattr(self, '_index_cache'):
                self._index_cache = {}
            self._index_cache[cache_key] = (now, value)
            
            return value
            
        except Exception as e:
            symbol_std = self._standardize_symbol(symbol)[1]
            print(f"[WARNING] {self.exchange_name} 标记价格获取失败 {symbol} → {symbol_std}, 使用现货中间价替代: {str(e)}")
            spot_ob = self.fetch_orderbook(symbol, is_swap=False)
            if spot_ob:
                return (spot_ob['bid0_price'] + spot_ob['ask0_price']) / 2
            return 10000.0

class RealtimeKlineManager:
    """实时10分钟K线管理器（支持高频数据采集）"""
    
    def __init__(self, max_rows: int = MAX_ROWS, save_dir: Path = KLINE_SAVE_DIR, exchange: str = "binance"):
        self.max_rows = max_rows
        self.save_dir = save_dir
        self.exchange = exchange.lower()
        self.kline_data: Dict[str, pd.DataFrame] = {}
        self.last_close: Dict[str, float] = {}
        self.raw_data_buffer: Dict[str, List[Dict]] = {}
        self.last_log_time: Dict[str, datetime] = {}  # 用于日志聚合
        
        # 初始化保存目录（按交易所区分）
        exchange_dir = save_dir / self.exchange
        exchange_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = exchange_dir
        
        # 初始化每个symbol的数据结构
        for symbol in CONFIG.symbol_list:
            self.kline_data[symbol] = pd.DataFrame(
                columns=['open', 'high', 'low', 'close', 'volume', 'amount']
            )
            self.last_close[symbol] = None
            self.raw_data_buffer[symbol] = []
            self.last_log_time[symbol] = datetime.now(timezone.utc)
        
        # 冷启动：尝试加载历史K线数据
        self.load_klines_from_disk()
    
    def add_raw_data_point(self, symbol: str, data_point: Dict) -> None:
        """添加原始数据点到缓冲区（高频采集优化）"""
        try:
            # 计算基础指标（使用log差）
            spot_mid = (data_point['spot']['bid0_price'] + data_point['spot']['ask0_price']) / 2
            swap_mid = (data_point['swap']['bid0_price'] + data_point['swap']['ask0_price']) / 2
            basis_mid = np.log(swap_mid) - np.log(spot_mid)
            
            basis_bid = np.log(data_point['swap']['bid0_price']) - np.log(data_point['spot']['ask0_price'])
            basis_ask = np.log(data_point['swap']['ask0_price']) - np.log(data_point['spot']['bid0_price'])
            
            amount = np.log(spot_mid) - np.log(data_point['index_price'])
            
            # 构建完整数据点
            full_point = {
                'timestamp': data_point['timestamp'],
                'basis_mid': basis_mid,
                'basis_bid': basis_bid,
                'basis_ask': basis_ask,
                'funding_rate': data_point['funding_rate'],
                'amount': amount,
                'spot_mid': spot_mid,
                'swap_mid': swap_mid,
                'index_price': data_point['index_price']
            }
            
            # 添加到缓冲区
            self.raw_data_buffer[symbol].append(full_point)
            
            # 安全机制：限制缓冲区大小（12分钟=720秒）
            if len(self.raw_data_buffer[symbol]) > BUFFER_MAX_POINTS:
                self.raw_data_buffer[symbol] = self.raw_data_buffer[symbol][-BUFFER_MAX_POINTS:]
            
            # 降低日志频率：每10秒输出1次
            now = datetime.now(timezone.utc)
            if (now - self.last_log_time[symbol]).total_seconds() >= LOG_AGGREGATION_INTERVAL:
                buffer_size = len(self.raw_data_buffer[symbol])
                latest_basis = full_point['basis_mid']
                print(f"[BUFFER] {symbol} 缓冲区大小: {buffer_size}/720, latest_basis={latest_basis:.6f} @ {now.strftime('%H:%M:%S')}")
                self.last_log_time[symbol] = now
            
        except Exception as e:
            print(f"[ERROR] 添加原始数据点失败 {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def compute_kline_from_buffer(self, symbol: str, window_start: datetime, window_end: datetime) -> Optional[Dict]:
        """从缓冲区计算10分钟K线（高频数据优化）"""
        try:
            # 筛选窗口内的数据点（10分钟=600秒）
            window_data = [
                d for d in self.raw_data_buffer[symbol]
                if window_start <= d['timestamp'] <= window_end
            ]
            
            if not window_data:
                print(f"[WARN] {symbol} 在窗口 [{window_start.strftime('%H:%M:%S')}, {window_end.strftime('%H:%M:%S')}] 无数据 (缓冲区大小: {len(self.raw_data_buffer[symbol])})")
                return None
            
            # 按时间排序（确保顺序正确）
            window_data.sort(key=lambda x: x['timestamp'])
            
            # 计算K线各字段
            open_price = window_data[0]['basis_mid']          # 窗口第一个点
            close_price = window_data[-1]['basis_mid']        # 窗口最后一个点
            high_price = max(d['basis_bid'] for d in window_data)  # 所有basis_bid的最大值
            low_price = min(d['basis_ask'] for d in window_data)   # 所有basis_ask的最小值
            volume = window_data[-1]['funding_rate']          # 最后一个点的funding_rate
            amount = window_data[-1]['amount']                # 最后一个点的amount
            
            # 构建K线
            kline = {
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'amount': amount,
                'timestamp': window_start,
                'point_count': len(window_data)  # 用于调试：窗口内数据点数量
            }
            
            # 从缓冲区移除已处理的数据（保留最后2分钟数据用于重叠窗口）
            cutoff_time = window_start + timedelta(minutes=2)
            self.raw_data_buffer[symbol] = [
                d for d in self.raw_data_buffer[symbol]
                if d['timestamp'] >= cutoff_time
            ]
            
            return kline
            
        except Exception as e:
            print(f"[ERROR] 计算K线失败 {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def add_kline(self, symbol: str, kline: Dict) -> bool:
        """将计算好的K线添加到DataFrame"""
        try:
            self.last_close[symbol] = kline['close']
            
            kline_row = {
                'open': kline['open'],
                'high': kline['high'],
                'low': kline['low'],
                'close': kline['close'],
                'volume': kline['volume'],
                'amount': kline['amount']
            }
            
            new_row = pd.DataFrame([kline_row], index=[kline['timestamp']])
            self.kline_data[symbol] = pd.concat([self.kline_data[symbol], new_row])
            
            if len(self.kline_data[symbol]) > self.max_rows:
                self.kline_data[symbol] = self.kline_data[symbol].iloc[-self.max_rows:]
            
            # 确保索引唯一
            self.kline_data[symbol] = self.kline_data[symbol][~self.kline_data[symbol].index.duplicated(keep='last')]
            
            print(f"[KLINE] {symbol} 生成K线 [{kline['timestamp'].strftime('%Y-%m-%d %H:%M')}]: "
                  f"O={kline['open']:.6f} H={kline['high']:.6f} L={kline['low']:.6f} C={kline['close']:.6f} "
                  f"V={kline['volume']:.6f} Points={kline['point_count']}")
            return True
            
        except Exception as e:
            print(f"[ERROR] 添加K线失败 {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_kline_df(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取指定symbol的K线DataFrame"""
        if symbol not in self.kline_data or self.kline_data[symbol].empty:
            return None
        return self.kline_data[symbol].copy()
    
    def get_latest_close(self, symbol: str) -> Optional[float]:
        """获取最新K线的close价格"""
        df = self.get_kline_df(symbol)
        if df is not None and not df.empty:
            return df['close'].iloc[-1]
        return None
    
    def get_latest_observations(self, symbol: str) -> Optional[np.ndarray]:
        """获取最新K线的完整观测值"""
        df = self.get_kline_df(symbol)
        if df is not None and not df.empty:
            return df[['open', 'high', 'low', 'close', 'volume', 'amount']].iloc[-1].values
        return None
    
    def save_klines_to_disk(self) -> None:
        """将所有symbol的K线数据保存到本地JSON文件"""
        try:
            saved_count = 0
            for symbol in CONFIG.symbol_list:
                df = self.kline_data[symbol]
                if df.empty:
                    continue
                
                save_data = {
                    "symbol": symbol,
                    "exchange": self.exchange,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "kline_count": len(df),
                    "columns": df.columns.tolist(),
                    "index": df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                    "data": df.values.tolist()
                }
                
                file_path = self.save_dir / f"{symbol}_klines.json"
                with open(file_path, 'w') as f:
                    json.dump(save_data, f, indent=2)
                
                saved_count += 1
                print(f"[SAVE] {symbol} K线数据已保存 ({len(df)}根K线) -> {file_path}")
            
            if saved_count > 0:
                print(f"[INFO] 共保存 {saved_count} 个币种的K线数据到 {self.save_dir}")
                
        except Exception as e:
            print(f"[ERROR] 保存K线数据失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def load_klines_from_disk(self) -> None:
        """冷启动时从本地JSON文件加载历史K线数据"""
        print(f"[INIT] 尝试从 {self.save_dir} 加载 {self.exchange} 历史K线数据...")
        
        loaded_count = 0
        for symbol in CONFIG.symbol_list:
            file_path = self.save_dir / f"{symbol}_klines.json"
            
            if not file_path.exists():
                print(f"[INFO] 未找到 {symbol} 的历史K线文件 ({file_path})")
                continue
            
            try:
                with open(file_path, 'r') as f:
                    save_data = json.load(f)
                
                required_keys = ['columns', 'index', 'data']
                if not all(key in save_data for key in required_keys):
                    print(f"[WARNING] {symbol} K线文件数据格式无效，跳过加载")
                    continue
                
                df = pd.DataFrame(
                    data=save_data['data'],
                    index=pd.to_datetime(save_data['index'], utc=True),
                    columns=save_data['columns']
                )
                
                required_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
                if not all(col in df.columns for col in required_columns):
                    print(f"[WARNING] {symbol} K线数据缺少必要列，跳过加载")
                    continue
                
                df = df[required_columns]
                
                if len(df) > self.max_rows:
                    df = df.iloc[-self.max_rows:]
                
                self.kline_data[symbol] = df
                self.last_close[symbol] = df['close'].iloc[-1] if not df.empty else None
                
                loaded_count += 1
                latest_time = df.index[-1] if not df.empty else 'N/A'
                print(f"[LOAD] 成功加载 {symbol} 的历史K线 ({len(df)}根), 最新时间: {latest_time}")
                
            except Exception as e:
                print(f"[ERROR] 加载 {symbol} K线数据失败: {str(e)}")
                import traceback
                traceback.print_exc()
                self.kline_data[symbol] = pd.DataFrame(
                    columns=['open', 'high', 'low', 'close', 'volume', 'amount']
                )
                self.last_close[symbol] = None
        
        if loaded_count > 0:
            print(f"[INIT] 成功加载 {loaded_count} 个币种的 {self.exchange} 历史K线数据")
        else:
            print(f"[INIT] 未加载任何历史K线数据，将从零开始累积")

# === 调度函数（高频采集优化）===
def should_collect_data(current_time: datetime) -> bool:
    """每秒整点采集原始数据"""
    return current_time.microsecond < 100000  # 容忍100ms漂移

def should_compute_kline(current_time: datetime) -> bool:
    """每10分钟整点计算K线（0,10,20,30,40,50分）"""
    return current_time.minute % 10 == 0 and current_time.second == 0 and current_time.microsecond < 100000

def should_trigger_full_prediction(current_time: datetime) -> bool:
    """00/30分触发完整预测"""
    return current_time.minute in (0, 30) and current_time.second == 0 and current_time.microsecond < 100000

def should_trigger_reweight_update(current_time: datetime) -> bool:
    """10/20/40/50分触发重加权更新"""
    return current_time.minute in (10, 20, 40, 50) and current_time.second == 0 and current_time.microsecond < 100000

def get_previous_window_bounds(current_time: datetime) -> Tuple[datetime, datetime]:
    """获取上一个10分钟窗口的边界"""
    current_window_start = current_time.replace(
        minute=(current_time.minute // 10) * 10,
        second=0,
        microsecond=0
    )
    prev_window_start = current_window_start - timedelta(minutes=10)
    prev_window_end = current_window_start - timedelta(microseconds=1)
    
    return prev_window_start, prev_window_end

def report_to_feishu(report_dict: Dict) -> None:

    def sanitize_for_json(obj):
        """递归转换NumPy类型为Python原生类型"""
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [sanitize_for_json(x) for x in obj]
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    # 在report_to_feishu开头添加
    print(f"[DEBUG] report_dict types:")
    for symbol, vals in report_dict.items():
        for k, v in vals.items():
            print(f"  {symbol}.{k}: {type(v)} = {v}")

    """通过飞书机器人发送报告"""
    if not CONFIG.my_url:
        print("[WARN] 未配置飞书Webhook URL，跳过发送报告")
        return
    
    my_url = CONFIG.my_url
    url_report = CONFIG.url_report
    my_bot = mb.Bot(my_url)
    report_bot = mb.Bot(url_report)
    
    msg_detail = '=============================\n'
    msg = '=============================\n'
    dic = {}
    
    for symbol in report_dict.keys():
        results = report_dict[symbol]
        dic[symbol] = [
            results['high_mean_last'] + results['high_std_last'],
            results['low_mean_last'] - results['low_std_last']
        ]
        msg += f"Symbol: {symbol}, High = {results['high_mean_last']:.6f} + {results['high_std_last']:.6f}, Low = {results['low_mean_last']:.6f} - {results['low_std_last']:.6f}\n"
        msg_detail += f"------------------------ {symbol} ---------------------\n"
        msg_detail += f"Last: ({results['high_mean_last']:.6f} + {results['high_std_last']:.6f}), ({results['low_mean_last']:.6f} - {results['low_std_last']:.6f})\n" \
                     f"Curr: ({results['high_mean']:.6f} + {results['high_std']:.6f}), ({results['low_mean']:.6f} - {results['low_std']:.6f})\n"
    
    my_bot.text(msg_detail)
    report_bot.text(msg)

    
    # # Redis发布消息
    dic_trans = sanitize_for_json(dic)
    print(f"[DEBUG] 准备发布到Redis: {dic_trans}")
    # 在report_to_feishu开头添加
    print(f"[DEBUG] translated report_dict types:")
    for symbol in dic_trans.keys():
        for k in dic_trans[symbol]:
            print(f"  {symbol}.{k}: {type(k)} = {k}")
    r = Redis(host=redis_config['redisUrl'], db=1, password=redis_config['redisPass'])
    signals_str = json.dumps(sanitize_for_json(dic), cls=DecimalEncoder)
    r.publish(f'kc_maxmin_estimate', signals_str)
    # r.publish(f'kc_maxmin_estimate_test', signals_str)

def print_prediction_summary(symbol: str, pred_sequence: np.ndarray, weights: Optional[np.ndarray] = None, update_type: str = "FULL"):
    """打印预测结果摘要"""
    print(f"\n{'='*70}")
    print(f"[{update_type}] {symbol} 预测结果 | 时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    if pred_sequence.ndim == 2:
        pred_sequence = pred_sequence[np.newaxis, :, :]
    
    n_samples = pred_sequence.shape[0]
    print(f"样本数量: {n_samples}")
    
    if weights is not None:
        print(f"权重统计: 均值={np.mean(weights):.4f}, 标准差={np.std(weights):.4f}, "
              f"最大值={np.max(weights):.4f}, 最小值={np.min(weights):.4f}")
    
    print("\n未来60分钟预测 (Close价格):")
    print(f"{'时间点':<10} {'均值':<12} {'标准差':<12} {'5%分位':<12} {'95%分位':<12}")
    print("-" * 70)
    
    for i in range(6):
        close_prices = pred_sequence[:, i, 3]
        mean = np.mean(close_prices)
        std = np.std(close_prices)
        p5 = np.percentile(close_prices, 5)
        p95 = np.percentile(close_prices, 95)
        print(f"T+{10*(i+1)}min   {mean:<12.6f} {std:<12.6f} {p5:<12.6f} {p95:<12.6f}")
    
    print(f"\n样本#0完整序列 (O,H,L,C,V,A):")
    for i in range(6):
        point = pred_sequence[0, i]
        print(f"  T+{10*(i+1)}min: [{point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f}, "
              f"{point[3]:.6f}, {point[4]:.6f}, {point[5]:.6f}]")

def validate_exchange_symbols(fetcher: BaseDataFetcher, symbol_list: List[str]) -> bool:
    """验证交易所交易对有效性"""
    print(f"[INIT] 验证 {fetcher.exchange_name} 交易对有效性...")
    valid_count = 0
    
    for symbol in symbol_list:
        spot_ob = fetcher.fetch_orderbook(symbol, is_swap=False)
        swap_ob = fetcher.fetch_orderbook(symbol, is_swap=True)
        
        if spot_ob and swap_ob:
            spot_sym, swap_sym = fetcher._standardize_symbol(symbol)
            print(f"  ✓ {symbol}: 现货({spot_sym})和合约({swap_sym})数据均可获取")
            valid_count += 1
        else:
            print(f"  ✗ {symbol}: 数据获取失败")
    
    if valid_count == 0:
        print(f"\n[CRITICAL] {fetcher.exchange_name} 所有交易对验证失败！")
        return False
    
    print(f"\n[INIT] {valid_count}/{len(symbol_list)} 个交易对验证通过，继续启动...\n")
    return True

def main(test_mode: bool = False, use_kucoin: bool = False):
    """主循环（高频采集优化版）"""
    # 确定使用的交易所
    exchange_name = "KuCoin" if use_kucoin else "Binance"
    
    print(f"{'='*70}")
    print(f"Kronos实时预测系统 v2.3 (高频采集版) 启动")
    print(f"数据源交易所: {exchange_name}")
    print(f"监控币种: {CONFIG.symbol_list}")
    print(f"K线保留: 最多{MAX_ROWS}个10分钟K线")
    print(f"数据采集: 每秒1次（原为每分钟1次）")
    print(f"K线计算: 每10分钟整点计算K线（基于600个数据点）")
    print(f"  - open = 窗口第一个数据点的basis_mid")
    print(f"  - close = 窗口最后一个数据点的basis_mid")
    print(f"  - high = max(窗口内所有basis_bid)")
    print(f"  - low = min(窗口内所有basis_ask)")
    print(f"  - volume = 窗口最后一个数据点的funding_rate")
    print(f"  - amount = 窗口最后一个数据点的log(spot_mid)-log(index_price)")
    print(f"缓冲区大小: {BUFFER_MAX_POINTS}个数据点（12分钟）")
    print(f"数据持久化: K线自动保存到 {KLINE_SAVE_DIR}/{exchange_name.lower()}")
    print(f"测试模式: {'ENABLED' if test_mode else 'DISABLED'}")
    print(f"{'='*70}\n")
    
    # 初始化对应的数据采集器
    if use_kucoin:
        fetcher = KuCoinDataFetcher(test_mode=test_mode)
    else:
        fetcher = BinanceDataFetcher(test_mode=test_mode)
    
    # 非测试模式下验证交易对
    if not test_mode:
        if not validate_exchange_symbols(fetcher, CONFIG.symbol_list):
            print(f"[EXIT] {exchange_name} 交易对验证失败，退出程序")
            return
    
    # 初始化K线管理器
    kline_manager = RealtimeKlineManager(
        max_rows=MAX_ROWS,
        save_dir=KLINE_SAVE_DIR,
        exchange=exchange_name.lower()
    )
    
    # 初始化预测器
    predictors: Dict[str, KronosPredictor] = {}
    signal_generators: Dict[str, DynamicSignalGenerator] = {}
    
    for symbol in CONFIG.symbol_list:
        if not predictors:
            predictor = KronosPredictor(
                tokenizer_path=CONFIG.tokenizer_10min,
                predictor_path=CONFIG.predictor_10min
            )
            predictors['shared'] = predictor
        
        signal_generators[symbol] = DynamicSignalGenerator(
            predictor=predictors['shared'],
            lookback=144,
            pred_length=6,
            n_samples=N_SAMPLES,
            temperature=TEMPERATURE
        )
    
    # 等待到下一秒整点开始
    now = datetime.now(timezone.utc)
    next_collect = now.replace(microsecond=0) + timedelta(seconds=1)
    print(f"[INIT] 等待到 {next_collect.strftime('%Y-%m-%d %H:%M:%S')} 开始首次数据采集...")
    
    while datetime.now(timezone.utc) < next_collect:
        time.sleep(0.01)  # 毫秒级精度等待
    
    last_data_collect_second = -1
    # last_kline_compute_minute = -1
    last_full_pred_minute = -1
    last_reweight_minute = -1
    # kline_scheduler = KlineScheduler()

    print(f"\n[SYSTEM] 开始实时数据采集与预测循环 ({exchange_name}) | 采集频率: 1次/秒 ...\n")
    
    # ===== 在主循环初始化处 =====
    last_kline_trigger_time = None  # 记录上次成功触发的UTC时间（非分钟值）

    while True:
        try:
            current_time = datetime.now(timezone.utc)
            current_minute = current_time.minute
            current_second = current_time.second

            # 【修复核心】使用绝对时间窗口判断，而非状态变量比较
            should_compute = False
            window_start = window_end = None

            if current_minute % 10 == 0 and current_second < 3:  # 放宽到秒0-2内均可触发
                # 计算目标窗口（上一个10分钟）
                target_window_start = current_time.replace(
                    minute=(current_minute // 10) * 10,
                    second=0,
                    microsecond=0
                ) - timedelta(minutes=10)
                
                # 关键修复：检查是否已触发过该窗口
                if (last_kline_trigger_time is None or 
                    target_window_start > last_kline_trigger_time):
                    should_compute = True
                    window_start = target_window_start
                    window_end = target_window_start + timedelta(minutes=10) - timedelta(microseconds=1)
                    
            if current_second == 0 and current_time.microsecond < 100000:
                print(f"[DIAG] {current_time.strftime('%H:%M:%S.%f')[:-3]} | "
                    f"min={current_minute} mod10={current_minute%10} sec={current_second} "
                    f"μs={current_time.microsecond:06d} | "
                    f"should_trigger={current_minute%10==0 and current_second==0 and current_time.microsecond<100000}")
            if current_second == 0:
                print(f"[DIAG] Time={current_time.strftime('%H:%M:%S.%f')[:-3]} | "
                    f"last_trigger={last_kline_trigger_time.strftime('%H:%M') if last_kline_trigger_time else 'None'}")
            # 1. 每秒整点采集原始数据（高频采集核心改动）
            if should_collect_data(current_time) and current_second != last_data_collect_second:
                # 交错采集：分散请求避免突发流量
                for i, symbol in enumerate(fetcher.symbol_order):
                    # 每个symbol之间间隔约100ms
                    if i > 0:
                        time.sleep(0.08 / len(CONFIG.symbol_list))
                    
                    spot_ob = fetcher.fetch_orderbook(symbol, is_swap=False)
                    swap_ob = fetcher.fetch_orderbook(symbol, is_swap=True)
                    funding_rate = fetcher.fetch_funding_rate(symbol)
                    index_price = fetcher.fetch_index_price(symbol)
                    
                    if not spot_ob or not swap_ob:
                        if i == 0:  # 仅在第一个symbol时输出SKIP日志
                            print(f"[SKIP] {symbol} 数据不完整，跳过本次采集")
                        continue
                    
                    current_data = {
                        'timestamp': current_time,
                        'spot': spot_ob,
                        'swap': swap_ob,
                        'funding_rate': funding_rate,
                        'index_price': index_price
                    }
                    
                    kline_manager.add_raw_data_point(symbol, current_data)
                
                last_data_collect_second = current_second
            
            # 2. 每10分钟整点计算K线（逻辑不变）
            # if should_compute_kline(current_time) and current_minute != last_kline_compute_minute:
            #     window_start, window_end = get_previous_window_bounds(current_time)
            #     print(f"\n{'#'*70}")
            #     print(f"#  [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] 计算10分钟K线窗口: "
            #           f"{window_start.strftime('%H:%M:%S')} - {window_end.strftime('%H:%M:%S')}  #")
            #     print(f"{'#'*70}\n")
            if should_compute:
                print(f"\n{'#'*70}")
                print(f"#  [{current_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:23]}] 计算10分钟K线 #")
                print(f"#  窗口: {window_start.strftime('%H:%M:%S')} - {window_end.strftime('%H:%M:%S.%f')[:-3]}  #")
                print(f"{'#'*70}\n")

                computed_symbols = []
                for symbol in CONFIG.symbol_list:
                    kline = kline_manager.compute_kline_from_buffer(symbol, window_start, window_end)
                    if kline is not None:
                        if kline_manager.add_kline(symbol, kline):
                            computed_symbols.append(symbol)
                
                if computed_symbols:
                    kline_manager.save_klines_to_disk()
                    last_kline_trigger_time = window_start  # ✅ 更新为窗口开始时间（非当前时间）
                    print(f"[SUCCESS] 成功计算 {len(computed_symbols)} 个币种的K线")
                else:
                    print(f"[INFO] 本次无有效K线可计算")
                
                # last_kline_compute_minute = current_minute
            
            # 3. 00/30分：执行完整预测（逻辑不变）
            report_dict = {}
            if current_minute in (0, 30) and current_second == 0 and last_full_pred_minute != current_minute:
            # if current_minute in (0, 5, 15, 25, 30, 45, 47, 55) and current_second <= 10 and last_full_pred_minute != current_minute:
                print(f"\n{'*'*70}")
                print(f"*  [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] 触发完整预测 (00/30分)  *")
                print(f"{'*'*70}\n")
                
                result_dict = {}
                for symbol in CONFIG.symbol_list:
                    df_10min = kline_manager.get_kline_df(symbol)
                    kline_count = len(df_10min) if df_10min is not None else 0
                    
                    if kline_count < MIN_KLINES_FOR_PREDICTION:
                        print(f"[SKIP] {symbol}: 历史K线不足 ({kline_count}/{MIN_KLINES_FOR_PREDICTION})")
                        continue
                    
                    params = signal_generators[symbol].generate_initial_signal(
                        df_10min=df_10min,
                        current_time=current_time,
                        feature_list=['open', 'high', 'low', 'close', 'volume', 'amount'],
                        time_features=['minute', 'hour', 'weekday', 'day', 'month']
                    )
                    
                    pred_seq = signal_generators[symbol].pred_sequences
                    
                    if pred_seq is not None:
                        result_dict[symbol] = pred_seq
                        report_dict[symbol] = {
                            'high_mean_last': signal_generators[symbol].estimates_last[0],
                            'high_std_last': signal_generators[symbol].estimates_last[1],
                            'low_mean_last': signal_generators[symbol].estimates_last[2],
                            'low_std_last': signal_generators[symbol].estimates_last[3],
                            'high_mean': signal_generators[symbol].estimates[0],
                            'high_std': signal_generators[symbol].estimates[1],
                            'low_mean': signal_generators[symbol].estimates[2],
                            'low_std': signal_generators[symbol].estimates[3]
                        }
                        print_prediction_summary(
                            symbol=symbol,
                            pred_sequence=pred_seq,
                            weights=signal_generators[symbol].pred_weights,
                            update_type="FULL_PRED"
                        )
                    else:
                        print(f"[{symbol}] 预测失败 - 无预测序列")
                
                if result_dict:
                    timestamp_str = current_time.strftime("%Y%m%d_%H%M")
                    result_path = Path(f"./data/predictions/{exchange_name.lower()}") / f"full_pred_{timestamp_str}.json"
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    serializable_result = {
                        symbol: pred[0].tolist()
                        for symbol, pred in result_dict.items()
                    }
                    
                    with open(result_path, 'w') as f:
                        json.dump(serializable_result, f, indent=2)
                    print(f"\n[SAVE] 完整预测结果已保存至: {result_path}")
                
                last_full_pred_minute = current_minute
            
            # 4. 10/20/40/50分：执行重加权更新（逻辑不变）
            elif current_minute in (10, 20, 40, 50) and current_second <= 10 and last_reweight_minute != current_minute:
                print(f"\n{'~'*70}")
                print(f"~  [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] 触发重加权更新 (10/20/40/50分)  ~")
                print(f"{'~'*70}\n")
                
                for symbol in CONFIG.symbol_list:
                    if (signal_generators[symbol].pred_sequences is None or 
                        len(signal_generators[symbol].pred_sequences.shape) < 3 or
                        signal_generators[symbol].pred_sequences.shape[1] == 0):
                        print(f"[SKIP] {symbol}: 无有效预测序列，跳过重加权")
                        continue
                    
                    latest_close = kline_manager.get_latest_close(symbol)
                    latest_observations = kline_manager.get_latest_observations(symbol)
                    
                    if latest_close is None:
                        print(f"[SKIP] {symbol}: 无最新K线数据，跳过重加权")
                        continue
                    
                    params = signal_generators[symbol].update_signal_with_full_observations(
                        observations=latest_observations,
                        timestamp=current_time
                    )
                    
                    updated_pred_seq = signal_generators[symbol].pred_sequences
                    
                    if updated_pred_seq is not None and updated_pred_seq.shape[1] > 0:
                        next_step_pred = updated_pred_seq[:, 0, :]
                        print_prediction_summary(
                            symbol=symbol,
                            pred_sequence=next_step_pred,
                            weights=signal_generators[symbol].pred_weights,
                            update_type="REWEIGHT"
                        )
                        report_dict[symbol] = {
                            'high_mean_last': signal_generators[symbol].estimates_last[0],
                            'high_std_last': signal_generators[symbol].estimates_last[1],
                            'low_mean_last': signal_generators[symbol].estimates_last[2],
                            'low_std_last': signal_generators[symbol].estimates_last[3],
                            'high_mean': signal_generators[symbol].estimates[0],
                            'high_std': signal_generators[symbol].estimates[1],
                            'low_mean': signal_generators[symbol].estimates[2],
                            'low_std': signal_generators[symbol].estimates[3]
                        }
                        estimates = signal_generators[symbol].estimates
                        if estimates:
                            print(f"  调整统计: High_mean={estimates[0]:.6f}, High_std={estimates[1]:.6f}, "
                                  f"Low_mean={estimates[2]:.6f}, Low_std={estimates[3]:.6f}")
                    else:
                        print(f"[{symbol}] 重加权后无有效预测序列")
                
                last_reweight_minute = current_minute
            
            # 5. 汇总报告（逻辑不变）
            if report_dict:
                print(f"\n{'#'*70}")
                print(f"#  预测摘要报告 | 时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}  #")
                print(f"{'#'*70}")
                print(f"{'Symbol':<10} {'High_Mean':<12} {'High_Std':<12} {'Low_Mean':<12} {'Low_Std':<12}")
                print("-" * 70)
                report_to_feishu(report_dict)
            
            # 6. 精确的毫秒级睡眠（避免时间漂移）
            next_second = (current_time.second + 1) % 60
            target_time = current_time.replace(second=next_second, microsecond=0)
            sleep_time = (target_time - datetime.now(timezone.utc)).total_seconds()
            if sleep_time > 0:
                time.sleep(min(sleep_time, 0.5))  # 最多睡0.5秒
            else:
                time.sleep(0.001)  # 最小睡眠1ms
            
        except KeyboardInterrupt:
            print("\n[INFO] 检测到用户中断，正在保存最新K线数据并退出...")
            kline_manager.save_klines_to_disk()
            print("[INFO] 已保存最新K线数据，安全退出")
            break
        except Exception as e:
            print(f"[CRITICAL] 主循环异常: {str(e)}")
            import traceback
            traceback.print_exc()
            time.sleep(1)  # 异常后等待1秒再重试

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kronos实时预测系统 v2.3 (高频采集版)")
    parser.add_argument("--test", action="store_true", help="启用测试模式（使用模拟数据）")
    parser.add_argument("--kc", action="store_true", help="使用KuCoin API替代Binance API")
    args = parser.parse_args()
    
    main(test_mode=args.test, use_kucoin=args.kc)