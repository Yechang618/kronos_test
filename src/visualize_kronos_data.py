#!/usr/bin/env python3
"""
Kronos数据可视化工具
- 绘制历史K线数据（high/close/low）
- 绘制预测结果（high/close/low）
- 支持Binance和KuCoin数据源
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非GUI后端
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Tuple, Optional

# 设置中文字体支持（避免中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class KlineVisualizer:
    """K线数据可视化器"""
    
    def __init__(self, exchange: str = "binance"):
        self.exchange = exchange.lower()
        self.kline_dir = Path("./datasets/temp") / self.exchange
        self.pred_dir = Path("./data/predictions") / self.exchange
        self.kline_fig_dir = Path("./figures/temp/kline") / self.exchange
        self.pred_fig_dir = Path("./data/predictions") / self.exchange
        
        # 创建输出目录
        self.kline_fig_dir.mkdir(parents=True, exist_ok=True)
        self.pred_fig_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[INIT] 交易所: {self.exchange.upper()}")
        print(f"[INIT] K线数据目录: {self.kline_dir}")
        print(f"[INIT] 预测数据目录: {self.pred_dir}")
        print(f"[INIT] K线图表输出: {self.kline_fig_dir}")
        print(f"[INIT] 预测图表输出: {self.pred_fig_dir}")
    
    def load_kline_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """加载K线数据"""
        file_path = self.kline_dir / f"{symbol}_klines.json"
        
        if not file_path.exists():
            print(f"[WARN] K线文件不存在: {file_path}")
            return None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 重建DataFrame
            df = pd.DataFrame(
                data=data['data'],
                index=pd.to_datetime(data['index'], utc=True),
                columns=data['columns']
            )
            
            # 确保必要列存在
            required_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
            if not all(col in df.columns for col in required_cols):
                print(f"[ERROR] {symbol} K线数据缺少必要列")
                return None
            
            df = df[required_cols]
            print(f"[LOAD] 成功加载 {symbol} K线数据 ({len(df)} 根K线), 时间范围: {df.index[0]} ~ {df.index[-1]}")
            return df
            
        except Exception as e:
            print(f"[ERROR] 加载 {symbol} K线数据失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_latest_prediction(self) -> Optional[Dict[str, List[List[float]]]]:
        """加载最新的预测结果"""
        if not self.pred_dir.exists():
            print(f"[WARN] 预测目录不存在: {self.pred_dir}")
            return None
        
        # 查找所有预测文件
        pred_files = list(self.pred_dir.glob("full_pred_*.json"))
        if not pred_files:
            print(f"[WARN] 未找到预测文件")
            return None
        
        # 按时间戳排序，取最新的
        pred_files.sort(key=lambda x: x.name, reverse=True)
        latest_file = pred_files[0]
        
        try:
            with open(latest_file, 'r') as f:
                pred_data = json.load(f)
            
            # 从文件名提取预测时间
            timestamp_str = latest_file.name.replace("full_pred_", "").replace(".json", "")
            pred_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M").replace(tzinfo=timezone.utc)
            
            print(f"[LOAD] 成功加载最新预测: {latest_file.name} (生成时间: {pred_time.strftime('%Y-%m-%d %H:%M:%S UTC')})")
            print(f"      包含 {len(pred_data)} 个币种的预测")
            return {
                'data': pred_data,
                'timestamp': pred_time,
                'file': latest_file
            }
            
        except Exception as e:
            print(f"[ERROR] 加载预测数据失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_kline_history(self, symbol: str, df: pd.DataFrame, max_points: int = 50) -> Path:
        """绘制历史K线图表"""
        # 取最近max_points个数据点
        df_plot = df.tail(max_points).copy()
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # 绘制三条线
        ax.plot(df_plot.index, df_plot['high'], 'g-', linewidth=2, label='High', alpha=0.8)
        ax.plot(df_plot.index, df_plot['close'], 'b-', linewidth=2.5, label='Close', alpha=0.9)
        ax.plot(df_plot.index, df_plot['low'], 'r-', linewidth=2, label='Low', alpha=0.8)
        
        # 填充high-low区域
        ax.fill_between(df_plot.index, df_plot['low'], df_plot['high'], 
                        color='gray', alpha=0.2, label='Range (High-Low)')
        
        # 标记最新价格
        latest = df_plot.iloc[-1]
        ax.plot(df_plot.index[-1], latest['close'], 'bo', markersize=10, label=f'Latest Close: {latest["close"]:.4f}')
        
        # 格式化
        ax.set_title(f'{symbol} 10分钟K线历史数据 ({self.exchange.upper()})\n'
                    f'数据点: {len(df_plot)} | 时间范围: {df_plot.index[0].strftime("%Y-%m-%d %H:%M")} ~ {df_plot.index[-1].strftime("%Y-%m-%d %H:%M")}',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('时间 (UTC)', fontsize=12)
        ax.set_ylabel('Basis 价格', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 优化时间轴格式
        if len(df_plot) > 20:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        plt.tight_layout()
        
        # 保存图表
        output_file = self.kline_fig_dir / f"{self.exchange}_{symbol}_kline.png"
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[SAVE] K线图表已保存: {output_file}")
        return output_file
    
    def plot_prediction(self, symbol: str, df_history: pd.DataFrame, 
                       pred_data: List[List[float]], pred_time: datetime) -> Path:
        """
        绘制预测结果图表
        :param df_history: 历史K线数据（最后12个点）
        :param pred_data: 预测序列 (6x6)
        :param pred_time: 预测生成时间（即第一个预测K线的时间戳）
        """
        # 准备历史数据（最后12个点）
        df_hist = df_history.tail(12).copy()
        
        # 准备预测数据
        pred_times = [pred_time + timedelta(minutes=10 * i) for i in range(6)]
        pred_high = [point[1] for point in pred_data]  # 索引1: high
        pred_close = [point[3] for point in pred_data]  # 索引3: close
        pred_low = [point[2] for point in pred_data]   # 索引2: low
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 绘制历史数据（实线）
        ax.plot(df_hist.index, df_hist['high'], 'g-', linewidth=2, label='历史 High', alpha=0.7)
        ax.plot(df_hist.index, df_hist['close'], 'b-', linewidth=2.5, label='历史 Close', alpha=0.8)
        ax.plot(df_hist.index, df_hist['low'], 'r-', linewidth=2, label='历史 Low', alpha=0.7)
        
        # 绘制预测数据（虚线）
        ax.plot(pred_times, pred_high, 'g--', linewidth=2.5, label='预测 High', alpha=0.9)
        ax.plot(pred_times, pred_close, 'b--', linewidth=3, label='预测 Close', alpha=0.95)
        ax.plot(pred_times, pred_low, 'r--', linewidth=2.5, label='预测 Low', alpha=0.9)
        
        # 填充预测区间
        ax.fill_between(pred_times, pred_low, pred_high, 
                        color='purple', alpha=0.15, label='预测区间 (High-Low)')
        
        # 标记预测起始点
        if not df_hist.empty:
            connect_point = df_hist.iloc[-1]['close']
            ax.plot(df_hist.index[-1], connect_point, 'ko', markersize=8, label=f'预测起始点 ({df_hist.index[-1].strftime("%H:%M")})')
        
        # 添加预测时间范围标注
        pred_range = f'预测时间范围: {pred_times[0].strftime("%H:%M")} ~ {pred_times[-1].strftime("%H:%M")} UTC'
        ax.text(0.02, 0.98, pred_range, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 格式化
        title = (f'{symbol} Basis 价格预测 ({self.exchange.upper()})\n'
                f'预测生成时间: {pred_time.strftime("%Y-%m-%d %H:%M UTC")} | '
                f'预测未来60分钟 (6个10分钟K线)')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('时间 (UTC)', fontsize=12)
        ax.set_ylabel('Basis 价格', fontsize=12)
        ax.legend(loc='best', fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 优化时间轴
        all_times = list(df_hist.index) + pred_times
        ax.set_xlim(min(all_times) - timedelta(minutes=5), max(all_times) + timedelta(minutes=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 添加y轴网格线增强可读性
        ax.yaxis.grid(True, alpha=0.3, linestyle=':')
        
        plt.tight_layout()
        
        # 保存图表
        timestamp_str = pred_time.strftime("%Y%m%d_%H%M")
        output_file = self.pred_fig_dir / f"{self.exchange}_{symbol}_prediction_{timestamp_str}.png"
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[SAVE] 预测图表已保存: {output_file}")
        return output_file
    
    def visualize_all_klines(self, max_points: int = 50) -> List[Path]:
        """可视化所有symbol的K线历史数据"""
        if not self.kline_dir.exists():
            print(f"[ERROR] K线数据目录不存在: {self.kline_dir}")
            return []
        
        # 获取所有symbol的json文件
        json_files = list(self.kline_dir.glob("*_klines.json"))
        if not json_files:
            print(f"[WARN] 未找到K线数据文件")
            return []
        
        print(f"\n{'='*70}")
        print(f"开始可视化K线历史数据 ({len(json_files)} 个币种)")
        print(f"{'='*70}\n")
        
        output_files = []
        for json_file in sorted(json_files):
            symbol = json_file.stem.replace("_klines", "")
            print(f"\n[PROCESS] 处理 {symbol} ...")
            
            df = self.load_kline_data(symbol)
            if df is not None and not df.empty:
                output_file = self.plot_kline_history(symbol, df, max_points)
                output_files.append(output_file)
            else:
                print(f"[SKIP] {symbol} 无有效K线数据")
        
        print(f"\n{'='*70}")
        print(f"K线历史数据可视化完成! 共生成 {len(output_files)} 张图表")
        print(f"输出目录: {self.kline_fig_dir}")
        print(f"{'='*70}\n")
        return output_files
    
    def visualize_predictions(self) -> List[Path]:
        """可视化所有symbol的预测结果"""
        # 加载最新预测
        pred_result = self.load_latest_prediction()
        if not pred_result:
            print(f"[ERROR] 无法加载预测数据")
            return []
        
        pred_data_all = pred_result['data']
        pred_time = pred_result['timestamp']
        
        # 加载所有symbol的历史K线
        symbol_klines = {}
        for symbol in pred_data_all.keys():
            df = self.load_kline_data(symbol)
            if df is not None and not df.empty:
                symbol_klines[symbol] = df
        
        if not symbol_klines:
            print(f"[ERROR] 无有效历史K线数据用于预测可视化")
            return []
        
        print(f"\n{'='*70}")
        print(f"开始可视化预测结果 ({len(pred_data_all)} 个币种)")
        print(f"预测生成时间: {pred_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"{'='*70}\n")
        
        output_files = []
        for symbol, pred_seq in pred_data_all.items():
            print(f"\n[PROCESS] 处理 {symbol} 预测可视化 ...")
            
            # 检查历史数据
            if symbol not in symbol_klines:
                print(f"[SKIP] {symbol} 无历史K线数据，跳过预测可视化")
                continue
            
            df_hist = symbol_klines[symbol]
            
            # 验证预测序列格式
            if not isinstance(pred_seq, list) or len(pred_seq) != 6 or len(pred_seq[0]) != 6:
                print(f"[WARN] {symbol} 预测序列格式无效，跳过")
                continue
            
            # 绘制预测图表
            try:
                output_file = self.plot_prediction(symbol, df_hist, pred_seq, pred_time)
                output_files.append(output_file)
            except Exception as e:
                print(f"[ERROR] 绘制 {symbol} 预测图表失败: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*70}")
        print(f"预测结果可视化完成! 共生成 {len(output_files)} 张图表")
        print(f"输出目录: {self.pred_fig_dir}")
        print(f"{'='*70}\n")
        return output_files


def main():
    parser = argparse.ArgumentParser(description="Kronos数据可视化工具")
    parser.add_argument("--exchange", type=str, default="binance", 
                       choices=["binance", "kucoin"],
                       help="交易所数据源 (默认: binance)")
    parser.add_argument("--kline", action="store_true", 
                       help="仅可视化K线历史数据")
    parser.add_argument("--pred", action="store_true", 
                       help="仅可视化预测结果")
    parser.add_argument("--max-points", type=int, default=50,
                       help="K线历史图表最大数据点数 (默认: 50)")
    parser.add_argument("--all", action="store_true", 
                       help="可视化所有数据 (K线+预测)，默认行为")
    
    args = parser.parse_args()
    
    # 默认行为：可视化所有数据
    if not args.kline and not args.pred:
        args.all = True
    
    print(f"\n{'#'*70}")
    print(f"# Kronos数据可视化工具 v1.0")
    print(f"# 交易所: {args.exchange.upper()}")
    print(f"{'#'*70}\n")
    
    visualizer = KlineVisualizer(exchange=args.exchange)
    
    output_files = []
    
    if args.all or args.kline:
        output_files.extend(visualizer.visualize_all_klines(max_points=args.max_points))
    
    if args.all or args.pred:
        output_files.extend(visualizer.visualize_predictions())
    
    if not output_files:
        print("\n[INFO] 未生成任何图表，请检查数据目录是否存在有效数据")
        return 1
    
    print(f"\n{'='*70}")
    print(f"✓ 所有可视化任务完成!")
    print(f"✓ 共生成 {len(output_files)} 张图表")
    print(f"✓ K线图表: {visualizer.kline_fig_dir}")
    print(f"✓ 预测图表: {visualizer.pred_fig_dir}")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())