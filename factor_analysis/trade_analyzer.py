#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量化交易记录分析工具
功能：数据清洗、指标计算、绩效分析、可视化
"""

from pathlib import Path

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.6f' % x)


class TradeAnalyzer:
    """交易记录分析器"""
    
    def __init__(self, file_path):
        """
        初始化分析器
        
        Args:
            file_path: CSV文件路径
        """
        self.file_path = file_path
        self.raw_data = None
        self.cleaned_data = None
        self.trade_pairs = None
        self.performance_metrics = {}
        
    def load_data(self):
        """加载原始数据"""
        print("=" * 60)
        print("正在加载交易数据...")
        print("=" * 60)
        
        self.raw_data = pd.read_csv(self.file_path)
        print(f"✓ 成功加载 {len(self.raw_data)} 条交易记录")
        print(f"✓ 数据列数：{len(self.raw_data.columns)}")
        
        return self
    
    def clean_data(self):
        """数据清洗"""
        print("\n" + "=" * 60)
        print("正在清洗数据...")
        print("=" * 60)
        
        df = self.raw_data.copy()
        
        # 1. 处理缺失值
        missing_before = df.isnull().sum().sum()
        df = df.dropna(subset=['date', 'symbol', 'operation'])
        missing_after = df.isnull().sum().sum()
        print(f"✓ 处理缺失值：{missing_before - missing_after} 个")
        
        # 2. 转换时间格式
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['hour'] = df['date'].dt.hour
        
        # 3. 标准化交易类型
        df['operation_type'] = df['operation'].apply(
            lambda x: 'OPEN' if 'open' in str(x).lower() else 'CLOSE'
        )
        
        # 4. 计算交易对
        df['base_asset'] = df['symbol'].apply(lambda x: x.replace('USDT', ''))
        
        # 5. 处理数值列
        numeric_cols = [
            'threshold', 'maker/spot_executed_price', 'taker/swap_executed_price',
            'maker/spot_executed_qty', 'executed_volume', 'gain_vs_threshold'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 6. 计算持仓时长（毫秒转秒）
        if 'timer_start_ts' in df.columns and 'maker/spot_executed_ts' in df.columns:
            df['hold_duration_ms'] = (
                df['maker/spot_executed_ts'] - df['timer_start_ts']
            ).fillna(0)
            df['hold_duration_sec'] = df['hold_duration_ms'] / 1000
        
        # 7. 计算实际收益率
        df['actual_return_rate'] = df['gain_vs_threshold'] / df['threshold'].abs()
        df['actual_return_rate'] = df['actual_return_rate'].fillna(0)
        
        self.cleaned_data = df
        print(f"✓ 清洗后有效记录：{len(df)} 条")
        print(f"✓ 交易对数量：{df['symbol'].nunique()} 个")
        print(f"✓ 时间范围：{df['date'].min()} 至 {df['date'].max()}")
        
        return self
    
    def calculate_performance_metrics(self):
        """计算绩效指标"""
        print("\n" + "=" * 60)
        print("正在计算绩效指标...")
        print("=" * 60)
        
        df = self.cleaned_data
        
        # 1. 基础统计
        total_trades = len(df)
        open_trades = len(df[df['operation_type'] == 'OPEN'])
        close_trades = len(df[df['operation_type'] == 'CLOSE'])
        
        # 2. 收益统计
        total_gain = df['gain_vs_threshold'].sum()
        avg_gain = df['gain_vs_threshold'].mean()
        std_gain = df['gain_vs_threshold'].std()
        
        # 3. 胜率计算（基于gain_vs_threshold正负）
        profitable_trades = len(df[df['gain_vs_threshold'] > 0])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # 4. 盈亏比
        avg_profit = df[df['gain_vs_threshold'] > 0]['gain_vs_threshold'].mean()
        avg_loss = df[df['gain_vs_threshold'] <= 0]['gain_vs_threshold'].mean()
        profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
        
        # 5. 最大回撤
        cumulative_returns = df['gain_vs_threshold'].cumsum()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        # 6. 持仓时间统计
        avg_hold_time = df['hold_duration_sec'].mean()
        max_hold_time = df['hold_duration_sec'].max()
        
        # 7. 交易量统计
        total_volume = df['executed_volume'].sum()
        avg_volume = df['executed_volume'].mean()
        
        self.performance_metrics = {
            'total_trades': total_trades,
            'open_trades': open_trades,
            'close_trades': close_trades,
            'total_gain': total_gain,
            'avg_gain': avg_gain,
            'std_gain': std_gain,
            'win_rate': win_rate,
            'profitable_trades': profitable_trades,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'max_drawdown': max_drawdown,
            'avg_hold_time_sec': avg_hold_time,
            'max_hold_time_sec': max_hold_time,
            'total_volume': total_volume,
            'avg_volume': avg_volume
        }
        
        print(f"✓ 总交易数：{total_trades}")
        print(f"✓ 总收益：{total_gain:.6f}")
        print(f"✓ 胜率：{win_rate:.2%}")
        print(f"✓ 盈亏比：{profit_loss_ratio:.2f}")
        print(f"✓ 最大回撤：{max_drawdown:.2%}")
        
        return self
    
    def analyze_by_symbol(self):
        """按交易对分析"""
        print("\n" + "=" * 60)
        print("正在按交易对分析...")
        print("=" * 60)
        
        df = self.cleaned_data
        
        symbol_stats = df.groupby('symbol').agg({
            'gain_vs_threshold': ['sum', 'mean', 'std', 'count'],
            'executed_volume': 'sum',
            'hold_duration_sec': 'mean',
            'actual_return_rate': 'mean'
        }).round(6)
        
        # 扁平化列名
        symbol_stats.columns = [
            'total_gain', 'avg_gain', 'std_gain', 'trade_count',
            'total_volume', 'avg_hold_time', 'avg_return_rate'
        ]
        
        # 计算胜率
        symbol_stats['win_rate'] = df.groupby('symbol').apply(
            lambda x: (x['gain_vs_threshold'] > 0).sum() / len(x)
        ).round(4)
        
        # 排序
        symbol_stats = symbol_stats.sort_values('total_gain', ascending=False)
        
        self.trade_pairs = symbol_stats
        
        print(f"✓ 分析完成 {len(symbol_stats)} 个交易对")
        print("\n前5名交易对（按总收益）:")
        print(symbol_stats.head())
        
        return symbol_stats
    
    def analyze_by_time(self):
        """按时间维度分析"""
        print("\n" + "=" * 60)
        print("正在按时间维度分析...")
        print("=" * 60)
        
        df = self.cleaned_data
        
        # 按小时分析
        hourly_stats = df.groupby('hour').agg({
            'gain_vs_threshold': ['sum', 'mean', 'count'],
            'executed_volume': 'sum'
        }).round(6)
        hourly_stats.columns = ['total_gain', 'avg_gain', 'trade_count', 'total_volume']
        
        # 按日分析
        daily_stats = df.groupby(df['date'].dt.date).agg({
            'gain_vs_threshold': ['sum', 'mean', 'count'],
            'executed_volume': 'sum'
        }).round(6)
        daily_stats.columns = ['total_gain', 'avg_gain', 'trade_count', 'total_volume']
        
        print(f"✓ 小时维度：{len(hourly_stats)} 个小时段")
        print(f"✓ 日维度：{len(daily_stats)} 个交易日")
        
        return hourly_stats, daily_stats
    
    def generate_report(self, output_file='trade_analysis_report.txt'):
        """生成分析报告"""
        print("\n" + "=" * 60)
        print("正在生成分析报告...")
        print("=" * 60)
        
        report = []
        report.append("=" * 60)
        report.append("量化交易记录分析报告")
        report.append("=" * 60)
        report.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"数据文件：{self.file_path}")
        report.append("")
        
        # 1. 概览
        report.append("-" * 60)
        report.append("一、交易概览")
        report.append("-" * 60)
        report.append(f"总交易数：{self.performance_metrics['total_trades']}")
        report.append(f"开仓交易：{self.performance_metrics['open_trades']}")
        report.append(f"平仓交易：{self.performance_metrics['close_trades']}")
        report.append(f"交易对数量：{len(self.trade_pairs)}")
        report.append("")
        
        # 2. 收益分析
        report.append("-" * 60)
        report.append("二、收益分析")
        report.append("-" * 60)
        report.append(f"总收益：{self.performance_metrics['total_gain']:.6f}")
        report.append(f"平均收益：{self.performance_metrics['avg_gain']:.6f}")
        report.append(f"收益标准差：{self.performance_metrics['std_gain']:.6f}")
        report.append(f"盈利交易数：{self.performance_metrics['profitable_trades']}")
        report.append(f"平均盈利：{self.performance_metrics['avg_profit']:.6f}")
        report.append(f"平均亏损：{self.performance_metrics['avg_loss']:.6f}")
        report.append("")
        
        # 3. 风险指标
        report.append("-" * 60)
        report.append("三、风险指标")
        report.append("-" * 60)
        report.append(f"胜率：{self.performance_metrics['win_rate']:.2%}")
        report.append(f"盈亏比：{self.performance_metrics['profit_loss_ratio']:.2f}")
        report.append(f"最大回撤：{self.performance_metrics['max_drawdown']:.2%}")
        report.append(f"平均持仓时间：{self.performance_metrics['avg_hold_time_sec']:.2f} 秒")
        report.append(f"最大持仓时间：{self.performance_metrics['max_hold_time_sec']:.2f} 秒")
        report.append("")
        
        # 4. 交易量
        report.append("-" * 60)
        report.append("四、交易量分析")
        report.append("-" * 60)
        report.append(f"总交易量：{self.performance_metrics['total_volume']:.6f}")
        report.append(f"平均交易量：{self.performance_metrics['avg_volume']:.6f}")
        report.append("")
        
        # 5. 交易对排名
        report.append("-" * 60)
        report.append("五、交易对收益排名（TOP 10）")
        report.append("-" * 60)
        
        if self.trade_pairs is not None:
            top_symbols = self.trade_pairs.head(10)
            for i, (symbol, row) in enumerate(top_symbols.iterrows(), 1):
                report.append(
                    f"{i}. {symbol}: 收益={row['total_gain']:.6f}, "
                    f"胜率={row['win_rate']:.2%}, 交易数={int(row['trade_count'])}"
                )
        
        report.append("")
        report.append("=" * 60)
        report.append("报告结束")
        report.append("=" * 60)
        
        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"✓ 报告已保存至：{output_file}")
        
        return '\n'.join(report)
    
    def export_cleaned_data(self, output_file='cleaned_trade_data.csv'):
        """导出清洗后的数据"""
        print("\n" + "=" * 60)
        print("正在导出清洗后的数据...")
        print("=" * 60)
        
        # 选择关键列
        key_columns = [
            'date', 'symbol', 'operation', 'operation_type', 'base_asset',
            'threshold', 'maker/spot_executed_price', 'taker/swap_executed_price',
            'maker/spot_executed_qty', 'executed_volume', 'gain_vs_threshold',
            'actual_return_rate', 'hold_duration_sec',
            'year', 'month', 'day', 'hour'
        ]
        
        available_columns = [col for col in key_columns if col in self.cleaned_data.columns]
        export_data = self.cleaned_data[available_columns]
        
        export_data.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✓ 数据已导出至：{output_file}")
        print(f"✓ 导出列数：{len(available_columns)}")
        
        return export_data


def main():
    """主函数"""
    print("\n" + "🚀" * 30)
    print("量化交易记录分析系统")
    print("🚀" * 30 + "\n")
    INPUT_BASE = Path("./dataset/bn_trade")
    # 初始化分析器
    filename = INPUT_BASE / "combined_bf1_20260101.csv"
    analyzer = TradeAnalyzer(filename)
    
    # 执行分析流程
    analyzer.load_data()
    analyzer.clean_data()
    analyzer.calculate_performance_metrics()
    analyzer.analyze_by_symbol()
    analyzer.analyze_by_time()
    
    # 生成报告
    report = analyzer.generate_report()
    print("\n" + report)
    
    # 导出数据
    analyzer.export_cleaned_data()
    
    print("\n" + "✅" * 30)
    print("分析完成！")
    print("✅" * 30 + "\n")
    
    return analyzer


if __name__ == '__main__':
    analyzer = main()