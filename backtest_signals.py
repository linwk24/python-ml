#!/usr/bin/env python3
"""
LSTM 预测信号历史回测脚本

功能：对指定时间段的历史数据进行逐日预测，输出涨跌信号
用法：python backtest_signals.py --symbol BTCUSDT --start 2020-03-01 --end 2025-04-01
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json

from models.enhanced_lstm import EnhancedLSTMPredictor
from config import get_all_klines, format_symbol, EXCHANGE


def fetch_kline_data(symbol: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
    """
    获取历史 K 线数据（支持多交易所，统一输出 Binance 格式）
    
    Args:
        symbol: 交易对，如 BTCUSDT
        start_date: 开始日期，格式 YYYY-MM-DD
        end_date: 结束日期，格式 YYYY-MM-DD
        interval: K 线周期，默认 1d（日线）
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    print(f"正在从 {EXCHANGE} 获取 {symbol} 从 {start_date} 到 {end_date} 的历史数据...")
    
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    
    try:
        all_data = get_all_klines(symbol, interval, start_ts, end_ts)
    except Exception as e:
        print(f"获取数据失败: {e}")
        return pd.DataFrame()
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    
    print(f"共获取 {len(df)} 条 {interval} K线数据")
    print(f"时间范围: {df.index[0]} 至 {df.index[-1]}")
    
    return df


def run_backtest(symbol: str, start_date: str, end_date: str, interval: str = "1d", lookback: int = 120, output_file: str = None, only_change: bool = False, sensitive: bool = False):
    """
    运行回测，输出每个时间点的预测信号
    
    Args:
        symbol: 交易对
        start_date: 开始日期
        end_date: 结束日期
        interval: K 线周期（1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w）
        lookback: 模型需要的历史数据长度（默认120条）
        output_file: 输出文件路径（可选）
        only_change: 是否只输出信号变化的记录（过滤连续相同信号）
        sensitive: 是否使用敏感模式（更及时的信号切换）
    """
    # 1. 初始化模型
    print(f"\n{'='*60}")
    print(f"LSTM 预测信号历史回测")
    print(f"{'='*60}")
    print(f"交易对: {symbol}")
    print(f"时间范围: {start_date} 至 {end_date}")
    print(f"K线周期: {interval}")
    print(f"回看窗口: {lookback} 条 K 线")
    print(f"敏感模式: {'是' if sensitive else '否'}")
    
    predictor = EnhancedLSTMPredictor(symbol=symbol, sensitive=sensitive)
    
    # 检查模型是否已训练
    if not predictor.load_model():
        print(f"\n错误: 未找到 {symbol} 的已训练模型")
        print("请先调用 /train 接口训练模型，或运行:")
        print(f"  curl -X POST http://localhost:8000/train -H 'Content-Type: application/json' -d '{{\"symbol\": \"{symbol}\", \"use_sample_data\": true}}'")
        return
    
    print(f"模型已加载: {predictor.model_path}")
    
    # 2. 获取历史数据
    # 为了有足够的回看数据，提前获取 lookback 条数据
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    
    # 根据周期计算需要提前获取的数据量
    interval_days = {"1m": 0.1, "5m": 0.3, "15m": 1, "30m": 2, "1h": 5, "4h": 20, "1d": lookback * 2, "1w": lookback * 14}
    days_ahead = interval_days.get(interval, lookback * 2)
    extended_start = (start_dt - timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    
    df = fetch_kline_data(symbol, extended_start, end_date, interval=interval)
    
    if df.empty:
        print("未获取到有效数据，请检查日期范围和币种。")
        return
    
    # 3. 逐日预测
    print(f"\n开始逐日预测...")
    print(f"{'='*60}")
    
    results = []
    trend_map = {0: "看跌", 1: "中性", 2: "看涨"}
    last_trend = None  # 上次的预测方向
    last_entry_price = None  # 上次预测方向切换时的入场价
    
    # 从 start_date 开始预测
    start_idx = df.index.get_loc(start_dt) if start_dt in df.index else 0
    
    for i in range(start_idx, len(df)):
        current_date = df.index[i]
        
        # 构建模型输入：使用过去 lookback 条数据
        if i < lookback:
            continue
            
        klines = []
        for j in range(i - lookback, i):
            klines.append([
                float(df.index[j].timestamp() * 1000),
                float(df.iloc[j]['open']),
                float(df.iloc[j]['high']),
                float(df.iloc[j]['low']),
                float(df.iloc[j]['close']),
                float(df.iloc[j]['volume'])
            ])
        
        # 预测
        try:
            result = predictor.predict(klines)
            
            if "error" in result:
                continue
            
            prediction = result["prediction"]
            trend_code = prediction["trend_code"]
            trend = prediction["trend"]
            confidence = prediction["confidence"]
            current_price = result["current_price"]
            
            # 更新入场价：只有当预测方向切换时才更新
            if last_trend is None or trend != last_trend:
                last_entry_price = current_price
                last_trend = trend
            entry_price = last_entry_price
            
            # 计算实际涨跌（次日）
            actual_change = None
            actual_trend = None
            if i + 1 < len(df):
                next_close = float(df.iloc[i + 1]['close'])
                actual_change = (next_close - entry_price) / entry_price * 100
                actual_trend = "涨" if actual_change > 0.1 else ("跌" if actual_change < -0.1 else "平")
            
            record = {
                "date": current_date.strftime("%Y-%m-%d"),
                "price": round(current_price, 2),  # 实时价（当前K线收盘价）
                "entry_price": round(entry_price, 2),  # 预测进入价（模型给出方向时的快照价格）
                "predicted_trend": trend,
                "trend_code": trend_code,
                "confidence": round(confidence, 2),
                "actual_change_pct": round(actual_change, 2) if actual_change is not None else None,
                "actual_trend": actual_trend,
                "correct": None
            }
            
            # 判断预测是否正确
            if actual_trend is not None:
                if trend == "看涨" and actual_trend == "涨":
                    record["correct"] = True
                elif trend == "看跌" and actual_trend == "跌":
                    record["correct"] = True
                elif trend == "中性" and actual_trend == "平":
                    record["correct"] = True
                else:
                    record["correct"] = False
            
            results.append(record)
            
            # 打印进度（每100条打印一次）
            if len(results) % 100 == 0:
                print(f"  已预测 {len(results)} 条...")
                
        except Exception as e:
            print(f"预测 {current_date.strftime('%Y-%m-%d')} 时出错: {e}")
            continue
    
    # 4. 输出结果
    if not results:
        print("未生成任何预测结果")
        return
    
    df_results = pd.DataFrame(results)
    
    # 如果启用 only_change，过滤掉连续相同的预测信号
    if only_change:
        filtered_results = []
        prev_trend = None
        for _, row in df_results.iterrows():
            current_trend = row['predicted_trend']
            if current_trend != prev_trend:
                filtered_results.append(row)
                prev_trend = current_trend
        df_results = pd.DataFrame(filtered_results)
        print(f"\n[信号过滤] 已启用 only_change 模式，仅输出信号变化点")
        print(f"原始记录数: {len(results)}, 过滤后记录数: {len(df_results)}")
    
    # 打印前20条和后20条
    print(f"\n{'='*60}")
    print(f"预测结果（前20条）")
    print(f"{'='*60}")
    print(df_results.head(20).to_string(index=False))
    
    print(f"\n{'='*60}")
    print(f"预测结果（后20条）")
    print(f"{'='*60}")
    print(df_results.tail(20).to_string(index=False))
    
    # 5. 统计准确率
    print(f"\n{'='*60}")
    print(f"回测统计")
    print(f"{'='*60}")
    
    total = len(df_results)
    valid_results = df_results[df_results['correct'].notna()]
    correct_count = valid_results['correct'].sum()
    
    print(f"总预测条数: {total}")
    print(f"有效预测数: {len(valid_results)}")
    print(f"正确预测数: {correct_count}")
    print(f"预测准确率: {correct_count / len(valid_results) * 100:.2f}%" if len(valid_results) > 0 else "N/A")
    
    # 按趋势统计
    print(f"\n按预测趋势统计:")
    for trend in ["看涨", "看跌", "中性"]:
        trend_df = df_results[df_results['predicted_trend'] == trend]
        trend_valid = trend_df[trend_df['correct'].notna()]
        if len(trend_valid) > 0:
            trend_correct = trend_valid['correct'].sum()
            print(f"  {trend}: {len(trend_valid)} 次, 准确率 {trend_correct / len(trend_valid) * 100:.2f}%")
        else:
            print(f"  {trend}: 0 次")
    
    # 按年份统计
    print(f"\n按年份统计:")
    df_results['year'] = pd.to_datetime(df_results['date']).dt.year
    for year in sorted(df_results['year'].unique()):
        year_df = df_results[df_results['year'] == year]
        year_valid = year_df[year_df['correct'].notna()]
        if len(year_valid) > 0:
            year_correct = year_valid['correct'].sum()
            print(f"  {year}: {len(year_valid)} 天, 准确率 {year_correct / len(year_valid) * 100:.2f}%")
    
    # 6. 保存到文件
    if output_file:
        df_results.to_csv(output_file, index=False)
        print(f"\n结果已保存到: {output_file}")
    else:
        # 默认保存到 data 目录
        os.makedirs("data", exist_ok=True)
        default_file = f"data/backtest_{symbol}_{interval}_{start_date}_{end_date}.csv"
        df_results.to_csv(default_file, index=False)
        print(f"\n结果已保存到: {default_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM 预测信号历史回测")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="交易对 (默认: BTCUSDT)")
    parser.add_argument("--start", type=str, default="2020-03-01", help="开始日期 YYYY-MM-DD (默认: 2020-03-01)")
    parser.add_argument("--end", type=str, default="2025-04-01", help="结束日期 YYYY-MM-DD (默认: 2025-04-01)")
    parser.add_argument("--interval", type=str, default="1d", help="K线周期: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w (默认: 1d)")
    parser.add_argument("--lookback", type=int, default=120, help="模型回看窗口大小 (默认: 120)")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--only-change", action="store_true", help="只输出信号变化的记录（过滤连续相同信号）")
    parser.add_argument("--sensitive", action="store_true", help="使用敏感模式（更及时的信号切换）")
    
    args = parser.parse_args()
    
    run_backtest(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        interval=args.interval,
        lookback=args.lookback,
        output_file=args.output,
        only_change=args.only_change,
        sensitive=args.sensitive
    )
