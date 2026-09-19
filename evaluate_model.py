import sys

import numpy as np
import pandas as pd
import requests
import datetime
import argparse
from config import LABEL_THRESHOLD
from models.enhanced_lstm import EnhancedLSTMPredictor
from services.model_meta import guard_in_sample_window
from services.backtest_utils import next_bar_label

def fetch_binance_data(symbol: str, start_date: str, end_date: str):
    print(f"正在获取 {symbol} 历史数据以评估精度...")
    url = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime.datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    
    all_data = []
    current_start = start_ts
    while current_start < end_ts:
        params = {"symbol": symbol.upper(), "interval": "1h", "startTime": current_start, "endTime": end_ts, "limit": 1000}
        resp = requests.get(url, params=params)
        data = resp.json()
        if not data: break
        all_data.extend(data)
        current_start = data[-1][0] + 1
        if len(data) < 1000: break

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def evaluate_accuracy(symbol, start_date, end_date, allow_in_sample: bool = False):
    # 0. 口径检查：评估区间与模型训练区间重叠时，准确率是 in-sample 结果
    if not guard_in_sample_window(symbol, start_date, end_date, allow_in_sample):
        sys.exit(2)  # 非零退出码，便于脚本/CI 察觉"评估被口径检查拦截"

    # 1. 初始化模型
    predictor = EnhancedLSTMPredictor(symbol=symbol)
    
    # 2. 获取数据
    df = fetch_binance_data(symbol, start_date, end_date)
    if df.empty:
        print("未获取到数据")
        return

    # 为了预测未来 1 小时的走势，我们需要准备数据
    # 我们从第 120 根线开始，每隔 1 根线预测一次
    results = []
    total_steps = len(df)
    
    print(f"开始评估... 总数据量: {total_steps}, 预测样本数: {total_steps - 120}")
    
    for i in range(120, total_steps - 1):
        # 提取当前时刻之前的 120 根线作为输入
        window = df.iloc[i-120:i].reset_index()
        klines = []
        for _, row in window.iterrows():
            klines.append([
                row['timestamp'].timestamp() * 1000, 
                row['open'], row['high'], row['low'], row['close'], row['volume']
            ])
        
        # 模型预测
        pred = predictor.predict(klines)
        if "error" in pred: continue
        
        # 真实标签必须对齐模型的目标：窗口 = df.iloc[i-120:i]（最后一根 i-1），
        # 训练标签是"行 i-1 -> 行 i"。原来用 (close[i+1] vs close[i])，整体错后一根。
        _, actual_change, _ = next_bar_label(df, i)
        actual_change = actual_change / 100.0
        if actual_change > LABEL_THRESHOLD: actual_label = 2      # 涨
        elif actual_change < -LABEL_THRESHOLD: actual_label = 0   # 跌
        else: actual_label = 1                                    # 中性
        
        pred_label = pred["prediction"]["trend_code"]
        results.append({'pred': pred_label, 'actual': actual_label})
        
        if i % 100 == 0:
            print(f"进度: {i}/{total_steps}...")

    # 3. 计算指标
    df_res = pd.DataFrame(results)
    correct = (df_res['pred'] == df_res['actual']).sum()
    total = len(df_res)
    accuracy = correct / total * 100 if total > 0 else 0
    
    # 分类精度
    def get_precision(label_name):
        mask = df_res['pred'] == label_name
        if mask.sum() == 0: return 0
        return (df_res[mask]['actual'] == label_name).sum() / mask.sum() * 100

    bull_prec = get_precision(2)
    bear_prec = get_precision(0)
    neut_prec = get_precision(1)

    print("\n" + "="*40)
    print(f" 模型精度评估报告 - {symbol}")
    print(f" 测试期间: {start_date} 至 {end_date}")
    print(f" 样本总量: {total}")
    print("-" * 40)
    print(f" 总体准确率 (Overall Accuracy): {accuracy:.2f}%")
    print(f" 看涨预测命中率 (Bullish Precision): {bull_prec:.2f}%")
    print(f" 看跌预测命中率 (Bearish Precision): {bear_prec:.2f}%")
    print(f" 中性预测命中率 (Neutral Precision): {neut_prec:.2f}%")
    print("="*40)
    
    # 打印混淆矩阵简单版
    print("\n[预测结果分布]")
    print(df_res['pred'].value_counts().to_string())
    print("\n[实际结果分布]")
    print(df_res['actual'].value_counts().to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--allow-in-sample", action="store_true",
                        help="允许评估区间与训练区间重叠（结果只能标注为 in-sample）")
    args = parser.parse_args()
    evaluate_accuracy(args.symbol, args.start, args.end, allow_in_sample=args.allow_in_sample)
