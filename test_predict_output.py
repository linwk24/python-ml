"""
模型输出测试脚本 - 连续输出版本（不去重）
每一条预测都输出，便于调试和分析
"""
import pandas as pd
import numpy as np
from models.enhanced_lstm import EnhancedLSTMPredictor
from datetime import datetime, timedelta
import requests
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fetch_klines(symbol: str, start_date: str, end_date: str, interval: str = "1h"):
    """
    从 Binance 获取 K 线数据
    """
    print(f"正在获取 {symbol} 从 {start_date} 到 {end_date} 的数据...")

    url = "https://api.binance.com/api/v3/klines"

    # 向前多取 7 天数据用于模型初始化
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    extended_start = start_dt - timedelta(days=7)
    start_ts = int(extended_start.timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    all_data = []
    current_start = start_ts

    while current_start < end_ts:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": current_start,
            "limit": 1000
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if not data:
                break

            all_data.extend(data)
            current_start = data[-1][0] + 1

            if len(data) < 1000:
                break

        except Exception as e:
            print(f"获取数据失败: {e}")
            break

    if not all_data:
        print("未获取到数据")
        return []

    # 转换为模型需要的格式
    klines = []
    for d in all_data:
        klines.append([
            d[0],                           # timestamp
            float(d[1]),                    # open
            float(d[2]),                    # high
            float(d[3]),                    # low
            float(d[4]),                    # close
            float(d[5])                     # volume
        ])

    print(f"成功获取 {len(klines)} 条 K 线数据")
    return klines


def test_model_output(symbol: str = "BTCUSDT",
                      start_date: str = "2026-01-01",
                      end_date: str = "2026-01-31",
                      interval: str = "1h"):
    """
    在指定日期范围内测试模型预测输出
    连续输出所有预测（不去重），便于调试
    """
    print("=" * 80)
    print(f"模型输出测试 - {symbol}")
    print(f"时间范围: {start_date} 至 {end_date}")
    print(f"K线周期: {interval}")
    print("=" * 80)

    # 1. 初始化模型
    print(f"\n正在初始化模型 {symbol}...")
    predictor = EnhancedLSTMPredictor(symbol=symbol)

    # 检查模型是否存在
    model_path = f"models/models/{symbol}_enhanced_lstm.h5"
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        print("请先训练模型")
        return

    if not predictor.load_model():
        print(f"❌ 模型加载失败")
        return

    print(f"✅ 模型加载成功")

    # 2. 获取历史数据
    klines = fetch_klines(symbol, start_date, end_date, interval)

    if not klines:
        print("❌ 未获取到有效数据")
        return

    # 3. 计算目标开始时间戳
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    target_start_ts = int(start_dt.timestamp() * 1000)

    # 4. 逐条预测并输出
    print(f"\n{'时间':<20} | {'价格':<10} | {'置信度':<8} | {'趋势':<6} | {'进入价':<12}")
    print("-" * 70)

    # 统计变量
    total_predictions = 0
    trend_counts = {'看涨': 0, '看跌': 0, '中性': 0}

    # 需要至少 120 条数据才能预测
    for i in range(120, len(klines)):
        # 模拟实时：每次只给模型看当前窗口
        window = klines[:i]

        try:
            res = predictor.predict(window)
        except Exception as e:
            continue

        if "error" in res:
            continue

        # 检查是否在目标日期范围内
        if klines[i-1][0] < target_start_ts:
            continue

        # 获取预测结果
        timestamp = datetime.fromtimestamp(klines[i-1][0]/1000).strftime('%Y-%m-%d %H:%M')
        price = res["current_price"]
        pred = res["prediction"]

        trend = pred['trend']
        confidence = pred['confidence']
        entry_price = pred.get('entry_price', 'N/A')

        # 输出每一条预测
        print(f"{timestamp:<20} | {price:<10.2f} | {confidence:<8.2f} | {trend:<6} | {entry_price:<12}")

        # 统计
        total_predictions += 1
        trend_counts[trend] = trend_counts.get(trend, 0) + 1

    # 打印统计信息
    print("\n" + "=" * 80)
    print("📊 统计信息")
    print("=" * 80)
    print(f"总预测次数: {total_predictions}")
    print(f"\n各趋势预测次数:")
    for trend, count in trend_counts.items():
        pct = count / total_predictions * 100 if total_predictions > 0 else 0
        print(f"  {trend}: {count} 次 ({pct:.1f}%)")

    print("=" * 80)
    print("测试完成")


def test_model_output_with_save(symbol: str = "BTCUSDT",
                                 start_date: str = "2026-01-01",
                                 end_date: str = "2026-01-31",
                                 interval: str = "1h",
                                 output_file: str = None):
    """
    连续输出版本 + 保存到文件
    """
    print("=" * 80)
    print(f"模型输出测试（保存版）- {symbol}")
    print(f"时间范围: {start_date} 至 {end_date}")
    print("=" * 80)

    # 初始化模型
    predictor = EnhancedLSTMPredictor(symbol=symbol)

    model_path = f"models/models/{symbol}_enhanced_lstm.h5"
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        return

    if not predictor.load_model():
        print(f"❌ 模型加载失败")
        return

    print(f"✅ 模型加载成功")

    # 获取数据
    klines = fetch_klines(symbol, start_date, end_date, interval)
    if not klines:
        return

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    target_start_ts = int(start_dt.timestamp() * 1000)

    # 存储结果
    results = []

    print(f"\n正在预测...")

    for i in range(120, len(klines)):
        window = klines[:i]

        try:
            res = predictor.predict(window)
        except Exception:
            continue

        if "error" in res:
            continue

        if klines[i-1][0] < target_start_ts:
            continue

        timestamp = datetime.fromtimestamp(klines[i-1][0]/1000)
        price = res["current_price"]
        pred = res["prediction"]

        results.append({
            'timestamp': timestamp,
            'datetime': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'price': price,
            'confidence': pred['confidence'],
            'trend': pred['trend'],
            'trend_code': pred['trend_code'],
            # probabilities 现为数值(0-100)；展示用字符串在 probabilities_pct
            'prob_bull': float(pred['probabilities']['看涨']),
            'prob_neutral': float(pred['probabilities']['中性']),
            'prob_bear': float(pred['probabilities']['看跌'])
        })

    # 转换为 DataFrame
    df = pd.DataFrame(results)

    # 打印到控制台
    print(f"\n{'时间':<20} | {'价格':<10} | {'置信度':<8} | {'趋势':<6} | {'看涨':<6} | {'中性':<6} | {'看跌':<6}")
    print("-" * 80)

    for _, row in df.iterrows():
        print(f"{row['datetime']:<20} | {row['price']:<10.2f} | {row['confidence']:<8.2f} | "
              f"{row['trend']:<6} | {row['prob_bull']:<6.1f} | {row['prob_neutral']:<6.1f} | {row['prob_bear']:<6.1f}")

    # 统计信息
    print("\n" + "=" * 80)
    print("📊 统计信息")
    print("=" * 80)
    print(f"总预测次数: {len(df)}")
    print(f"\n各趋势预测次数:")
    for trend, count in df['trend'].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {trend}: {count} 次 ({pct:.1f}%)")

    print(f"\n置信度统计:")
    print(f"  平均置信度: {df['confidence'].mean():.2f}%")
    print(f"  最高置信度: {df['confidence'].max():.2f}%")
    print(f"  最低置信度: {df['confidence'].min():.2f}%")

    # 保存到文件
    if output_file:
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n💾 结果已保存到: {output_file}")

    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="模型输出测试（连续输出）")
    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="交易对")
    parser.add_argument("--start", type=str, default="2026-05-01", help="开始日期")
    parser.add_argument("--end", type=str, default="2026-05-15", help="结束日期")
    parser.add_argument("--interval", type=str, default="1h", help="K线周期")
    parser.add_argument("--save", type=str, default=None, help="保存到CSV文件")

    args = parser.parse_args()

    if args.save:
        test_model_output_with_save(args.symbol, args.start, args.end, args.interval, args.save)
    else:
        test_model_output(args.symbol, args.start, args.end, args.interval)