import os
import numpy as np
import pandas as pd
import requests
import datetime
import argparse
from models.enhanced_lstm import EnhancedLSTMPredictor

# 优化后的超参数
LEARNING_RATE = 0.0001
EPOCHS = 150
BATCH_SIZE = 64

def fetch_long_term_data(symbol: str, days: int = 365):
    """获取长周期历史数据用于基础训练"""
    print(f"🚀 正在抓取过去 {days} 天的 {symbol} 全量数据...")
    url = "https://api.binance.com/api/v3/klines"
    end_ts = int(datetime.datetime.now().timestamp() * 1000)
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
    
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
    print(f"✅ 成功获取 {len(df)} 条 K 线数据")
    return df

def main():
    parser = argparse.ArgumentParser(description="Model Redemption Plan")
    parser.add_argument("--symbol", type=str, default="SOLUSDT")
    parser.add_argument("--days", type=int, default=365, help="训练数据覆盖天数")
    args = parser.parse_args()
    
    symbol = args.symbol
    
    # 1. 彻底清理旧权重
    print("\n" + "="*50)
    print("Step 1: 清理旧权重，准备重新开始...")
    model_dir = f"models/models/"
    for f in os.listdir(model_dir):
        if symbol in f:
            os.remove(os.path.join(model_dir, f))
    print(f"🧹 已清除 {symbol} 的所有旧模型文件。")
    
    # 2. 获取长周期训练数据
    df = fetch_long_term_data(symbol, args.days)
    
    # 将 DataFrame 转换为 predictor 需要的 klines 格式 (List of Lists)
    klines = []
    for idx, row in df.iterrows():
        klines.append([
            idx.timestamp() * 1000, 
            row['open'], row['high'], row['low'], row['close'], row['volume']
        ])
    
    # 3. 初始化并训练模型
    print("\n" + "="*50)
    print("Step 2: 开始全量训练 (优化超参数)...")
    predictor = EnhancedLSTMPredictor(symbol=symbol)
    
    # 注入优化后的学习率
    # 注意：由于 predictor.train 内部定义了 optimizer，我们直接调用
    try:
        # 我们调用 train 之前先手动修改内部构建模型的逻辑
        # 为了简单，我们通过 predictor 实例直接训练
        # 提示：我们在 enhanced_lstm.py 中已经修改了阈值
        history = predictor.train(
            klines=klines, 
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE, 
            is_fine_tune=False
        )
        print("✅ 训练完成！")
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        return

    # 4. 自动验证
    print("\n" + "="*50)
    print("Step 3: 自动验证模型精度...")
    # 这里我们利用 evaluate_model.py 的逻辑
    # 为了避免重复写代码，我们直接在 shell 中调用它
    import subprocess
    start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    cmd = f"./venv/bin/python evaluate_model.py --symbol {symbol} --start {start_date} --end {end_date}"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    print(stdout)
    if stderr:
        print(f"Error during evaluation: {stderr}")

if __name__ == "__main__":
    main()
