import json
import os
from datetime import datetime

class PredictionTracker:
    def __init__(self, symbol="BTCUSDT"):
        self.file_path = f"predictions_{symbol}.json"
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                json.dump([], f)

    def record_prediction(self, symbol, current_price, trend_code, confidence):
        """记录一次预测 (仅在 trend_code 变化时记录)"""
        with open(self.file_path, 'r+') as f:
            data = json.load(f)
            
            # 只有在文件为空，或者当前趋势与最后一次记录的趋势不同时才记录
            if not data or data[-1].get('trend_code') != trend_code:
                data.append({
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "price": current_price,
                    "trend_code": trend_code,
                    "confidence": confidence,
                    "actual": None  # 待核对
                })
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()
            else:
                # 趋势未变化，不记录快照
                pass

    def verify_predictions(self, symbol, current_price):
        """对比预测结果与当前真实价格，返回需要学习的样本"""
        verified_samples = []
        with open(self.file_path, 'r+') as f:
            data = json.load(f)
            updated = False
            for entry in data:
                if entry["actual"] is None:
                    # 计算实际走势 (简单逻辑: 涨0.2%看涨, 跌0.2%看跌)
                    change = (current_price - entry["price"]) / entry["price"]
                    if change > 0.002: actual = 2
                    elif change < -0.002: actual = 0
                    else: actual = 1
                    
                    entry["actual"] = actual
                    updated = True
                    
                    # 如果预测错误，将其标记为需要学习的样本
                    if entry["trend_code"] != actual:
                        verified_samples.append(entry)
            
            if updated:
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()
        
        return verified_samples
