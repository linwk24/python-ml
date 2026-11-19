"""
LSTM 趋势预测 API 服务
使用 FastAPI 提供预测接口，集成自我学习能力
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio

from models.enhanced_lstm import EnhancedLSTMPredictor
from models.lstm_model import get_sample_klines
from services.self_learning_manager import SelfLearningManager
from config import fetch_klines, format_symbol, EXCHANGE, SIGNAL_REVERSAL_ENABLED, SIGNAL_REVERSAL_THRESHOLD
import json

app = FastAPI(
    title="LSTM 趋势预测 API",
    description="加密货币趋势预测服务（集成自我学习微调）",
    version="2.1.0"
    
)


@app.on_event("startup")
def restore_models_from_r2():
    """Render 启动时恢复持久化模型"""

    print("=" * 50)
    print("正在检查 Cloudflare R2 模型存储...")

    try:
        from services.model_storage import download_models

        downloaded = download_models()

        print(
            f"模型恢复完成：{downloaded} 个文件"
        )

    except Exception as exc:
        print(f"模型恢复失败：{exc}")

        if os.getenv("R2_REQUIRED", "false").lower() == "true":
            raise

    print("=" * 50)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 存储自我学习管理器
# Key: symbol, Value: SelfLearningManager
managers_cache = {}

# 请求模型
class TrainRequest(BaseModel):
    symbol: str = "BTCUSDT"
    klines: Optional[List[List]] = None
    use_sample_data: bool = True
    epochs: int = 50
    batch_size: int = 32

class PredictRequest(BaseModel):
    symbol: str = "BTCUSDT"
    klines: Optional[List[List]] = None
    use_sample_data: bool = True

class BinanceKlinesRequest(BaseModel):
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    limit: int = 200

def get_manager(symbol: str) -> SelfLearningManager:
    """获取或创建自我学习管理器实例"""
    if symbol not in managers_cache:
        managers_cache[symbol] = SelfLearningManager(symbol=symbol)
    return managers_cache[symbol]

def fetch_kline_data(symbol: str, interval: str = "1h", limit: int = 200):
    """获取 K 线数据（支持多交易所，统一输出 Binance 格式）"""
    import time
    end_time = int(time.time() * 1000)
    start_time = end_time - limit * 60 * 60 * 1000  # 根据 limit 估算开始时间
    
    print(f"正在从 {EXCHANGE} 获取 {symbol} {interval} K线数据...")
    
    try:
        data = fetch_klines(symbol, interval, start_time, end_time, min(limit, 1000))
        
        # 提取前 6 列: [open_time, open, high, low, close, volume]
        klines = [[
            float(row[0]),  # open_time
            float(row[1]),  # open
            float(row[2]),  # high
            float(row[3]),  # low
            float(row[4]),  # close
            float(row[5]),  # volume
        ] for row in data]
        
        print(f"获取到 {len(klines)} 条 K线数据")
        return klines
        
    except Exception as e:
        print(f"获取 K线数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取K线数据失败: {str(e)}")

def format_history(history):
    """将 Keras 训练历史对象转换为 JSON 可序列化的字典"""
    if history is None:
        return {}
    h = history.history if hasattr(history, 'history') else history
    if isinstance(h, dict):
        return {k: [float(v) for v in vals] for k, vals in h.items()}
    return str(h)

# ========== API 接口 ==========

@app.get("/health")
async def health():
    """健康检查"""
    manager_count = len(managers_cache)
    return {
        "status": "healthy",
        "managers_loaded": manager_count,
        "symbols": list(managers_cache.keys()) if manager_count > 0 else []
    }

@app.get("/features")
async def get_features(symbol: str = "BTCUSDT"):
    """获取当前特征权重"""
    manager = get_manager(symbol)
    weights = manager.predictor.get_feature_importance_report()
    return {
        "symbol": symbol,
        "features": weights
    }

@app.post("/train")
async def train_model_post(request: TrainRequest):
    """训练模型（POST 方式）"""
    try:
        manager = get_manager(request.symbol)
        
        if request.use_sample_data or request.klines is None:
            print(f"使用模拟数据训练 {request.symbol}...")
            klines = [[float(i) for i in row] for row in get_sample_klines(request.symbol, 200)]
        else:
            klines = [[float(i) for i in row] for row in request.klines]
        
        print(f"开始全量训练 {request.symbol} 模型...")
        # 调用 train 且 is_fine_tune=False (默认全量训练)
        result = manager.predictor.train(klines, epochs=request.epochs, batch_size=request.batch_size, is_fine_tune=False)
        weights = manager.predictor.get_feature_importance_report()
        
        return {
            "success": True,
            "symbol": request.symbol,
            "message": "模型全量训练完成",
            "training": format_history(result),
            "feature_weights": weights
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"训练失败: {str(e)}")

@app.get("/train")
async def train_model_get(symbol: str = Query("BTCUSDT"), interval: str = Query("1h"), epochs: int = Query(50)):
    """训练模型（GET 方式）"""
    try:
        klines = fetch_kline_data(symbol, interval, 500)
        manager = get_manager(symbol)
        print(f"开始训练 {symbol} 模型...")
        result = manager.predictor.train(klines, epochs=epochs, is_fine_tune=False)
        weights = manager.predictor.get_feature_importance_report()
        
        return {
            "success": True,
            "symbol": symbol,
            "interval": interval,
            "message": "模型训练完成",
            "data_count": len(klines),
            "training": format_history(result),
            "feature_weights": weights
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"训练失败: {str(e)}")

@app.post("/predict")
async def predict_post(request: PredictRequest):
    """预测趋势（POST 方式 - 集成记录）"""
    try:
        manager = get_manager(request.symbol)
        
        if request.use_sample_data or request.klines is None:
            klines = [[float(i) for i in row] for row in get_sample_klines(request.symbol, 100)]
        else:
            klines = [[float(i) for i in row] for row in request.klines]
        
        # 使用 wrap_predict 实现：预测 + 记录
        prediction = manager.wrap_predict(klines)
        weights = manager.predictor.get_feature_importance_report()
        
        if "error" in prediction:
             raise HTTPException(status_code=400, detail=prediction["error"])
             
        return {
            "success": True,
            "symbol": request.symbol,
            "prediction": prediction,
            "feature_weights": weights
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.get("/predict")
async def predict_get(symbol: str = Query("BTCUSDT"), interval: str = Query("1h")):
    """预测趋势（GET 方式 - 集成记录）"""
    try:
        klines = fetch_kline_data(symbol, interval, 100)
        manager = get_manager(symbol)
        
        # 使用 wrap_predict 实现：预测 + 记录
        prediction = manager.wrap_predict(klines)
        weights = manager.predictor.get_feature_importance_report()
        
        if "error" in prediction:
             raise HTTPException(status_code=400, detail=prediction["error"])
        
        current_price = klines[-1][4] if klines else 0
        current_trend = prediction["prediction"]["trend_code"]
        
        # 信号反转检测
        reversal_info = None
        if SIGNAL_REVERSAL_ENABLED:
            reversal_info = check_signal_reversal(symbol, current_price, current_trend)
        
        result = {
            "success": True,
            "symbol": symbol,
            "interval": interval,
            "current_price": current_price,
            "prediction": prediction,
            "feature_weights": weights
        }
        
        if reversal_info:
            result["signal_reversal"] = reversal_info
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


def check_signal_reversal(symbol: str, current_price: float, current_trend: int) -> dict:
    """
    检查信号反转
    当预测方向与价格走势矛盾时触发反转
    
    逻辑：
    - 价格涨了，但预测看跌 → 反转为看涨
    - 价格跌了，但预测看涨 → 反转为看跌
    """
    predictions_file = f"predictions_{symbol}.json"
    
    if not os.path.exists(predictions_file):
        return None
    
    try:
        with open(predictions_file, 'r') as f:
            data = json.load(f)
        
        if not data:
            return None
        
        # 获取倒数第二条预测记录（最后一条是当前预测，已经写入）
        if len(data) < 2:
            return None  # 没有历史预测记录
        
        last_prediction = data[-2]  # 倒数第二条是上次预测
        last_trend = last_prediction.get("trend_code")
        last_price = last_prediction.get("price")  # 上次的 entry_price
        
        if last_trend is None or last_price is None:
            return None
        
        # 计算价格变化（相对于上次 entry_price）
        price_change = (current_price - last_price) / last_price
        
        # 检查价差是否超过阈值
        if abs(price_change) <= SIGNAL_REVERSAL_THRESHOLD:
            return {
                "triggered": False,
                "last_trend": last_trend,
                "last_price": last_price,
                "current_price": current_price,
                "price_change": f"{price_change:.2%}",
                "threshold": f"{SIGNAL_REVERSAL_THRESHOLD:.2%}",
                "message": f"价差 {abs(price_change):.2%} <= {SIGNAL_REVERSAL_THRESHOLD:.2%}"
            }
        
        # 检查预测方向与价格走势是否矛盾
        # 情况1: 价格涨了，但当前预测看跌 → 应该反转为看涨
        if price_change > 0 and current_trend == 0:
            return {
                "triggered": True,
                "type": "reverse_to_bullish",
                "action": "反转预测为看涨",
                "last_trend": last_trend,
                "last_price": last_price,
                "current_price": current_price,
                "price_change": f"+{price_change:.2%}",
                "current_prediction": "看跌",
                "corrected_prediction": "看涨",
                "threshold": f"{SIGNAL_REVERSAL_THRESHOLD:.2%}",
                "message": f"价格涨了 {price_change:.2%}，但预测看跌，应反转为看涨"
            }
        
        # 情况2: 价格跌了，但当前预测看涨 → 应该反转为看跌
        if price_change < 0 and current_trend == 2:
            return {
                "triggered": True,
                "type": "reverse_to_bearish",
                "action": "反转预测为看跌",
                "last_trend": last_trend,
                "last_price": last_price,
                "current_price": current_price,
                "price_change": f"{price_change:.2%}",
                "current_prediction": "看涨",
                "corrected_prediction": "看跌",
                "threshold": f"{SIGNAL_REVERSAL_THRESHOLD:.2%}",
                "message": f"价格跌了 {price_change:.2%}，但预测看涨，应反转为看跌"
            }
        
        # 预测方向与价格走势一致，无需反转
        return {
            "triggered": False,
            "last_trend": last_trend,
            "last_price": last_price,
            "current_price": current_price,
            "price_change": f"{price_change:.2%}",
            "current_prediction": "看涨" if current_trend == 2 else "看跌",
            "message": "预测方向与价格走势一致，无需反转"
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/predict/{symbol}")
async def predict_by_symbol(symbol: str):
    """通过 URL 路径预测"""
    return await predict_get(symbol=symbol, interval="1h")

@app.get("/self-learn")
async def trigger_self_learn(symbol: str = Query("BTCUSDT"), interval: str = Query("1h")):
    """手动触发自我学习周期"""
    try:
        klines = fetch_kline_data(symbol, interval, 500)
        current_price = klines[-1][4] if klines else 0
        manager = get_manager(symbol)
        
        result = manager.run_self_learning_cycle(klines, current_price)
        return {
            "success": True,
            "symbol": symbol,
            "learning_result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"自我学习失败: {str(e)}")

@app.post("/klines")
async def get_klines(request: BinanceKlinesRequest):
    """获取 K 线数据"""
    try:
        klines = fetch_kline_data(request.symbol, request.interval, request.limit)
        return {
            "success": True,
            "symbol": request.symbol,
            "interval": request.interval,
            "count": len(klines),
            "data": klines[-60:] if len(klines) > 60 else klines
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取 K 线失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("LSTM 趋势预测 API 服务 v2.1")
    print(f"11项技术指标 + 自适应权重 + 自我学习闭环")
    print("=" * 50)
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
