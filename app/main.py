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

from models.enhanced_lstm import EnhancedLSTMPredictor, MIN_KLINES_FOR_PREDICT
from models.lstm_model import get_sample_klines
from services.self_learning_manager import SelfLearningManager
from services.signal_reversal import check_signal_reversal, effective_trend
from services.live_price import build_live_block
from services.signal_state_machine import (
    SignalStateMachine,
    StateMachineParams,
    STATE_POSITION,
    STATE_TO_TREND_CODE,
    replay as replay_state_machine,
)
from config import (
    drop_unclosed_klines,
    fetch_klines,
    get_all_klines,
    interval_to_ms,
    EXCHANGE,
    SIGNAL_REVERSAL_ENABLED,
    PREDICTION_HORIZON_BARS,
    SIGNAL_STATE_MACHINE_ENABLED,
    SIGNAL_STATE_MACHINE_APPLY,
    SIGNAL_STATE_MACHINE_REBUILD_BARS,
    STATE_MACHINE_PARAMS,
    STATE_MACHINE_USE_CALIBRATED,
)

# 预测默认取数根数：覆盖 60 步序列 + 60 根特征预热 + ewm 类指标（MACD span26/RSI span14）
# 的收敛需求。取值过少会让输入序列里出现被 fillna(0) 填平的滚动指标（见 models/enhanced_lstm）。
DEFAULT_PREDICT_KLINES = 300

app = FastAPI(
    title="LSTM 趋势预测 API",
    description="加密货币趋势预测服务（集成自我学习微调）",
    version="2.1.0"
)

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

def fetch_kline_data(
    symbol: str,
    interval: str = "1h",
    limit: int = DEFAULT_PREDICT_KLINES,
    drop_unclosed: bool = True,
    return_forming: bool = False,
):
    """获取 K 线数据（支持多交易所，统一输出 Binance 格式）

    ``drop_unclosed=True``（默认）会丢弃尚未收盘的当前 K 线：训练只见过完整 K 线，
    拿半根去做推理会造成 train/serve skew（详见 config.drop_unclosed_klines）。
    原始行情导出（/klines）可显式传 False 以包含实时半根。

    ``return_forming=True`` 时返回 ``(klines, forming_bar)``：``forming_bar`` 是被丢弃的
    那根未收盘 K 线（没有则 None）。**它只供仓位/风控层取实时价用，绝不可拼回 klines
    再喂给模型**。这样做到零额外延迟 —— 行情本来就已经取到了，只是此前直接丢掉。

    取数窗口按 **周期** 计算（此前硬编码为小时）：Binance 从 startTime 起向后返回 limit 根，
    所以窗口算错会同时错两头 —— 周期比 1h 细时拿到的是几天前的陈旧数据（1m 实测滞后 98 小时），
    周期比 1h 粗时根数不够（4h 只得 25 根、1d 只得 4 根）导致无法预测。

    另外多留 2 根并取尾部，避免起点未对齐时丢掉最新一根。
    """
    import time

    limit = max(int(limit), MIN_KLINES_FOR_PREDICT)
    interval_ms = interval_to_ms(interval)
    end_time = int(time.time() * 1000)
    fetch_bars = limit + 2  # 对齐余量
    start_time = end_time - interval_ms * fetch_bars

    print(f"正在从 {EXCHANGE} 获取 {symbol} {interval} K线数据 (目标 {limit} 根)...")

    try:
        if fetch_bars <= 1000:
            data = fetch_klines(symbol, interval, start_time, end_time, fetch_bars)
        else:
            # 单次请求上限 1000 根，超出部分走分页
            data = get_all_klines(symbol, interval, start_time, end_time)

        # 提取前 6 列: [open_time, open, high, low, close, volume]
        klines = [[
            float(row[0]),  # open_time
            float(row[1]),  # open
            float(row[2]),  # high
            float(row[3]),  # low
            float(row[4]),  # close
            float(row[5]),  # volume
        ] for row in data]

        klines = klines[-limit:]  # 只保留最新的 limit 根

        dropped = False
        forming_bar = None
        if drop_unclosed:
            forming_bar = klines[-1] if klines else None
            klines, dropped = drop_unclosed_klines(klines, interval_ms)
            if not dropped:
                forming_bar = None   # 没丢任何东西，说明最后一根本来就已收盘

        print(
            f"获取到 {len(klines)} 条 K线数据 (周期 {interval}"
            f"{', 已丢弃未收盘的当前 K 线' if dropped else ''})"
        )
        if return_forming:
            return klines, forming_bar
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
            klines = [[float(i) for i in row] for row in get_sample_klines(request.symbol, DEFAULT_PREDICT_KLINES)]
        else:
            klines = [[float(i) for i in row] for row in request.klines]
        
        # 使用 wrap_predict 实现：预测 + 记录
        prediction = manager.wrap_predict(klines)
        weights = manager.predictor.get_feature_importance_report()
        
        if "error" in prediction:
             raise HTTPException(status_code=400, detail=prediction["error"])

        # 信号反转检测（与 GET 同一套逻辑，基准价来自本次预测所用的 K 线）
        result = {
            "success": True,
            "symbol": request.symbol,
            "current_price": klines[-1][4] if klines else None,
            # 说明 current_price 的口径，避免被当成"现价"（见下面的 live 块）
            "current_price_basis": PRICE_BASIS_CLOSED_BAR,
            "prediction": prediction,
            "feature_weights": weights
        }
        result.update(horizon_block("1h"))
        result.update(build_position_state(request.symbol, "1h", klines,
                                           manager.predictor))
        # 顺序要紧：先把 live 算出来，signal_reversal 的 message 才能带上实时价
        # POST 的 klines 由调用方提供、不含周期信息，与 wrap_predict 一样按 1h 处理。
        # 调用方给的 klines 里可能已含未收盘那根，这里无法可靠识别，故单独取一次实时价。
        result.update(build_live_block(request.symbol, "1h"))
        result.update(build_signal_block(klines, prediction, result["live"]["price"]))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.get("/predict")
async def predict_get(symbol: str = Query("BTCUSDT"), interval: str = Query("1h")):
    """预测趋势（GET 方式 - 集成记录）"""
    try:
        klines, forming_bar = fetch_kline_data(
            symbol, interval, DEFAULT_PREDICT_KLINES, return_forming=True)
        manager = get_manager(symbol)
        
        # 使用 wrap_predict 实现：预测 + 记录（记录 interval 便于闭环按视界核对）
        prediction = manager.wrap_predict(klines, interval=interval)
        weights = manager.predictor.get_feature_importance_report()
        
        if "error" in prediction:
             raise HTTPException(status_code=400, detail=prediction["error"])
        
        current_price = klines[-1][4] if klines else 0
        
        result = {
            "success": True,
            "symbol": symbol,
            "interval": interval,
            "current_price": current_price,
            # 说明 current_price 的口径，避免被当成"现价"（见下面的 live 块）
            "current_price_basis": PRICE_BASIS_CLOSED_BAR,
            "prediction": prediction,
            "feature_weights": weights
        }
        result.update(horizon_block(interval))
        result.update(build_position_state(symbol, interval, klines,
                                           manager.predictor))
        # 顺序要紧：先把 live 算出来，signal_reversal 的 message 才能带上实时价
        result.update(build_live_block(symbol, interval, forming_bar))
        result.update(build_signal_block(klines, prediction, result["live"]["price"]))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


def horizon_block(interval: str = "1h") -> dict:
    """声明本次预测的**视界** —— 方向只针对未来这么多时间，不是多日趋势。

    实测：17 维特征在 h=1..24 上的方向命中率都贴着 50%（见 experiments/horizon_probe.py），
    而模型名字叫"趋势预测"、字段叫 trend，很容易被读成多日观点。线上真的出现过
    "BTC 两小时涨 3.6% 而模型报看跌"被当成矛盾/死抗的误读。故把视界显式写进响应。
    """
    try:
        bar_ms = interval_to_ms(interval)
    except Exception:
        bar_ms = 3600 * 1000
    minutes = int(bar_ms * PREDICTION_HORIZON_BARS / 60000)
    return {
        "prediction_horizon_bars": PREDICTION_HORIZON_BARS,
        "prediction_horizon_minutes": minutes,
        "prediction_scope": (
            f"方向针对**下一根 K 线**（{interval} × {PREDICTION_HORIZON_BARS} = "
            f"{minutes} 分钟），不是多日趋势判断"
        ),
    }


# 状态机结果缓存：状态只在**新的已收盘 K 线**出现时才会变，
# 因此同一根 K 线内重复调用必须复用，否则每次请求都要多花 ~0.6~0.9s 做批量推理。
_POSITION_STATE_CACHE: dict = {}


# current_price 的口径声明：它是**最后一根已收盘 K 线**的收盘价，不是实时价。
# 线上实测过它滞后 0~59 分钟（1h 周期）；调用方若拿它做实时判断，在瀑布行情里会被误导。
PRICE_BASIS_CLOSED_BAR = "last_closed_bar_close"


def atr_series(bars: list, period: int = 14) -> list:
    """简单 ATR（真实波幅的滚动均值）—— 供状态机的移动失效线使用。

    生产模型**不用** ATR 做特征（它算出 atr 但没进 17 维，见 README 特征一节），
    这里只是给风控层的失效线提供"绝对价格单位"的波动度量。
    """
    trs, out = [], []
    for i, b in enumerate(bars):
        high, low, prev_close = float(b[2]), float(b[3]), float(b[4 - 1]) if i == 0 else float(bars[i - 1][4])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
        window = trs[-period:]
        out.append(sum(window) / len(window))
    return out


def _resolved_params(symbol: str) -> dict:
    """状态机参数：优先用训练期标定的阈值，缺失才用 config 兜底。

    为什么必须标定：模型 directional_score 实测只在 47.5~51.9 之间，
    任何"看起来合理"的固定阈值（60%、70/30、55/45）实际上永远触发不了 ——
    旧 trend_manager 就是因此方向输出恒为 0（准确率掉到 32.23%），
    本模块第一版用写死的 55/45 同样跑出 240 根零切换。
    """
    ps = dict(STATE_MACHINE_PARAMS)
    if not STATE_MACHINE_USE_CALIBRATED:
        return ps
    try:
        from services.model_meta import load_train_meta

        meta = load_train_meta(symbol) or {}
        cal = meta.get("calibrated_thresholds") or {}
        bull, bear = cal.get("bullish"), cal.get("bearish")
        if bull is not None and bear is not None and bear < bull:
            ps["bull_enter"] = float(bull)
            ps["bear_enter"] = float(bear)
            # 退出线取两者中点：迟滞带 = 进入线到中线
            mid = (float(bull) + float(bear)) / 2.0
            ps["bull_exit"] = mid
            ps["bear_exit"] = mid
    except Exception as e:      # pragma: no cover
        print(f"读取标定阈值失败，使用 config 兜底: {e}")
    return ps


def build_position_state(symbol: str, interval: str, klines: list, predictor) -> dict:
    """用状态机重建当前位置状态（每次都从最近 N 根已收盘 K 线确定性重放）。

    为什么每请求重放而不是持久化进程内状态：服务会重启，落盘还要处理与 K 线数据
    不一致的问题；重放只要输入相同结果就相同。代价是一次批量推理（见
    EnhancedLSTMPredictor.batch_directional_scores）。
    """
    if not SIGNAL_STATE_MACHINE_ENABLED:
        return {"position_state": {"enabled": False}}
    import pandas as pd

    # 不假设调用方已经加载过模型：缺 scaler / 权重统计量时自行加载。
    # （此前隐式依赖 wrap_predict 先跑，直接调用会因 self.scaler is None 崩掉。）
    if getattr(predictor, "scaler", None) is None:
        predictor.load_model()

    n = len(klines)
    params_dict = _resolved_params(symbol)
    cache_key = (int(klines[-1][0]) if klines else None, interval,
                 tuple(sorted(params_dict.items())))
    hit = _POSITION_STATE_CACHE.get(symbol)
    if hit and hit[0] == cache_key:
        return {"position_state": dict(hit[1], cached=True)}

    look = min(int(SIGNAL_STATE_MACHINE_REBUILD_BARS), n)
    offset = n - look                      # 重放窗口的起点
    atrs = atr_series(klines)
    scores = predictor.batch_directional_scores(
        pd.DataFrame(klines, columns=["open_time", "open", "high", "low",
                                      "close", "volume"]),
        list(range(offset, n)),
    )
    bars = [
        {"close": float(klines[i][4]), "high": float(klines[i][2]),
         "low": float(klines[i][3]), "atr": atrs[i]}
        for i in range(offset, n)
    ]

    def score_at(j: int) -> float:
        return scores.get(offset + j, 50.0)

    params = StateMachineParams(**params_dict)
    sm = replay_state_machine(bars, score_at, params)
    snap = sm.snap

    state = snap.state
    apply_flag = bool(SIGNAL_STATE_MACHINE_APPLY)
    block = {
            "enabled": True,
            "applied": apply_flag,
            "state": state,
            "trend_code": STATE_TO_TREND_CODE[state],
            "position": STATE_POSITION[state],
            "state_since_bar_ms": (int(klines[offset + snap.state_since_bar][0])
                                   if 0 <= snap.state_since_bar < look else None),
            "bars_in_state": snap.bars_in_state,
            "score": round(snap.score, 2),
            "signal_price": snap.signal_price,
            "invalidation_price": (round(snap.invalidation_price, 2)
                                   if snap.invalidation_price is not None else None),
            "invalidations": sm.invalidations,
            "transitions": sm.transitions,
            "structure": snap.structure,
            "latched": snap.latched,
            "reason": snap.reason,
            "replayed_bars": look,
            "params": params.to_dict(),
            "note": ("状态机按已收盘 K 线逐根推进；WEAK_* 为预警，仓位仍维持原方向。"
                     "applied=false 时仅给建议，effective_signal 仍来自模型。"),
            "cached": False,
    }
    _POSITION_STATE_CACHE[symbol] = (cache_key, dict(block))
    return {"position_state": block}


def build_signal_block(
    klines: list,
    prediction: dict,
    live_price: Optional[float] = None,
) -> dict:
    """组装权威信号块。

    - ``prediction`` 始终是模型原始输出（不被静默改写）；
    - ``effective_signal`` 是下游应当采用的最终方向（唯一权威字段）；
    - ``signal_reversal`` 是反转判定明细（含是否真的覆盖了模型输出）。

    ``live_price`` **只用于 message 文案**：判定依旧只用已收盘 K 线。
    传进来是为了让消息里能同时看到"上根收盘价"和真实现价 —— 否则消息里的
    "现价"其实是上一根收盘价（滞后 0~59 分钟），与响应里的 ``live.price`` 矛盾。
    """
    model_trend_code = prediction["prediction"]["trend_code"]
    reversal = None
    if SIGNAL_REVERSAL_ENABLED:
        reversal = check_signal_reversal(klines, model_trend_code,
                                         live_price=live_price)

    block = {"effective_signal": effective_trend(model_trend_code, reversal)}
    if reversal is not None:
        block["signal_reversal"] = reversal
    return block


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
        klines = fetch_kline_data(request.symbol, request.interval, request.limit,
                                  drop_unclosed=False)
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
