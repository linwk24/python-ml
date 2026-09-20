"""
交易所配置和数据获取模块
统一不同交易所的 API 接口，输出 Binance K线格式
"""

import os
import requests
import time
from typing import List, Optional

# ============== 路径配置 ==============
# 项目根目录：所有运行时产物（预测记录、日志、模型）都锚定到这里，
# 避免因启动时的工作目录不同而读写到不同文件。
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


def prediction_file_path(symbol: str) -> str:
    """预测记录文件的唯一权威路径（tracker 与 signal_reversal 必须共用）"""
    return os.path.join(PROJECT_ROOT, f"predictions_{symbol}.json")


# ============== 交易所配置 ==============
# 当前使用的交易所：binance / gate / okx
EXCHANGE = "binance"

# 交易所 API 配置
EXCHANGE_CONFIG = {
    "binance": {
        "base_url": "https://data-api.binance.vision",
        "klines_endpoint": "/api/v3/klines",
        "symbol_format": "{symbol}",  # BTCUSDT
    },
    "gate": {
        "base_url": "https://api.gateio.ws",
        "klines_endpoint": "/api/v4/spot/candlesticks",
        "symbol_format": "{symbol}",  # BTC_USDT
    },
    "okx": {
        "base_url": "https://www.okx.com",
        "klines_endpoint": "/api/v5/market/candles",
        "symbol_format": "{symbol}",  # BTC-USDT
    },
}

# ============== 实时价通路（仓位/风控层专用）==============
# 与模型输入**完全分离**的第二条通路。模型只用已收盘 K 线（drop_unclosed=True），
# 这条通路提供正在走的那根的实时价，供仓位层算未实现盈亏 / 判止损。
#
# 为什么不能把未收盘的那根拼进模型输入（实测，见 experiments/partial_bar_probe.py）：
#   - 半根 K 线的振幅被系统性压缩：同一批 60 小时里，走到第 30 分钟的半根
#     high_low_ratio 中位数只有完整 bar 的 67.5%，第 15 分钟只有 52.7%；
#   - 方向被改变的比例 13.3%（83 样本，95% 区间 [7.6%, 22.2%]），含完整反向；
#   - 根因是模型概率极平（约 35/31/34，仅略高于均匀 33.3），argmax 由噪声主导。
# 实时价是标量，没有训练分布问题，因此作为独立通路安全且可靠。
LIVE_PRICE_ENABLED = True          # 关掉可省一次 HTTP 往返
LIVE_PRICE_TIMEOUT = 5.0           # 秒；失败不影响 /predict 正常返回

# ============== 信号反转配置 ==============
# 前置动量校验：把"模型方向"与"最近 N 根 K 线的实际涨跌"对照，矛盾时给出反转建议。
#
# 基准价必须来自本次预测所用的 K 线（见 services/signal_reversal.py）。
# 历史实现曾用 predictions_{symbol}.json 里"倒数第二条记录"的价格当基准，而该记录只在
# 方向翻转时才落盘，基准价可能陈旧数月，线上实测出现过
#     last_price 64266.69（5 月记录） vs current_price 80624（+25.45%）
# 这种把三个月涨幅当成"当前反转信号"的荒唐结论。
SIGNAL_REVERSAL_ENABLED = True
SIGNAL_REVERSAL_THRESHOLD = 0.001  # 价差阈值 0.1% (0.001 = 0.1%)
SIGNAL_REVERSAL_LOOKBACK_BARS = 1  # 基准价取多少根 K 线之前（1 = 上一根收盘价）
# 是否让反转建议直接覆盖模型输出：
# 默认 False —— "价格动量推翻模型"是一条未经回测验证的经验规则（见 README 的风险提示），
# 不做静默改写。开启后接口的 effective_signal 会与模型原始 prediction 不一致，
# 但两者始终都会返回，便于对比/评估。
SIGNAL_REVERSAL_APPLY = False

# ============== 趋势状态机 ==============
# 状态机（services/trend_manager.py）是可选的方向平滑层，**默认关闭**。
#
# 为什么默认关闭（实测，12000 根 1h，留出段 3600 根 ≈ 5 个月，同一模型同一批概率）：
#     原始模型 argmax（无状态机） : 37.70% ~ 40.32%  （多数类基线 33.95%）
#     状态机 70/30（旧默认）      : 32.23%，方向输出 0%   —— 低于基线
#     状态机 55/45                : 32.20%，方向信号精度 23.1%（基础发生率 34.0%）
#     状态机 52/48                : 33.48%，方向信号精度 34.5%（基础发生率 34.0%）
#     状态机 51/49                : 34.76%，方向信号精度 35.7%
#   → **没有任何一种阈值配置能超过"直接用模型 argmax"**，所以默认不走状态机。
#   旧的 70/30 是沿用早期评分尺度（p_bull/(p_bull+p_bear)，铺满 0~100）定的；
#   现在的 directional_score 实际只在 43.6~58.0 之间，70/30 永远够不到，方向输出恒为 0。
#
# 打开后会使用训练时按模型自身分数分布标定出的阈值（见模型目录 *_train_meta.json 的
# calibrated_thresholds），而不是写死的 70/30。
TREND_MANAGER_ENABLED = False

TREND_BULLISH_THRESHOLD = 70.0   # 仅在状态机开启、且模型没有标定结果时使用
TREND_BEARISH_THRESHOLD = 30.0
TREND_ABSOLUTE_THRESHOLD = 85.0  # >= 该分数(或 <= 100-该值) -> 允许直接跨方向切换
TREND_PRICE_BREAK_PCT = 0.01     # 价格相对锚点破位幅度，配合模型同向可确认切换

# 标定用的分位数：bullish 取该分位、bearish 取 100-该分位
TREND_CALIBRATION_PERCENTILE = 90.0

# 敏感模式（更及时的信号切换，阈值仍关于 50 对称）
TREND_SENSITIVE_BULLISH_THRESHOLD = 60.0
TREND_SENSITIVE_BEARISH_THRESHOLD = 40.0
TREND_SENSITIVE_ABSOLUTE_THRESHOLD = 75.0
TREND_SENSITIVE_PRICE_BREAK_PCT = 0.005

# ============== 特征加权模式 ==============
# "none"        : 只用训练期固化统计量做标准化（默认）
# "mutual_info" : 在标准化**之后**再按训练段互信息缩放各特征
#
# 历史提醒：早期实现把指标权重乘在标准化之前，而那会被下游 StandardScaler
# 逐列精确抵消（实测极端不均匀权重与均匀权重的模型输入差异仅 3e-15）——即那一版
# 加权对模型毫无影响。权重只有作用在标准化之后才真正生效。
FEATURE_WEIGHTING = "none"

# ============== 标签 / 核对阈值（统一维护，不要在各文件里各写一套）==============
# 训练标签阈值：预测未来一根 K 线的涨跌幅超过 ±0.1% 才记为看涨/看跌，否则中性
LABEL_THRESHOLD = 0.001
# 核对阈值：判定预测对错时使用的涨跌阈值。
# 必须与训练标签阈值保持一致，否则"模型准确率"是被不同口径算出来的，不可比。
VERIFY_THRESHOLD = LABEL_THRESHOLD
# 预测视界：预测未来多少根 K 线（与 models/enhanced_lstm.PREDICTION_HORIZON 对齐）
PREDICTION_HORIZON_BARS = 1
# 兜底核对（拿不到 K 线时用"当前价"近似）允许的最大滞后倍数：
# 超过 horizon * (1 + 该值) 仍未核对的预测直接标记过期，绝不用陈旧价格去判定。
VERIFY_MAX_AGE_BARS = 3

# ============== 自我学习闭环配置 ==============
SELF_LEARNING_ACCURACY_THRESHOLD = 0.6  # 近期准确率低于该值才触发微调
SELF_LEARNING_WINDOW = 10               # 用最近 N 条已核对样本计算准确率
SELF_LEARNING_MIN_SAMPLES = 10          # 已核对样本少于该数量时不做任何判断（避免 1 错 1 就触发训练）

# ============== 当前生效的配置 ==============
CURRENT_CONFIG = EXCHANGE_CONFIG[EXCHANGE]
BASE_URL = CURRENT_CONFIG["base_url"]


# ============== Binance K线格式 ==============
# [
#   1499040000000,      # 开盘时间 (ms)
#   "0.01634000",       # 开盘价
#   "0.80000000",       # 最高价
#   "0.01575800",       # 最低价
#   "0.01577100",       # 收盘价
#   "148976.11427815",  # 成交量
#   1499644799999,      # 收盘时间 (ms)
#   "2434.19055334",    # 成交额
#   308,                # 成交笔数
#   "1756.87402397",    # 主动买入成交量
#   "28.46694368",      # 主动买入成交额
#   "17928899.62484339" # 忽略
# ]


def format_symbol(symbol: str) -> str:
    """
    格式化交易对符号，将 Binance 格式转换为各交易所格式
    
    Binance: BTCUSDT
    Gate.io: BTC_USDT
    OKX: BTC-USDT
    """
    # 如果是 Binance 格式，保持不变
    if EXCHANGE == "binance":
        return symbol
    
    # 解析 Binance 格式的交易对（如 BTCUSDT -> BTC, USDT）
    # 常见的 USDT 交易对
    stablecoins = ["USDT", "USDC", "BUSD", "DAI"]
    base = None
    quote = None
    
    for sc in stablecoins:
        if symbol.endswith(sc):
            base = symbol[:-len(sc)]
            quote = sc
            break
    
    if base is None:
        # 如果无法解析，尝试在中间分割
        for i, char in enumerate(symbol):
            if char.isalpha() and i > 0 and symbol[i-1].isalpha():
                # 找到字母边界
                pass
        # 默认假设最后 4 个字符是 quote
        base = symbol[:-4]
        quote = symbol[-4:]
    
    # 根据交易所格式化
    if EXCHANGE == "gate":
        return f"{base}_{quote}"
    elif EXCHANGE == "okx":
        return f"{base}-{quote}"
    
    return symbol


def fetch_klines_binance(symbol: str, interval: str, start_time: int, end_time: int, limit: int = 1000) -> List:
    """获取 Binance K线数据"""
    url = f"{BASE_URL}{CURRENT_CONFIG['klines_endpoint']}"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": limit,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_klines_gate(symbol: str, interval: str, start_time: int, end_time: int, limit: int = 1000) -> List:
    """
    获取 Gate.io K线数据并转换为 Binance 格式
    
    Gate.io 响应格式:
    [
        1680000000,       # 时间戳 (秒)
        "1234.56",        # 成交额
        "28000.5",        # 最高价
        "27500.0",        # 最低价
        "27800.0",        # 开盘价
        "123.45"          # 成交量 (基础资产)
    ]
    """
    url = f"{CURRENT_CONFIG['base_url']}{CURRENT_CONFIG['klines_endpoint']}"
    
    # Gate.io 使用秒级时间戳
    from_sec = start_time // 1000
    to_sec = end_time // 1000
    
    params = {
        "currency_pair": symbol,
        "interval": interval,
        "from": from_sec,
        "to": to_sec,
        "limit": limit,
    }
    
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    # 转换为 Binance 格式
    binance_format = []
    for item in data:
        timestamp_sec = int(item[0])
        timestamp_ms = timestamp_sec * 1000
        
        # Gate: [timestamp, volume_quote, high, low, open, volume_base]
        open_price = item[4]
        high_price = item[2]
        low_price = item[3]
        close_price = item[4]  # Gate 没有单独的收盘价，用开盘价近似
        volume = item[5]  # 基础资产成交量
        
        # 计算收盘时间（下一个K线开始时间 - 1ms）
        interval_ms = _get_interval_ms(interval)
        close_time = timestamp_ms + interval_ms - 1
        
        binance_format.append([
            timestamp_ms,       # 开盘时间
            str(open_price),    # 开盘价
            str(high_price),    # 最高价
            str(low_price),     # 最低价
            str(close_price),   # 收盘价（Gate 返回的是开盘价，这里需要调整）
            str(volume),        # 成交量
            close_time,         # 收盘时间
            item[1],            # 成交额
            0,                  # 成交笔数（Gate 不提供）
            "0",                # 主动买入成交量
            "0",                # 主动买入成交额
            "0"                 # 忽略
        ])
    
    return binance_format


def fetch_klines_okx(symbol: str, interval: str, start_time: int, end_time: int, limit: int = 1000) -> List:
    """
    获取 OKX K线数据并转换为 Binance 格式
    
    OKX 响应格式:
    {
        "code": "0",
        "data": [
            ["1680000000000", "27800.0", "28000.5", "27500.0", "27900.0", "123.45", "3456789.01"]
        ]
    }
    """
    url = f"{CURRENT_CONFIG['base_url']}{CURRENT_CONFIG['klines_endpoint']}"
    
    # OKX 使用毫秒时间戳
    params = {
        "instId": symbol,
        "bar": _convert_interval_okx(interval),
        "before": str(start_time),
        "after": str(end_time),
        "limit": str(limit),
    }
    
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    result = response.json()
    
    if result.get("code") != "0":
        raise Exception(f"OKX API error: {result.get('msg', 'Unknown error')}")
    
    data = result.get("data", [])
    
    # 转换为 Binance 格式
    binance_format = []
    for item in data:
        # OKX: [timestamp, open, high, low, close, vol, volCcy]
        timestamp_ms = int(item[0])
        open_price = item[1]
        high_price = item[2]
        low_price = item[3]
        close_price = item[4]
        volume = item[5]
        volume_currency = item[6]
        
        # 计算收盘时间
        interval_ms = _get_interval_ms(interval)
        close_time = timestamp_ms + interval_ms - 1
        
        binance_format.append([
            timestamp_ms,
            str(open_price),
            str(high_price),
            str(low_price),
            str(close_price),
            str(volume),
            close_time,
            str(volume_currency),
            0,
            "0",
            "0",
            "0"
        ])
    
    return binance_format


_INTERVAL_UNIT_MS = {
    "m": 60 * 1000,
    "h": 60 * 60 * 1000,
    "d": 24 * 60 * 60 * 1000,
    "w": 7 * 24 * 60 * 60 * 1000,
}


def interval_to_ms(interval: str) -> int:
    """将 K线周期（1m/5m/15m/30m/1h/4h/1d/1w）转换为毫秒。

    取数窗口必须用它来算，不能假设周期是小时 —— 否则 1m 会取到几天前的陈旧数据、
    4h/1d 会因为根数不够而无法预测（见 app/main.py:fetch_kline_data）。
    """
    unit = interval[-1]
    value = int(interval[:-1])
    if unit not in _INTERVAL_UNIT_MS:
        raise ValueError(f"不支持的 K 线周期: {interval}")
    return value * _INTERVAL_UNIT_MS[unit]


def kline_window_ms(interval: str, bars: int) -> int:
    """bars 根该周期 K 线覆盖的时间跨度（毫秒）"""
    return interval_to_ms(interval) * int(bars)


def drop_unclosed_klines(klines, interval_ms: int, now_ms: int = None):
    """丢弃尚未收盘的最后一根 K 线。

    为什么必须丢：训练只见过**完整** K 线（收盘后才入库），而取数接口会把当前正在走的
    那根一并返回 —— 它的 close/high/low/volume 都还在变。把这半根喂进模型，等于让
    推理输入的第 60 步与训练分布系统性不同（典型 train/serve skew）；
    模型的标签定义也是"窗口最后一根已收盘 → 下一根"，半根不满足这个前提。

    返回 (klines, dropped)：
      - 若最后一根的开盘时间 + 周期 > 当前时间，说明它还没收盘 -> 丢掉；
      - 时间戳缺失或无法解析时保守起见不丢（返回原样），避免误删数据。
    """
    if not klines:
        return klines, False
    import time as _time

    if now_ms is None:
        now_ms = int(_time.time() * 1000)
    try:
        last_open = float(klines[-1][0])
    except (TypeError, ValueError, IndexError):
        return klines, False
    if last_open + int(interval_ms) > now_ms:
        return klines[:-1], True
    return klines, False


def _get_interval_ms(interval: str) -> int:
    """兼容旧调用点，等价于 interval_to_ms"""
    return interval_to_ms(interval)


def _convert_interval_okx(interval: str) -> str:
    """将 Binance 周期格式转换为 OKX 格式"""
    mapping = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1H",
        "4h": "4H",
        "1d": "1D",
        "1w": "1W",
    }
    return mapping.get(interval, interval)


def fetch_klines(symbol: str, interval: str, start_time: int, end_time: int, limit: int = 1000) -> List:
    """
    统一 K线数据获取接口
    根据 EXCHANGE 配置自动选择交易所，并返回 Binance 格式数据
    """
    # 格式化交易对符号
    formatted_symbol = format_symbol(symbol)
    
    if EXCHANGE == "binance":
        return fetch_klines_binance(formatted_symbol, interval, start_time, end_time, limit)
    elif EXCHANGE == "gate":
        return fetch_klines_gate(formatted_symbol, interval, start_time, end_time, limit)
    elif EXCHANGE == "okx":
        return fetch_klines_okx(formatted_symbol, interval, start_time, end_time, limit)
    else:
        raise ValueError(f"Unsupported exchange: {EXCHANGE}")


def get_all_klines(symbol: str, interval: str, start_time: int, end_time: int) -> List:
    """
    获取所有 K线数据（自动分页）
    返回 Binance 格式数据
    """
    all_klines = []
    current_start = start_time
    limit = 1000
    
    while current_start < end_time:
        klines = fetch_klines(symbol, interval, current_start, end_time, limit)
        
        if not klines:
            break
        
        all_klines.extend(klines)
        print(f"  已获取 {len(all_klines)} 条数据...")
        
        if len(klines) < limit:
            break
        
        # 更新起始时间为最后一条K线的收盘时间 + 1
        current_start = klines[-1][6] + 1
        time.sleep(0.1)
    
    return all_klines
