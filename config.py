"""
交易所配置和数据获取模块
统一不同交易所的 API 接口，输出 Binance K线格式
"""

import requests
import time
from typing import List, Optional

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

# ============== 信号反转配置 ==============
# 前置信号反转模式：当预测方向切换时，检查价差是否超过阈值
# 如果超过阈值，则认为上次预测错误，触发信号反转
SIGNAL_REVERSAL_ENABLED = True  # 是否启用信号反转模式
SIGNAL_REVERSAL_THRESHOLD = 0.001  # 价差阈值 0.1% (0.001 = 0.1%)

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


def _get_interval_ms(interval: str) -> int:
    """将 K线周期转换为毫秒"""
    unit = interval[-1]
    value = int(interval[:-1])
    
    multipliers = {
        "m": 60 * 1000,
        "h": 60 * 60 * 1000,
        "d": 24 * 60 * 60 * 1000,
        "w": 7 * 24 * 60 * 60 * 1000,
    }
    
    return value * multipliers.get(unit, 60 * 1000)


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
