"""
实时价通路 —— 供**仓位/风控层**使用，绝不是模型输入

为什么必须单独一条通路（实测依据，见 experiments/partial_bar_probe.py）：

1. 模型是在**完整的 60 分钟 K 线**上训练的。把"正在走"的半根 K 线喂进去，
   它的 high_low_ratio 会被系统性压缩 —— 同一批 60 小时里，走到第 30 分钟的半根
   振幅中位数只有完整 bar 的 67.5%，第 15 分钟只有 52.7%。这是**有方向的偏差**
   （让模型系统性低估波动），不是随机噪声；而且 high_low_ratio 恰好在极端行情
   OOD 排序里列第三。

2. 实测方向被改变的比例 13.3%（83 个样本，95% 区间 [7.6%, 22.2%]），含完整的反向。
   根因是模型概率极平（约 35/31/34，仅略高于均匀分布 33.3），argmax 本就由噪声主导。

3. 反向的证据：仓位层真正需要的是**实时价格**这个标量（算未实现盈亏、判止损），
   而不是"实时特征"。价格没有训练分布问题，可靠得多。

所以本模块只做一件事：给仓位层一个权威的实时价 + 正在走的那根的进度。
调用方**不得**把它拼进 klines 再喂给模型（fetch_kline_data 的 drop_unclosed=True 是
模型侧的硬约束，两者不可混用）。
"""

import json
import logging
import time
import urllib.request
from typing import Any, Dict, Optional

import config

logger = logging.getLogger("live_price")

# 与 config.EXCHANGE 一致
_CFG = config.EXCHANGE_CONFIG[config.EXCHANGE]
_BASE = _CFG["base_url"]
_KLINES = _CFG["klines_endpoint"]

# 实时价的语义锚点：它来自正在走的那根 K 线的最新成交价
_LIVE_SOURCE = f"{config.EXCHANGE}_forming_kline"


def _get_json(url: str, timeout: float) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "python-ml/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_live(
    symbol: str,
    interval: str = "1h",
    timeout: float = 5.0,
) -> Dict[str, Any]:
    """取实时价 + 正在走的那根 K 线的进度。

    只发**一个** HTTP 请求（klines limit=2，最后一次就是未收盘那根），
    避免给 /predict 增加额外往返。

    任何失败都不抛异常 —— 返回结构里带 ``error``，让 /predict 照常返回预测结果。
    风控层拿不到实时价时必须能明确区分"价格是 0"和"没取到"。
    """
    out: Dict[str, Any] = {
        "price": None,
        "as_of_ms": int(time.time() * 1000),
        "source": _LIVE_SOURCE,
        # 显式声明，防止被误当作模型输入拼回 klines
        "for_model": False,
        "note": "实时价，仅供仓位/风控层使用；模型输入始终只用已收盘 K 线",
        "forming_bar": None,
        "error": None,
    }

    url = f"{_BASE}{_KLINES}?symbol={symbol}&interval={interval}&limit=2"
    try:
        data = _get_json(url, timeout)
    except Exception as e:
        out["error"] = f"实时价获取失败: {type(e).__name__}: {e}"
        logger.warning(f"[{symbol}] {out['error']}")
        return out

    if not isinstance(data, list) or not data:
        out["error"] = f"实时价响应格式异常: {type(data).__name__}"
        logger.warning(f"[{symbol}] {out['error']}")
        return out

    last = data[-1]
    try:
        open_ms = int(float(last[0]))
        o, h, l, c, v = (float(last[1]), float(last[2]),
                         float(last[3]), float(last[4]), float(last[5]))
    except (IndexError, TypeError, ValueError) as e:
        out["error"] = f"实时价字段解析失败: {e}"
        logger.warning(f"[{symbol}] {out['error']}")
        return out

    interval_ms = config.interval_to_ms(interval)
    now_ms = int(time.time() * 1000)
    elapsed_ms = max(0, now_ms - open_ms)
    complete = elapsed_ms >= interval_ms

    out["price"] = c
    out["forming_bar"] = {
        "open_ms": open_ms,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
        "elapsed_minutes": round(elapsed_ms / 60000.0, 2),
        "interval_ms": interval_ms,
        # 若恰好落在边界上（上一次请求正好卡在收盘瞬间），这里会是 True，
        # 调用方应据此判断"这根其实已经收盘了"。
        "complete": complete,
        "progress": min(1.0, elapsed_ms / interval_ms) if interval_ms else 0.0,
    }
    return out


def position_guard(price: Optional[float]) -> bool:
    """仓位层的最小校验：拿不到实时价就不允许做仓位决策。

    静默用 0 或 None 当价格，会让止损逻辑算出错误的未实现盈亏 —— 宁可不动。
    """
    return price is not None and price > 0


def live_from_forming_bar(
    forming_bar: Optional[list],
    interval: str = "1h",
    now_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """从**已经取到的**那根未收盘 K 线构造 live 块 —— 零额外 HTTP 往返。

    这是首选路径：``fetch_kline_data`` 本来就已经把未收盘那根取回来了，
    只是在返回前丢掉（见 config.drop_unclosed_klines）。把它顺手带出来即可，
    不必为此再发一次请求（实测单独请求约 420ms）。

    ``forming_bar`` 为 None 表示这根不存在（恰好卡在收盘瞬间）——此时
    ``price`` 为 None，风控层应据 position_guard 决定不动。
    """
    now = int(now_ms if now_ms is not None else time.time() * 1000)
    out: Dict[str, Any] = {
        "price": None,
        "as_of_ms": now,
        "source": f"{config.EXCHANGE}_forming_kline",
        "for_model": False,
        "note": "实时价，仅供仓位/风控层使用；模型输入始终只用已收盘 K 线",
        "forming_bar": None,
        "error": None,
        "enabled": True,
    }

    if not forming_bar:
        out["error"] = "本次未取到未收盘的 K 线（可能恰好卡在收盘边界）"
        return out

    try:
        open_ms = int(float(forming_bar[0]))
        o, h, l, c, v = (float(forming_bar[1]), float(forming_bar[2]),
                         float(forming_bar[3]), float(forming_bar[4]),
                         float(forming_bar[5]))
    except (IndexError, TypeError, ValueError) as e:
        out["error"] = f"未收盘 K 线字段解析失败: {e}"
        logger.warning(out["error"])
        return out

    interval_ms = config.interval_to_ms(interval)
    elapsed_ms = max(0, now - open_ms)
    out["price"] = c
    out["forming_bar"] = {
        "open_ms": open_ms,
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "elapsed_minutes": round(elapsed_ms / 60000.0, 2),
        "interval_ms": interval_ms,
        "complete": elapsed_ms >= interval_ms,
        "progress": min(1.0, elapsed_ms / interval_ms) if interval_ms else 0.0,
    }
    return out


def build_live_block(
    symbol: str,
    interval: str = "1h",
    forming_bar: Optional[list] = None,
) -> Dict[str, Any]:
    """组装响应里的 ``live`` 块 —— 给**仓位/风控层**用的第二条通路。

    优先用调用方**已经取到**的 forming_bar（零延迟）；没给才自己发一次请求。
    刻意放在本模块（而不是 API 层）：它是纯逻辑，不依赖 FastAPI，
    这样"实时价失败不拖垮预测"这条不变量可以在没有 Web 依赖的环境里被测试。
    """
    if not config.LIVE_PRICE_ENABLED:
        return {"live": {"price": None, "for_model": False, "enabled": False,
                         "error": "LIVE_PRICE_ENABLED=false"}}
    if forming_bar is not None:
        return {"live": live_from_forming_bar(forming_bar, interval)}
    block = fetch_live(symbol, interval, timeout=config.LIVE_PRICE_TIMEOUT)
    block["enabled"] = True
    return {"live": block}
