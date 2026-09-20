"""
信号反转（前置动量校验）

作用：把「模型给出的方向」与「最近 ``SIGNAL_REVERSAL_LOOKBACK_BARS`` 根 K 线的实际涨跌」
对照。当二者矛盾且价差超过阈值时，给出反转建议。

**基准价必须来自本次预测所用的 K 线。**

历史实现（app/main.py 内联版本）从 ``predictions_{symbol}.json`` 取"倒数第二条记录"的价格
当基准，而该记录只在方向翻转时才落盘 —— 基准价可能陈旧数月。线上实测复现过::

    last_price 64266.69   （2026-05-23 落盘的记录）
    current_price 80624   （当前价）
    → price_change +25.45% → "应反转为看涨"

拿三个月的涨幅当作"当前动量"是完全无效的信号。本模块改为纯函数：输入 K 线、
输出结论，不读写任何状态文件，也因此可以在没有 TensorFlow 的环境里被单测覆盖。

语义（务必明确，避免"反转了但没人用"）:
    - 返回的 ``model_trend_code`` 是模型原始输出；
    - ``effective_trend_code`` 是"最终应当采用"的方向：仅当 ``SIGNAL_REVERSAL_APPLY``
      为 True 且触发反转时才不同于模型输出，否则恒等于模型输出；
    - 调用方（API 层）应始终返回 ``effective_trend_code`` 作为权威信号，
      这样下游不必猜"到底该信 prediction 还是 signal_reversal"。
"""

from typing import Any, Dict, List, Optional, Sequence

from config import (
    SIGNAL_REVERSAL_APPLY,
    SIGNAL_REVERSAL_LOOKBACK_BARS,
    SIGNAL_REVERSAL_THRESHOLD,
)

TREND_LABELS = {0: "看跌", 1: "中性", 2: "看涨"}


def format_price(price: float) -> str:
    """按价格量级自适应小数位 —— 固定 2 位会把 DOGE(0.075) 显示成 0.07，等于没显示"""
    if price is None:
        return "n/a"
    p = float(price)
    if abs(p) >= 100:
        return f"{p:,.2f}"
    if abs(p) >= 1:
        return f"{p:.4f}"
    return f"{p:.6f}"


def _price_span(baseline_price: float, current_price: float) -> str:
    """基准 -> 现价，便于直接肉眼核对"""
    return f"基准 {format_price(baseline_price)} → 现价 {format_price(current_price)}"


def _bar_open_ms(row: Sequence[Any]) -> Optional[float]:
    try:
        return float(row[0])
    except (TypeError, ValueError, IndexError):
        return None


def _bar_close(row: Sequence[Any]) -> Optional[float]:
    try:
        return float(row[4])
    except (TypeError, ValueError, IndexError):
        return None


def effective_trend(
    model_trend_code: int,
    reversal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """由模型输出与反转结论推导出权威信号（API 层应当返回它）。"""
    applied = bool(reversal and reversal.get("applied"))
    if applied:
        return {
            "trend": TREND_LABELS.get(reversal["effective_trend_code"], "未知"),
            "trend_code": reversal["effective_trend_code"],
            "source": "signal_reversal",
        }
    return {
        "trend": TREND_LABELS.get(model_trend_code, "未知"),
        "trend_code": model_trend_code,
        "source": "model",
    }


def check_signal_reversal(
    klines: Optional[List[Sequence[Any]]],
    current_trend: int,
    lookback_bars: Optional[int] = None,
    threshold: Optional[float] = None,
    apply_reversal: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """基于 K 线的前置动量校验。

    参数:
        klines: 与预测同源的 K 线，形如 [open_time, open, high, low, close, volume, ...]
        current_trend: 模型输出的方向 0/1/2
        lookback_bars: 基准价回溯根数，默认取 config.SIGNAL_REVERSAL_LOOKBACK_BARS
        threshold: 价差阈值，默认取 config.SIGNAL_REVERSAL_THRESHOLD
        apply_reversal: 是否让反转覆盖模型输出，默认取 config.SIGNAL_REVERSAL_APPLY

    返回 None 表示数据不足、无法判断（调用方应据此跳过该字段）。
    """
    lookback = int(lookback_bars if lookback_bars is not None else SIGNAL_REVERSAL_LOOKBACK_BARS)
    threshold = float(threshold if threshold is not None else SIGNAL_REVERSAL_THRESHOLD)
    apply_flag = bool(apply_reversal if apply_reversal is not None else SIGNAL_REVERSAL_APPLY)

    if not klines or lookback < 1 or len(klines) <= lookback:
        return None

    # 以开盘时间排序，避免调用方传入逆序数据时取错基准
    ordered = sorted(
        (r for r in klines if _bar_open_ms(r) is not None and _bar_close(r) is not None),
        key=_bar_open_ms,
    )
    if len(ordered) <= lookback:
        return None

    current_bar = ordered[-1]
    baseline_bar = ordered[-1 - lookback]
    current_price = _bar_close(current_bar)
    baseline_price = _bar_close(baseline_bar)
    if not baseline_price or not current_price:
        return None

    price_change = (current_price - baseline_price) / baseline_price

    result: Dict[str, Any] = {
        "enabled": True,
        "triggered": False,
        "applied": False,
        "lookback_bars": lookback,
        "baseline_price": baseline_price,
        "baseline_bar_ms": _bar_open_ms(baseline_bar),
        "current_price": current_price,
        "price_change": price_change,
        "price_change_pct": f"{price_change:+.2%}",
        "threshold": threshold,
        "threshold_pct": f"{threshold:.2%}",
        "model_trend_code": current_trend,
        "effective_trend_code": current_trend,
        "corrected_trend_code": None,
        "message": "",
    }

    if abs(price_change) <= threshold:
        result["message"] = (
            f"近 {lookback} 根 K 线价差 {price_change:+.2%} 未超过阈值 {threshold:.2%}，维持模型方向"
            f"（{_price_span(baseline_price, current_price)}）"
        )
        return result

    corrected: Optional[int] = None
    if price_change > 0 and current_trend == 0:
        corrected = 2  # 价格在涨，模型看跌 -> 反转为看涨
        result["reason"] = "reverse_to_bullish"
    elif price_change < 0 and current_trend == 2:
        corrected = 0  # 价格在跌，模型看涨 -> 反转为看跌
        result["reason"] = "reverse_to_bearish"

    if corrected is None:
        # 中性没有方向，说“与模型方向一致”会误导（价格明明动了）
        if current_trend == 1:
            result["message"] = (
                f"近 {lookback} 根 K 线价差 {price_change:+.2%} 已超过阈值 {threshold:.2%}，"
                f"但模型方向为中性，无方向可反转"
                f"（{_price_span(baseline_price, current_price)}）"
            )
        else:
            result["message"] = (
                f"近 {lookback} 根 K 线价差 {price_change:+.2%} 与模型方向"
                f"（{TREND_LABELS.get(current_trend, current_trend)}）一致，无需反转"
                f"（{_price_span(baseline_price, current_price)}）"
            )
        return result

    result["triggered"] = True
    result["corrected_trend_code"] = corrected
    result["applied"] = apply_flag
    if apply_flag:
        result["effective_trend_code"] = corrected
        result["message"] = (
            f"近 {lookback} 根 K 线价差 {price_change:+.2%} 与模型方向"
            f"（{TREND_LABELS.get(current_trend, current_trend)}）矛盾，"
            f"已反转为{TREND_LABELS[corrected]}"
            f"（{_price_span(baseline_price, current_price)}）"
        )
    else:
        result["message"] = (
            f"近 {lookback} 根 K 线价差 {price_change:+.2%} 与模型方向"
            f"（{TREND_LABELS.get(current_trend, current_trend)}）矛盾，"
            f"建议反转为{TREND_LABELS[corrected]}（当前配置为仅建议，不覆盖模型输出；"
            f"{_price_span(baseline_price, current_price)}）"
        )
    return result
