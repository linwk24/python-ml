"""
回测口径工具（不含 TensorFlow 依赖，便于单测）

这个模块存在的唯一理由：**评价基准必须对齐模型真正被训练去预测的那一根 K 线**。

同一类"拿错基准"的 bug 在本项目里出现过三次：
  1. `check_signal_reversal` 拿几个月前的预测记录价当基准；
  2. `PredictionTracker` 拿"当前价"去核对几天前的预测；
  3. `backtest_signals` 拿输入窗口最后一根的价格去比 `df.iloc[i+1]`（跨度 2 根），
     导致相邻样本目标重叠一根 —— 实测滞后 1 期自相关被抬到 +0.51（真实值应接近 0）。
"""

from typing import Any, Optional, Tuple

from config import LABEL_THRESHOLD


def next_bar_label(
    df: Any,
    i: int,
    threshold: float = LABEL_THRESHOLD,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """计算"模型在 idx=i 这一步预测的那一根"的涨跌与标签。

    口径（与 models/enhanced_lstm.prepare_data 的训练标签严格一致）：
        输入窗口 = df[i-lookback : i]，最后一根是 ``i-1``；
        训练标签取的是 **行 i-1 -> 行 i** 这一根的涨跌，即窗口最后一根的"下一根"。

    这同时也是线上闭环的口径：记录价 = 窗口最后一根收盘价，核对价 = 它的下一根。
    训练 / 推理 / 评价三处必须一致，否则"准确率"没有意义。

    返回 ``(target_base_price, change_pct, actual_trend)``。
    """
    # 注：本模块不导入 pandas/numpy，纯函数部分可在零依赖环境使用
    if i <= 0 or i >= len(df):
        raise IndexError(f"下标越界: i={i}, 长度={len(df)}（要求 1 <= i <= {len(df)-1}）")

    base_price = float(df.iloc[i - 1]["close"])
    next_close = float(df.iloc[i]["close"])
    change_pct = (next_close - base_price) / base_price * 100

    if change_pct > threshold * 100:
        actual_trend = "涨"
    elif change_pct < -threshold * 100:
        actual_trend = "跌"
    else:
        actual_trend = "平"

    return base_price, change_pct, actual_trend


def is_prediction_correct(predicted_trend: str, actual_trend: Optional[str]) -> Optional[bool]:
    """方向标签对错判定（三分类，含中性）"""
    if actual_trend is None:
        return None
    expected = {"看涨": "涨", "看跌": "跌", "中性": "平"}.get(predicted_trend)
    if expected is None:
        return None
    return expected == actual_trend


def flat_band_share(changes_pct) -> float:
    """落在中性带内（±threshold）的比例 —— 即"永远说中性"的准确率基线"""
    import numpy as np

    arr = np.asarray([c for c in changes_pct if c is not None], dtype=float)
    if arr.size == 0:
        return 0.0
    lo, hi = -LABEL_THRESHOLD * 100, LABEL_THRESHOLD * 100
    return float(((arr >= lo) & (arr <= hi)).mean())
