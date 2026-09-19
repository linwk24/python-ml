"""
标签定义（把"预测什么"变成可插拔、可对比的一等参数）

固定阈值 ±0.1% 的问题：它把"波动"当成了绝对量。同一个 0.1% 对 BTC 可能是真实波动，
对 SOL 可能只是噪声 —— 于是同一个阈值在不同币种上定义了**难度不同的任务**
（实测：中性占比 BTC 33% → SOL 22%，多数类基线 33.7% → 39.8%）。

本模块提供三种标签定义，供跨币种公平比较：

A. ``fixed``      : ``|未来1根收益| > ±0.1%``（生产默认，作为对照基准）
B. ``volatility`` : ``threshold = k * sigma``，sigma 取**过去 window 根收益的标准差**；
                    也可以把尺度换成 ATR（relative）—— 同一种"按波动归一化"的思路
C. ``quantile``   : 用**过去 window 根收益的分位数**当阈值，天然类别均衡

**因果性要求（本模块所有函数必须满足）**：第 j 根的标签只允许用到第 j+1 根的收益，
而阈值只能用到第 j 根及更早的信息。任何"用未来分布定阈值"的写法都是标签泄漏，
本模块用 `tests/test_labeling.py` 的因果性测试钉住。

注意 ``quantile`` 的一种常见错误实现是"在整段数据上取未来收益的 33% 分位"——
那等于用未来信息定义标签。这里的实现是**滚动历史分位**，保持因果。
"""

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from config import LABEL_THRESHOLD

NEUTRAL, DOWN, UP = 1, 0, 2


def _simple_returns(closes: np.ndarray) -> np.ndarray:
    """ret[j] = (close[j] - close[j-1]) / close[j-1]，ret[0] = nan"""
    ret = np.full(len(closes), np.nan, dtype=float)
    if len(closes) > 1:
        ret[1:] = (closes[1:] - closes[:-1]) / closes[:-1]
    return ret


def _empty_labels(n: int) -> np.ndarray:
    """默认全部中性；最后一根没有"下一根"，保持中性（与生产实现的退化行为一致）"""
    return np.full(n, NEUTRAL, dtype=int)


def _classify(next_returns: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """把 (下一根收益, 阈值) 映射成标签数组；收益或阈值无效处为中性。

    返回长度与输入一致；调用方负责放到 out[:-1]（最后一根没有"下一根"）。
    """
    out = np.full(len(next_returns), NEUTRAL, dtype=int)
    valid = ~np.isnan(next_returns) & ~np.isnan(thr) & (thr > 0)
    nxt, t = next_returns[valid], thr[valid]
    out[valid] = np.where(nxt > t, UP, np.where(nxt < -t, DOWN, NEUTRAL))
    return out


def labels_fixed(df: pd.DataFrame, threshold: float = LABEL_THRESHOLD) -> np.ndarray:
    """A：固定百分比阈值（生产默认口径）"""
    closes = df["close"].to_numpy(dtype=float)
    n = len(closes)
    out = _empty_labels(n)
    if n < 2:
        return out
    ret = _simple_returns(closes)
    out[:-1] = _classify(ret[1:], np.full(n - 1, threshold))
    return out


def _atr_relative(df: pd.DataFrame, window: int) -> np.ndarray:
    """相对 ATR：真实波幅的 window 均值 / 收盘价（因果：只用到当前及更早）"""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    prev_close = np.concatenate([[np.nan], close[:-1]])
    tr = np.nanmax(np.vstack([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ]), axis=0)
    atr = pd.Series(tr).rolling(window=window).mean().to_numpy()
    return atr / np.where(close == 0, np.nan, close)


def labels_volatility(df: pd.DataFrame, k: float = 0.5, window: int = 60,
                      scale: str = "sigma") -> np.ndarray:
    """B：``threshold = k * 波动尺度``（按波动归一化，跨币种可比）

    scale="sigma" : 过去 window 根收益的标准差
    scale="atr"   : 过去 window 根相对 ATR（真实波幅均值 / 收盘价）
    """
    closes = df["close"].to_numpy(dtype=float)
    n = len(closes)
    out = _empty_labels(n)
    if n < 2:
        return out

    if scale == "sigma":
        ret = _simple_returns(closes)
        # rolling 含当前根：第 j 根的阈值只用 ret[..j]，因果
        vol = pd.Series(ret).rolling(window=window, min_periods=window).std().to_numpy()
    elif scale == "atr":
        vol = _atr_relative(df, window)
    else:
        raise ValueError(f"未知的波动尺度: {scale}")

    ret = _simple_returns(closes)
    out[:-1] = _classify(ret[1:], k * vol[:-1])
    return out


def labels_quantile(df: pd.DataFrame, window: int = 500,
                    lower_q: float = 1 / 3, upper_q: float = 2 / 3,
                    normalize_volatility: bool = False,
                    vol_window: int = 60) -> np.ndarray:
    """C：用**过去** window 根收益的分位数当阈值（保持因果）

    与"直接在整段未来收益上取分位"不同：阈值来自滚动历史分布，
    第 j 根只用到 ret[..j]，不存在未来信息。

    ``normalize_volatility=True`` 时改对**波动率标准化后的收益**取分位
    （``ret / sigma``，sigma 为过去 vol_window 根标准差）。原因：真实加密数据存在
    波动率聚集，直接对原始收益取滚动分位会得到一个过宽的判定带 ——
    实测 ETHUSDT 上类别分布是 跌19%/平63%/涨18%，"天然类别均衡"并不成立；
    先按波动率标准化再排序可以显著改善均衡性，同时保留"相对强弱"的含义。
    """
    closes = df["close"].to_numpy(dtype=float)
    n = len(closes)
    out = _empty_labels(n)
    if n < 2:
        return out

    ret = _simple_returns(closes)
    if normalize_volatility:
        sigma = pd.Series(ret).rolling(window=vol_window, min_periods=vol_window).std().to_numpy()
        ret = ret / np.where(sigma > 0, sigma, np.nan)

    series = pd.Series(ret)
    lo = series.rolling(window=window, min_periods=window).quantile(lower_q).to_numpy()
    hi = series.rolling(window=window, min_periods=window).quantile(upper_q).to_numpy()

    nxt = ret[1:]
    lo_t, hi_t = lo[:-1], hi[:-1]
    valid = ~np.isnan(nxt) & ~np.isnan(lo_t) & ~np.isnan(hi_t)
    labels = np.full(n - 1, NEUTRAL, dtype=int)
    v, a, b = nxt[valid], lo_t[valid], hi_t[valid]
    labels[valid] = np.where(v > b, UP, np.where(v < a, DOWN, NEUTRAL))
    out[:-1] = labels
    return out


# ------------------------------------------------------------------ 注册表

LABEL_MODES: Dict[str, Dict] = {
    "A_fixed_0.1pct": {
        "fn": lambda df: labels_fixed(df),
        "desc": "A 固定 ±0.1%（生产默认，对照基准）",
    },
    "B_vol_k0.5": {
        "fn": lambda df: labels_volatility(df, k=0.5),
        "desc": "B σ 归一化 k=0.5",
    },
    "B_vol_k0.8": {
        "fn": lambda df: labels_volatility(df, k=0.8),
        "desc": "B σ 归一化 k=0.8",
    },
    "B_vol_k1.0": {
        "fn": lambda df: labels_volatility(df, k=1.0),
        "desc": "B σ 归一化 k=1.0",
    },
    "B_atr_k0.5": {
        "fn": lambda df: labels_volatility(df, k=0.5, scale="atr"),
        "desc": "B 相对 ATR 归一化 k=0.5",
    },
    "C_quantile": {
        "fn": lambda df: labels_quantile(df),
        "desc": "C 滚动历史分位（原始收益）",
    },
    "C_quantile_vol": {
        "fn": lambda df: labels_quantile(df, normalize_volatility=True),
        "desc": "C 滚动历史分位（先按波动率标准化）",
    },
}


def get_label_fn(mode: str) -> Callable[[pd.DataFrame], np.ndarray]:
    if mode not in LABEL_MODES:
        raise ValueError(f"未知的标签模式: {mode}（可选: {list(LABEL_MODES)}）")
    return LABEL_MODES[mode]["fn"]


def label_distribution(labels: np.ndarray) -> Dict[str, float]:
    """标签分布（用于确认"类别是否均衡"）"""
    labels = np.asarray(labels)
    n = len(labels)
    if n == 0:
        return {"down": 0.0, "neutral": 0.0, "up": 0.0}
    return {
        "down": float((labels == DOWN).mean()),
        "neutral": float((labels == NEUTRAL).mean()),
        "up": float((labels == UP).mean()),
    }


def threshold_series(df: pd.DataFrame, mode: str, **kwargs) -> Optional[np.ndarray]:
    """返回该标签模式下每根的有效阈值（便于诊断阈值随波动如何变化）"""
    if mode.startswith("B_vol"):
        cfg = LABEL_MODES[mode]["fn"]
        closes = df["close"].to_numpy(dtype=float)
        ret = pd.Series(_simple_returns(closes))
        k = float(mode.split("k")[-1])
        if "atr" in mode:
            return k * _atr_relative(df, 60)
        return k * ret.rolling(60, min_periods=60).std().to_numpy()
    if mode.startswith("C_"):
        closes = df["close"].to_numpy(dtype=float)
        ret = pd.Series(_simple_returns(closes))
        return ret.rolling(500, min_periods=500).quantile(2 / 3).to_numpy()
    return None
