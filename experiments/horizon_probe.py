"""
多视界探针：1 根视界到底是"太短"，还是这个市场在任何视界上都不可预测？

动机
----
生产口径写死 ``PREDICTION_HORIZON_BARS = 1``：标签是"下一根 1h K 线的涨跌（±0.1%）"。
而手续费是**按笔**收的，边际是**按笔**赚的 —— 视界越短，成本/边际之比越差。
实测 1 根视界下每笔毛边际只有 3~6bp，而成本 5~10bp/边。

所以关键问题是：把视界拉长到 h 根，特征里**是否含有更多可提取的信息**？
本脚本用同一套 17 维特征 + 线性模型，在 h ∈ {1,2,4,8,24} 上各训一次，
报告三分类 balanced accuracy 与二分类方向命中率，并与各自的多数类基线对比。

注意：这是**特征信息量的探针**，不是 LSTM。用线性模型是为了快，
且能干净地回答"信息在不在特征里"，不受 LSTM 优化噪声干扰。

用法：.venv/bin/python experiments/horizon_probe.py [--symbols BTCUSDT,ETHUSDT,SOLUSDT] [--bars 8000]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.main import fetch_kline_data  # noqa: E402
from models.enhanced_lstm import EnhancedLSTMPredictor  # noqa: E402

# 生产使用的 17 维特征（与线上 feature_weights.feature_importance 一致）
FEATURES = [
    "rsi", "macd_histogram", "bb_width", "bb_position", "ma_ratio_5_20",
    "ma_ratio_10_60", "adx", "cci", "williams_r", "stoch_k", "stoch_d",
    "mfi", "ichimoku_tenkan_sen", "ichimoku_kijun_sen", "price_change_pct",
    "high_low_ratio", "close_open_ratio",
]
NEUTRAL, DOWN, UP = 1, 0, 2
WARMUP = 120          # 与生产一致：特征预热
HORIZONS = [1, 2, 4, 8, 24]


def build(symbol, n_bars):
    """返回 (特征矩阵, 收盘价数组)。特征第 i 行只用第 i 根及之前的信息。"""
    bars = fetch_kline_data(symbol, "1h", n_bars)
    df = pd.DataFrame(bars, columns=["open_time", "open", "high", "low",
                                     "close", "volume"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    p = EnhancedLSTMPredictor(symbol=symbol)      # 只为复用特征管线，不加载模型
    feats = p._compute_features(df)
    X = feats[FEATURES].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    return X, close


def labels_for_horizon(close, h, threshold=None):
    """标签：第 i 行的"下一根"= 未来 h 根的累计收益。

    与生产口径对齐：输入窗口最后一根是 i，标签用的是 i 之后的 h 根，
    即 (close[i+h] - close[i]) / close[i]。h=1 时退化成生产的 next_bar_label。

    阈值默认按该视界下收益的标准差归一化（threshold = 0.25*sigma），
    这样不同 h 的类别占比可比 —— 否则固定 0.1% 在 h=24 时几乎没有"中性"。
    """
    n = len(close)
    fwd = np.full(n, np.nan)
    fwd[:n - h] = (close[h:] - close[:n - h]) / close[:n - h]
    valid = ~np.isnan(fwd)
    if threshold is None:
        sigma = np.nanstd(fwd[valid])
        threshold = 0.25 * sigma
    lab = np.full(n, NEUTRAL, dtype=int)
    lab[valid & (fwd > threshold)] = UP
    lab[valid & (fwd < -threshold)] = DOWN
    return lab, fwd, threshold


def evaluate(X, close, h, split=0.7):
    lab, fwd, thr = labels_for_horizon(close, h)
    n = len(X)
    # 特征第 i 行对应"窗口截至第 i 根" -> 标签用第 i 行之后的 h 根
    ok = (np.arange(n) >= WARMUP) & ~np.isnan(fwd) & ~np.isnan(X).any(axis=1)
    Xv, yv, fv = X[ok], lab[ok], fwd[ok]
    cut = int(len(Xv) * split)
    Xtr, ytr, Xte, yte, fte = Xv[:cut], yv[:cut], Xv[cut:], yv[cut:], fv[cut:]

    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    bal = balanced_accuracy_score(yte, pred)
    majority = np.bincount(yte, minlength=3).max() / len(yte)
    # 多数类 balanced accuracy 基线恒为 1/3
    share = {k: float((yv == v).mean()) for k, v in
             (("DOWN", DOWN), ("NEUTRAL", NEUTRAL), ("UP", UP))}

    # 方向命中率：只在"模型给了方向 **且** 真实也有方向"的样本上算
    # （把"模型报中性"算成错误是错的 —— 那是弃权，不是判断）
    m = (yte != NEUTRAL) & (pred != NEUTRAL)
    dir_acc = float((pred[m] == yte[m]).mean()) if m.sum() else float("nan")
    coverage = float(m.sum() / len(yte))
    taken = m.sum()
    # 取仓位时的实际幅度
    move = float(np.median(np.abs(fte[m]))) if taken else float("nan")
    # 每笔毛边际 ≈ (2p-1) × 平均幅度；与"按笔收"的手续费直接可比
    edge_bp = (2 * dir_acc - 1) * move * 1e4 if taken else float("nan")

    return {
        "h": h, "thr": thr, "n_train": len(Xtr), "n_test": len(Xte),
        "balanced_acc": bal, "majority_acc": majority,
        "dir_acc": dir_acc, "coverage": coverage, "taken": int(taken),
        "share": share, "abs_move_median": move, "edge_bp": edge_bp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    ap.add_argument("--bars", type=int, default=8000)
    args = ap.parse_args()

    print("口径：17 维特征 + 多项式 LogisticRegression；按时间 70/30 切分，只用训练段拟合标准化；")
    print("      阈值 = 0.25 * 该视界下未来收益的 σ（让不同 h 的类别占比可比）")
    print("      balanced accuracy 的多数类基线恒为 33.3%\n")

    for s in args.symbols.split(","):
        try:
            X, close = build(s, args.bars)
        except Exception as e:
            print(f"[{s}] 跳过: {e}")
            continue
        print(f"{'='*104}")
        print(f"{s}   {len(X)} 根 1h K 线")
        print(f"{'='*104}")
        print(f"{'视界h':>6}{'balanced acc':>13}{'多数类基线':>11}{'方向命中':>10}{'取仓占比':>10}"
              f"{'|幅度|中位':>11}{'每笔毛边际':>12}{'   vs 10bp成本'}")
        for h in HORIZONS:
            r = evaluate(X, close, h)
            verdict = "够付" if r["edge_bp"] > 10 else "不够"
            print(f"{r['h']:>6}{r['balanced_acc']:>13.4f}{r['majority_acc']:>11.4f}"
                  f"{r['dir_acc']:>10.4f}{r['coverage']:>10.1%}{r['abs_move_median']:>11.3%}"
                  f"{r['edge_bp']:>11.2f}bp   {verdict}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
