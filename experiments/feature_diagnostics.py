"""
特征集诊断：冗余 / 信息质量 / 尺度漂移

回答三个问题（都是用数据答，不靠讨论）：
  1. 冗余：这 17 个特征之间相关性有多高？哪些是同一个东西的不同写法？
  2. 信息质量：每个特征与标签的互信息有多少？（含"随机特征"做对照基线）
  3. 尺度漂移：`ichimoku_*` 这类**绝对价格**特征，会不会随着价格中枢移动而漂出训练分布？

第 3 点尤其关键：绝对价格特征经标准化后变成"(价格 - 训练段均值)/训练段标准差"，
它是一个**非平稳**的量 —— 价格一旦走出训练区间，z 分数就会持续偏大，
表现为"模型看到没见过的东西"（OOD），而不是"转换线高于基准线多少"这种相对关系。

用法：.venv/bin/python experiments/feature_diagnostics.py [--symbol BTCUSDT] [--bars 12000]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.main import fetch_kline_data  # noqa: E402
from models.enhanced_lstm import EnhancedLSTMPredictor  # noqa: E402

FEATURES = [
    "rsi", "macd_histogram", "bb_width", "bb_position", "ma_ratio_5_20",
    "ma_ratio_10_60", "adx", "cci", "williams_r", "stoch_k", "stoch_d",
    "mfi", "ichimoku_tenkan_sen", "ichimoku_kijun_sen", "price_change_pct",
    "high_low_ratio", "close_open_ratio",
]
WARMUP = 120
# 已知高度相关的组（用户提出的假设，本脚本负责验证）
MOMENTUM_GROUP = ["rsi", "williams_r", "stoch_k", "stoch_d", "cci"]


def load(symbol, n_bars):
    bars = fetch_kline_data(symbol, "1h", n_bars)
    df = pd.DataFrame(bars, columns=["open_time", "open", "high", "low",
                                     "close", "volume"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    p = EnhancedLSTMPredictor(symbol=symbol)
    feats = p._compute_features(df)
    return df, feats


def labels(df, horizon=1, thr=0.001):
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    fwd = np.full(n, np.nan)
    fwd[:n - horizon] = (close[horizon:] - close[:n - horizon]) / close[:n - horizon]
    lab = np.full(n, 1, dtype=int)
    lab[(fwd > thr)] = 2
    lab[(fwd < -thr)] = 0
    return lab, fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--bars", type=int, default=12000)
    args = ap.parse_args()

    df, feats = load(args.symbol, args.bars)
    lab, fwd = labels(df)
    ok = (np.arange(len(feats)) >= WARMUP) & ~np.isnan(fwd)
    X = feats.loc[ok, FEATURES].to_numpy(dtype=float)
    y = lab[ok]
    print(f"[{args.symbol}] {len(X)} 个有效样本，特征 {len(FEATURES)} 个\n")

    # ───────── 1. 冗余：相关矩阵 ─────────
    C = np.corrcoef(X, rowvar=False)
    print("=" * 100)
    print("1. 冗余：|相关系数| 最高的 12 对")
    print("=" * 100)
    pairs = []
    for i in range(len(FEATURES)):
        for j in range(i + 1, len(FEATURES)):
            pairs.append((abs(C[i, j]), FEATURES[i], FEATURES[j], C[i, j]))
    pairs.sort(reverse=True)
    print(f"{'|r|':>7}   {'特征 A':<24}{'特征 B':<24}")
    for a, f1, f2, r in pairs[:12]:
        flag = "  <== 近乎重复" if a > 0.9 else ("  <== 高度相关" if a > 0.8 else "")
        print(f"{a:>7.3f}   {f1:<24}{f2:<24}{flag}")
    print(f"\n  |r| > 0.9 的对数: {sum(1 for a,_,_,_ in pairs if a > 0.9)}")
    print(f"  |r| > 0.8 的对数: {sum(1 for a,_,_,_ in pairs if a > 0.8)}")
    print(f"  |r| > 0.7 的对数: {sum(1 for a,_,_,_ in pairs if a > 0.7)}")

    # 用户假设的动量组内部相关
    idx = [FEATURES.index(f) for f in MOMENTUM_GROUP]
    sub = C[np.ix_(idx, idx)]
    off = sub[np.triu_indices(len(idx), 1)]
    print(f"\n  用户假设的动量组 {MOMENTUM_GROUP}")
    print(f"    组内 |r|: 均值 {np.abs(off).mean():.3f}, 中位 {np.median(np.abs(off)):.3f}, "
          f"最大 {np.abs(off).max():.3f}, 最小 {np.abs(off).min():.3f}")

    # ───────── 2. 信息质量：与标签的互信息 ─────────
    print(f"\n{'='*100}")
    print("2. 信息质量：与标签(h=1, ±0.1%)的互信息（含随机特征作对照）")
    print("=" * 100)
    rng = np.random.default_rng(0)
    Xa = np.column_stack([X, rng.normal(size=len(X)), rng.normal(size=len(X))])
    names = FEATURES + ["[随机噪声1]", "[随机噪声2]"]
    mi = mutual_info_classif(Xa, y, random_state=0, n_neighbors=5)
    order = np.argsort(-mi)
    noise_max = max(mi[FEATURES.__len__():])
    print(f"{'互信息':>10}  {'特征':<26}{'vs 噪声基线':>14}")
    for i in order:
        rel = "  <== 低于噪声" if mi[i] <= noise_max and names[i] not in ("[随机噪声1]", "[随机噪声2]") else ""
        print(f"{mi[i]:>10.5f}  {names[i]:<26}{mi[i]/noise_max:>13.2f}x{rel}")
    print(f"\n  随机噪声的互信息上限: {noise_max:.5f}（这是本次采样的噪声地板）")

    # ───────── 3. 尺度漂移：绝对价格特征的 z 分数 ─────────
    print(f"\n{'='*100}")
    print("3. 尺度漂移：用前 70% 拟合标准化，看后 30% 的 max|z|")
    print("=" * 100)
    cut = int(len(X) * 0.7)
    mu, sd = X[:cut].mean(0), X[:cut].std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    late = Z[cut:]
    rows = []
    for k, f in enumerate(FEATURES):
        rows.append((np.abs(late[:, k]).mean(), np.abs(late[:, k]).max(), f,
                     Z[:cut][:, k].std(), mu[k], sd[k]))
    rows.sort(reverse=True)
    print(f"{'后30% 平均|z|':>14}{'后30% 最大|z|':>14}  {'特征':<26}")
    for m, mx, f, _, _, _ in rows[:8]:
        print(f"{m:>14.2f}{mx:>14.2f}  {f:<26}")
    print(f"{'...':>28}")
    for m, mx, f, _, _, _ in rows[-3:]:
        print(f"{m:>14.2f}{mx:>14.2f}  {f:<26}")

    # 绝对价格特征 vs 相对特征的漂移对比
    absf = ["ichimoku_tenkan_sen", "ichimoku_kijun_sen"]
    iabs = [FEATURES.index(f) for f in absf]
    others = [i for i in range(len(FEATURES)) if i not in iabs]
    print(f"\n  绝对价格特征 {absf}:")
    for i in iabs:
        print(f"    {FEATURES[i]:<24} 后30% 平均|z| = {np.abs(late[:, i]).mean():.2f}  "
              f"最大 {np.abs(late[:, i]).max():.2f}   原始尺度 mean={mu[i]:,.1f} sd={sd[i]:,.1f}")
    print(f"  其余 15 个特征          后30% 平均|z| 中位 = "
          f"{np.median([np.abs(late[:, i]).mean() for i in others]):.2f}")

    # 相对化改造后对比
    close = df["close"].to_numpy(dtype=float)[ok]
    for name, num in (("tenkan", feats["ichimoku_tenkan_sen"].to_numpy()[ok]),
                      ("kijun", feats["ichimoku_kijun_sen"].to_numpy()[ok])):
        rel = num / close - 1
        mu2, sd2 = rel[:cut].mean(), rel[:cut].std()
        z = np.abs((rel - mu2) / (sd2 if sd2 else 1))[cut:]
        print(f"    改造为 {name}/close - 1 后: 后30% 平均|z| = {z.mean():.2f}  最大 {z.max():.2f}")
    tk = feats["ichimoku_tenkan_sen"].to_numpy()[ok] / feats["ichimoku_kijun_sen"].to_numpy()[ok] - 1
    mu3, sd3 = tk[:cut].mean(), tk[:cut].std()
    z3 = np.abs((tk - mu3) / (sd3 if sd3 else 1))[cut:]
    print(f"    新增 tenkan/kijun - 1 后   : 后30% 平均|z| = {z3.mean():.2f}  最大 {z3.max():.2f}")

    direction_vs_volatility(args.symbol, args.bars)
    redundancy_rank(args.symbol, args.bars)
    return 0




def direction_vs_volatility(symbol, n_bars, horizon=1, thr=0.001, n_null=20):
    """决定性诊断：特征到底在预测「波动大小」还是「涨跌方向」？

    生产标签 |未来收益| > 阈值 混了两件事：
      (a) 这一根会不会动（波动聚集，通常高度可预测）
      (b) 动的话往哪边（方向，才是交易真正需要的）

    k-NN 互信息估计器对连续特征有向上偏差，单个随机特征不足以当零分布，
    因此这里放 n_null 个随机列取零分布的最大值/95 分位，并对最好的特征做置换检验。
    """
    df, feats = load(symbol, n_bars)
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    fwd = np.full(n, np.nan)
    fwd[:n - horizon] = (close[horizon:] - close[:n - horizon]) / close[:n - horizon]
    ok = (np.arange(n) >= WARMUP) & ~np.isnan(fwd)
    X = feats.loc[ok, FEATURES].to_numpy(dtype=float)
    r = fwd[ok]

    moved = (np.abs(r) > thr).astype(int)
    dm = np.abs(r) > thr
    direction = (r[dm] > 0).astype(int)
    Xd = X[dm]

    rng = np.random.default_rng(0)

    def mi_with_null(Xk, yk, seed):
        r_ = np.random.default_rng(seed)
        N = r_.normal(size=(len(Xk), n_null))
        Xa = np.column_stack([Xk, N])
        mi = mutual_info_classif(Xa, yk, random_state=seed, n_neighbors=5)
        return mi[:Xk.shape[1]], mi[Xk.shape[1]:]

    mi_m, null_m = mi_with_null(X, moved, 1)
    mi_d, null_d = mi_with_null(Xd, direction, 2)

    print(f"\n{'='*100}")
    print(f"4. 决定性诊断：预测「会不会动」 vs 预测「往哪边」（{symbol}, h={horizon}, ±{thr:.1%}）")
    print(f"{'='*100}")
    print(f"  零分布（{n_null} 个随机特征）: 「会动」最大 {null_m.max():.5f}, "
          f"95分位 {np.quantile(null_m,0.95):.5f}")
    print(f"                                 「方向」最大 {null_d.max():.5f}, "
          f"95分位 {np.quantile(null_d,0.95):.5f}")
    print()
    print(f"{'特征':<24}{'MI(会动)':>11}{'过零':>6}{'MI(方向)':>11}{'过零':>6}")
    order = np.argsort(-mi_d)
    for i in order:
        print(f"{FEATURES[i]:<24}{mi_m[i]:>11.5f}"
              f"{('是' if mi_m[i] > null_m.max() else '否'):>6}"
              f"{mi_d[i]:>11.5f}{('是' if mi_d[i] > null_d.max() else '否'):>6}")

    n_over_m = int((mi_m > null_m.max()).sum())
    n_over_d = int((mi_d > null_d.max()).sum())
    print(f"\n  超过零分布最大值的特征数: 「会动」{n_over_m}/{len(FEATURES)}，"
          f"「方向」{n_over_d}/{len(FEATURES)}")

    # 对方向 MI 最高的特征做置换检验（打乱标签，看 MI 能有多高）
    top = int(order[0])
    perm = []
    for k in range(30):
        yp = np.random.default_rng(100 + k).permutation(direction)
        perm.append(float(mutual_info_classif(Xd[:, [top]], yp, random_state=0,
                                              n_neighbors=5)[0]))
    perm = np.array(perm)
    pval = float((perm >= mi_d[top]).mean())
    print(f"\n  置换检验（方向 MI 最高的 {FEATURES[top]}）:")
    print(f"    实测 MI = {mi_d[top]:.5f}   置换 30 次的 MI: 均值 {perm.mean():.5f}, "
          f"最大 {perm.max():.5f}")
    print(f"    p = {pval:.2f}  -> {'不显著（与随机无异）' if pval > 0.05 else '显著'}")
    print(f"\n  方向样本 {int(dm.sum())} 条，UP 占比 {direction.mean():.1%}")
    print(f"  任何常数预测的准确率上限 = {max(direction.mean(), 1-direction.mean()):.1%}")


def redundancy_rank(symbol, n_bars):
    """冗余的量化：这 17 维实际有几个独立方向（PCA 有效秩）"""
    df, feats = load(symbol, n_bars)
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    fwd = np.full(n, np.nan)
    fwd[:n - 1] = (close[1:] - close[:n - 1]) / close[:n - 1]
    ok = (np.arange(n) >= WARMUP) & ~np.isnan(fwd)
    X = feats.loc[ok, FEATURES].to_numpy(dtype=float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    ev = np.linalg.svd(Z, compute_uv=False) ** 2
    ev = ev / ev.sum()
    cum = np.cumsum(ev)
    eff = float(np.exp(-np.sum(ev * np.log(ev + 1e-12))))     # 熵有效秩
    print(f"\n{'='*100}")
    print(f"5. 冗余量化：17 维的 PCA 有效秩（{symbol}）")
    print(f"{'='*100}")
    print(f"  {len(FEATURES)} 个特征，熵有效秩 = {eff:.1f}")
    for k in (1, 3, 5, 8, 12, 17):
        print(f"    前 {k:>2} 个主成分解释方差 {cum[k-1]:.1%}")
    print(f"  → 名义 17 维，实际约 {eff:.0f} 个独立方向")


if __name__ == "__main__":
    raise SystemExit(main())
