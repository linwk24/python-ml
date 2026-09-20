"""
"把未收盘的那根加回模型输入"会怎样？—— 用真实 1m K 线重建半根 K 线来实测

做法：
  1. 取 1h K 线（收盘的），以及同一时段的 1m K 线；
  2. 把某一小时之后的 1m 数据聚合成"走到 15/30/45 分钟"的**半根 1h K 线**
     （open=首根1m开盘, high=max, low=min, close=最后一根1m收盘, volume=求和）；
  3. 比较两种输入下模型的**特征值、z 分数、输出概率**：
       A. 只用已收盘 K 线（当前线上做法）
       B. 末尾拼上那根半成品（"加回来"的做法）

预期后果：半根 K 线的 price_change_pct / high_low_ratio 等特征只反映**部分**行程，
在训练中从未出现过（训练里每根都是完整的 60 分钟），属于语义级错配，比单纯 OOD 更糟。

用法：.venv/bin/python experiments/partial_bar_probe.py [SYMBOL] [--hours 3]
"""

import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.main import fetch_kline_data  # noqa: E402
from models.enhanced_lstm import EnhancedLSTMPredictor  # noqa: E402

HOUR_MS = 3600 * 1000
MIN_MS = 60 * 1000


def utc(ms):
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc)


def to_df(bars):
    df = pd.DataFrame(bars, columns=["open_time", "open", "high", "low",
                                     "close", "volume"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    return df


def partial_bar(m1_bars, hour_open_ms, minutes):
    """把某小时开头 minutes 分钟的 1m K 线聚合成一根半成品 1h K 线"""
    end = hour_open_ms + minutes * MIN_MS
    sel = [b for b in m1_bars if hour_open_ms <= b[0] < end]
    if not sel:
        return None
    return [float(hour_open_ms),
            float(sel[0][1]),                       # open = 首根开盘
            max(float(b[2]) for b in sel),          # high
            min(float(b[3]) for b in sel),          # low
            float(sel[-1][4]),                      # close = 最后一根收盘
            sum(float(b[5]) for b in sel)]          # volume


def sanity_check(p, h1, m1):
    """构造校验：走到第 60 分钟的半根，必须精确等于真实的已收盘 1h bar。

    若这一步不通过，后面 15/30/45 分钟的结论全部不可信。
    """
    bar = {int(b[0]): b for b in h1}
    checked = 0
    maxdiff = 0.0
    for ot in sorted(bar)[-8:]:
        full = bar[ot]
        pb = partial_bar(m1, ot, 60)
        if pb is None:
            continue
        d = max(abs(pb[k] - float(full[k])) for k in (1, 2, 3, 4))
        maxdiff = max(maxdiff, d)
        checked += 1
    ok = checked > 0 and maxdiff < 1e-6
    print(f"[构造校验] 对 {checked} 根真实 1h bar 重建 60 分钟版本，"
          f"OHLC 最大偏差 {maxdiff:.2e} -> {'通过 ✅' if ok else '**不通过** ❌'}")
    return ok


def aggregate(p, h1, m1, minutes, top):
    """扩大样本：对最近 top 根已收盘 K 线，统计"拼上半根"造成的扰动与方向翻转率"""
    base = to_df(h1)
    hours = [int(x) for x in base["open_time"].tolist()]
    rows = []
    for k in range(1, min(top + 1, len(hours) - 1)):
        idx = len(hours) - 1 - k
        h = hours[idx]
        nxt = h + HOUR_MS                      # 正在走的那根
        pb = partial_bar(m1, nxt, minutes)
        if pb is None:
            continue
        correct = h1[:idx + 1]
        wrong = correct + [pb]
        fA = p._compute_features(to_df(correct)).iloc[-1]
        fB = p._compute_features(to_df(wrong)).iloc[-1]
        zA = p.build_input_matrix(p._compute_features(to_df(correct)))[-1]
        zB = p.build_input_matrix(p._compute_features(to_df(wrong)))[-1]
        prA = p.predict(correct)["prediction"]
        prB = p.predict(wrong)["prediction"]
        rows.append({
            "hour": nxt,
            "d_rsi": fB["rsi"] - fA["rsi"],
            "d_pcp": fB["price_change_pct"] - fA["price_change_pct"],
            "d_hlr": fB["high_low_ratio"] - fA["high_low_ratio"],
            "dist": float(np.linalg.norm(zB - zA)),
            "flip": prA["trend_code"] != prB["trend_code"],
            "tA": prA["trend"], "tB": prB["trend"],
            "confA": prA["confidence"], "confB": prB["confidence"],
        })
    if not rows:
        print("样本不足")
        return None

    flips = sum(1 for r in rows if r["flip"])
    dist = np.array([r["dist"] for r in rows])
    drsi = np.array([abs(r["d_rsi"]) for r in rows])
    dconf = np.array([abs(r["confB"] - r["confA"]) for r in rows])
    print(f"\n{'='*100}")
    print(f"扩大样本（最近 {len(rows)} 根已收盘 K 线，半根走到第 {minutes} 分钟）")
    print(f"{'='*100}")
    n = len(rows)
    lo, hi = wilson(flips, n)
    print(f"  方向翻转: {flips}/{n} = {flips/n:.1%}   95% 区间 [{lo:.1%}, {hi:.1%}]")
    print(f"  输入向量位移(欧氏): 中位 {np.median(dist):.2f}  90分位 {np.quantile(dist,0.9):.2f}  最大 {dist.max():.2f}")
    print(f"  |Δrsi|:            中位 {np.median(drsi):.2f}  90分位 {np.quantile(drsi,0.9):.2f}  最大 {drsi.max():.2f}")
    print(f"  |Δconfidence|:     中位 {np.median(dconf):.2f}  90分位 {np.quantile(dconf,0.9):.2f}  最大 {dconf.max():.2f}")
    if flips:
        print(f"\n  发生翻转的样本:")
        for r in rows:
            if r["flip"]:
                t = utc(r["hour"])
                print(f"    {t:%m-%d %H:%M}  {r['tA']}(conf {r['confA']:.1f}) -> "
                      f"{r['tB']}(conf {r['confB']:.1f})   位移 {r['dist']:.2f}  Δrsi {r['d_rsi']:+.2f}")
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / d, (c + h) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?", default="BTCUSDT")
    ap.add_argument("--hours", type=int, default=1, help="详细打印最近几根")
    ap.add_argument("--agg", type=int, default=0,
                    help="扩大样本模式：统计最近 N 根的方向翻转率")
    ap.add_argument("--minutes", type=int, default=30, help="--agg 模式下模拟走到第几分钟")
    ap.add_argument("--m1", type=int, default=1000, help="抓多少根 1m K 线")
    args = ap.parse_args()
    symbol = args.symbol

    p = EnhancedLSTMPredictor(symbol=symbol)
    p.load_model()

    h1 = fetch_kline_data(symbol, "1h", 300)
    m1 = fetch_kline_data(symbol, "1m", args.m1)
    print(f"[{symbol}] 1h {len(h1)} 根，1m {len(m1)} 根 "
          f"({utc(m1[0][0]):%m-%d %H:%M} ~ {utc(m1[-1][0]):%m-%d %H:%M} UTC)")

    if not sanity_check(p, h1, m1):
        print("构造校验失败，后续结论不可信，已中止")
        return 1

    if args.agg:
        aggregate(p, h1, m1, args.minutes, args.agg)
        return 0
    # 从倒数第 2 根已收盘 K 线往回检查（最后一根可能还在走）
    base = to_df(h1)
    closed_hours = [int(x) for x in base["open_time"].tolist()]

    for k in range(1, args.hours + 1):
        idx = len(closed_hours) - 1 - k          # 已收盘的那根
        h = closed_hours[idx]
        # 它的**下一根**就是要模拟"正在走"的那根
        nxt = h + HOUR_MS

        correct = h1[:idx + 1]                   # A: 只用已收盘
        for mins in (15, 30, 45, 60):
            pb = partial_bar(m1, nxt, mins)
            if pb is None:
                print(f"  {utc(nxt):%m-%d %H:%M} +{mins}min: 无 1m 数据，跳过")
                continue
            wrong = correct + [pb]               # B: 拼上半成品

            # ── 特征对比 ──
            fA = p._compute_features(to_df(correct)).iloc[-1]
            fB = p._compute_features(to_df(wrong)).iloc[-1]
            # ── 模型输入 z 分数对比 ──
            zA = p.build_input_matrix(p._compute_features(to_df(correct)))[-1]
            zB = p.build_input_matrix(p._compute_features(to_df(wrong)))[-1]
            # ── 输出对比 ──
            prA = p.predict(correct)["prediction"]
            prB = p.predict(wrong)["prediction"]

            print(f"\n{'='*100}")
            print(f"正在走的那根 = {utc(nxt):%m-%d %H:%M} UTC，走到第 {mins} 分钟"
                  f"   （半根: O={pb[1]:,.2f} H={pb[2]:,.2f} L={pb[3]:,.2f} "
                  f"C={pb[4]:,.2f} 行程={(pb[4]-pb[1])/pb[1]:+.2%}）")
            print(f"{'='*100}")
            print(f"{'':<26}{'A 只用收盘K线':>18}{'B 拼上这半根':>18}{'差':>16}")
            print(f"{'特征 price_change_pct':<24}{fA['price_change_pct']:>18.5f}"
                  f"{fB['price_change_pct']:>18.5f}{fB['price_change_pct']-fA['price_change_pct']:>+16.5f}")
            print(f"{'特征 high_low_ratio':<24}{fA['high_low_ratio']:>18.5f}"
                  f"{fB['high_low_ratio']:>18.5f}{fB['high_low_ratio']-fA['high_low_ratio']:>+16.5f}")
            print(f"{'特征 close_open_ratio':<24}{fA['close_open_ratio']:>18.5f}"
                  f"{fB['close_open_ratio']:>18.5f}{fB['close_open_ratio']-fA['close_open_ratio']:>+16.5f}")
            print(f"{'特征 rsi':<26}{fA['rsi']:>18.2f}{fB['rsi']:>18.2f}"
                  f"{fB['rsi']-fA['rsi']:>+16.2f}")
            print(f"{'模型输入 max|z|':<24}{np.abs(zA).max():>18.2f}{np.abs(zB).max():>18.2f}"
                  f"{np.abs(zB).max()-np.abs(zA).max():>+16.2f}")
            print(f"{'输入向量与A的欧氏距离':<22}{0.0:>18.2f}"
                  f"{np.linalg.norm(zB-zA):>18.2f}")
            print(f"{'-'*100}")
            pa = prA["probabilities"]
            pb_ = prB["probabilities"]
            triA = f"{pa['看跌']:.1f}/{pa['中性']:.1f}/{pa['看涨']:.1f}"
            triB = f"{pb_['看跌']:.1f}/{pb_['中性']:.1f}/{pb_['看涨']:.1f}"
            flip = "   <== 方向变了!" if prA["trend"] != prB["trend"] else ""
            print(f"{'输出 看跌/中性/看涨':<22}{triA:>18}{triB:>18}")
            print(f"{'argmax 方向':<26}{prA['trend']:>18}{prB['trend']:>18}{flip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
