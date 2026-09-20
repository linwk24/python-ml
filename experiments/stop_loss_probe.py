"""
持仓的损失到底能不能"避免"？—— 止损（盘中触发）vs 收盘改信号

核心区别：
  - 收盘后改信号 / 强制空仓：只能在第 i 根**收盘后**生效，第 i 根自己已经吃满。
  - 止损：用盘中价格触发，**在 K 线收盘之前**就能出场 —— 这是唯一可能在
    冲击那根 K 线内部起作用的机制。

本脚本量化止损的可行性：对模型持有多仓的每一根 K 线，计算
  盘中最大不利偏移 MAE = (low - prev_close) / prev_close
以及"盘中触发止损但收盘又涨回来"的假触发比例。
假触发比例高 → 止损会在震荡中被反复打脸，省下的尾部损失被手续费和错过的反弹吃掉。

用法：.venv/bin/python experiments/stop_loss_probe.py [SYMBOL ...]
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
from services.model_meta import load_train_meta  # noqa: E402

sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))
from shock_replay import batch_trends  # noqa: E402


def utc(ms):
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc)


def analyse(symbol, n_bars=6000):
    meta = load_train_meta(symbol) or {}
    fit_end = meta.get("train_end_ms")
    p = EnhancedLSTMPredictor(symbol=symbol)
    p.load_model()
    bars = fetch_kline_data(symbol, "1h", n_bars)
    trends, _ = batch_trends(p, bars)

    df = pd.DataFrame(bars, columns=["open_time", "open", "high", "low",
                                     "close", "volume"]).astype(
        {c: float for c in ("open", "high", "low", "close", "volume")})

    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    c = df["close"].to_numpy()
    t = df["open_time"].to_numpy()

    prev_c = np.roll(c, 1)
    ret = np.zeros(len(c)); ret[1:] = (c[1:] - c[:-1]) / c[:-1]

    rows = []
    for i in range(61, len(c) - 1):
        if fit_end and t[i] < fit_end:
            continue                       # 只用模型没拟合过的区间
        d = trends[i]                       # 第 i 根开始时的模型方向（事前）
        if d is None or d == 1:             # 只看有方向的
            continue
        # 方向 d=2 看涨 -> 多仓；d=0 看跌 -> 空仓（做空）
        adverse = (lo[i] - prev_c[i]) / prev_c[i] if d == 2 else (prev_c[i] - h[i]) / prev_c[i]
        favor = (h[i] - prev_c[i]) / prev_c[i] if d == 2 else (prev_c[i] - lo[i]) / prev_c[i]
        rows.append({
            "ret": ret[i], "mae": adverse, "mfe": favor,
            "dir": int(d), "close_ret": ret[i],
        })

    if not rows:
        return None
    mae = np.array([r["mae"] for r in rows])
    ret = np.array([r["ret"] for r in rows])
    mfe = np.array([r["mfe"] for r in rows])

    out = {"symbol": symbol, "n": len(rows),
           "mae_median": float(np.median(mae)),
           "mae_p90": float(np.quantile(mae, 0.10)),
           "mae_p99": float(np.quantile(mae, 0.01)),
           "mae_worst": float(mae.min()),
           "close_mean": float(ret.mean())}

    # 各止损档位：触发频率、触发后收盘是否已经涨回（假触发）、假触发占比
    out["stops"] = {}
    for s in (0.005, 0.01, 0.02, 0.03, 0.05):
        hit = mae <= -s
        if hit.sum() == 0:
            out["stops"][s] = {"freq": 0.0, "n": 0}
            continue
        # 假触发：盘中触发了，但该根**收盘**时其实没亏到止损位
        false_trig = hit & (ret > -s)
        out["stops"][s] = {
            "freq": float(hit.mean()),
            "n": int(hit.sum()),
            "false_trig_share": float(false_trig.sum() / hit.sum()),
            # 触发后收盘仍亏（真该止损）的占比
            "real_share": float((hit & (ret <= -s)).sum() / hit.sum()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*", default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
    ap.add_argument("--bars", type=int, default=6000)
    args = ap.parse_args()

    print("口径：模型 fit 窗口之后（真 OOS）；只统计模型给了方向的 K 线；")
    print("      MAE = 该根盘中相对上一根收盘的最大不利偏移（多仓看 low，空仓看 high）")
    print("      假触发 = 盘中触及止损位，但该根收盘时并未亏到该位（即被盘中插针打掉）\n")

    results = []
    for s in args.symbols:
        try:
            r = analyse(s, args.bars)
        except Exception as e:
            print(f"[{s}] 跳过: {e}")
            continue
        if r:
            results.append(r)

    print(f"{'币种':<10}{'样本':>7}{'MAE中位':>10}{'MAE 10分位':>12}{'MAE 1分位':>11}"
          f"{'MAE最差':>10}{'该根平均收益':>13}")
    for r in results:
        print(f"{r['symbol']:<10}{r['n']:>7}{r['mae_median']:>10.2%}{r['mae_p90']:>12.2%}"
              f"{r['mae_p99']:>11.2%}{r['mae_worst']:>10.2%}{r['close_mean']:>13.3%}")

    print(f"\n{'='*96}")
    print("各止损档位：触发频率 与 假触发占比")
    print(f"{'='*96}")
    for s in (0.005, 0.01, 0.02, 0.03, 0.05):
        print(f"\n止损 -{s:.1%}:")
        print(f"  {'币种':<10}{'触发频率':>10}{'触发次数':>10}{'假触发占比':>12}{'真该止损占比':>14}")
        for r in results:
            v = r["stops"].get(s, {})
            if not v.get("n"):
                continue
            print(f"  {r['symbol']:<10}{v['freq']:>10.2%}{v['n']:>10}"
                  f"{v['false_trig_share']:>12.1%}{v['real_share']:>14.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
