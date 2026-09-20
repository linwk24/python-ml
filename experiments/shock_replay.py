"""
暴跌场景端到端回放：16:15 看涨，16:30 利空暴跌 —— 系统到底会返回什么

做法：在模型 fit 窗口**之后**的真实历史里找出最大的单根跌幅，然后把
"那一根正在走"/"那一根刚收盘"两种时点的 K 线切片喂给**生产同一条代码路径**
（fetch 口径 -> wrap_predict -> build_signal_block），逐时点打印 API 会返回什么。

这样得到的不是推演，而是"当时调用 /predict 会拿到什么"的真实答案。

用法：.venv/bin/python experiments/shock_replay.py [SYMBOL] [--top 5]
"""

import argparse
import datetime as dt
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.main import build_signal_block  # noqa: E402
from models.enhanced_lstm import EnhancedLSTMPredictor  # noqa: E402
from services.model_meta import load_train_meta  # noqa: E402

HOUR_MS = 3600 * 1000


def utc(ms):
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc)


def load_history(symbol, bars=6000):
    from app.main import fetch_kline_data
    return fetch_kline_data(symbol, "1h", bars)


def find_crashes(bars, fit_end_ms, top):
    """在 fit 窗口之后找出单根跌幅最大的 K 线"""
    closes = [b[4] for b in bars]
    opens = [b[0] for b in bars]
    cands = []
    for i in range(1, len(bars) - 2):
        if opens[i] < fit_end_ms:
            continue                      # 只取模型没拟合过的区间
        r = (closes[i] - closes[i - 1]) / closes[i - 1]
        cands.append((r, i))
    cands.sort()
    return cands[:top]


def replay(predictor, bars, i, label):
    """回放某一时点：把 K 线切片喂给生产代码路径，返回 API 会给出的结果"""
    sliced = bars[:i + 1]                 # 含第 i 根（= 第 i 根已收盘）
    pred = predictor.predict(sliced) if len(sliced) >= 120 else {"error": "K线不足"}
    if "error" in pred:
        return {"label": label, "error": pred["error"]}

    block = build_signal_block(sliced, pred)
    sr = block.get("signal_reversal") or {}
    eff = block["effective_signal"]
    p = pred["prediction"]
    return {
        "label": label,
        "closed_at": utc(sliced[-1][0] + HOUR_MS),
        "current_price": pred["current_price"],
        "trend": p["trend"],
        "trend_code": p["trend_code"],
        "confidence": p.get("confidence"),
        "effective_trend": eff["trend"],
        "effective_source": eff["source"],
        "reversal_triggered": sr.get("triggered"),
        "reversal_applied": sr.get("applied"),
        "corrected": sr.get("corrected_trend_code"),
        "price_change_pct": sr.get("price_change_pct"),
        "message": sr.get("message"),
    }


def show(rows, header):
    print(f"\n{header}")
    print("-" * 118)
    for r in rows:
        if "error" in r:
            print(f"{r['label']:<34}  {r['error']}")
            continue
        print(f"{r['label']:<34} 收盘 {r['closed_at']:%m-%d %H:%M} UTC  "
              f"价 {r['current_price']:>10,.2f}")
        print(f"{'':34} 模型方向 {r['trend']}({r['trend_code']}) "
              f"conf={r['confidence']}  -> effective_signal {r['effective_trend']}"
              f" [source={r['effective_source']}]")
        if r["message"]:
            print(f"{'':34} 反转: triggered={r['reversal_triggered']} "
                  f"applied={r['reversal_applied']} corrected={r['corrected']} "
                  f"价差={r['price_change_pct']}")
            print(f"{'':34} message: {r['message']}")
        print()


def batch_trends(predictor, bars):
    """批量算出每根 K 线**开始时刻**模型会给出的方向。

    预测第 i 根时，输入窗口是截至第 i-1 根收盘的 60 根，
    即标准化矩阵的 M[i-60:i] —— 与 predictor.predict(df[:i]) 等价，但一次算完。
    返回数组 t[i]，i < 60 处为 None。
    """
    import numpy as np
    from models.enhanced_lstm import SEQUENCE_LENGTH

    df = pd.DataFrame(bars, columns=["open_time", "open", "high", "low",
                                     "close", "volume"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    M = predictor.build_input_matrix(predictor._compute_features(df))
    L = SEQUENCE_LENGTH
    idx = list(range(L, len(M)))
    X = np.array([M[i - L:i] for i in idx])
    probs = predictor.model.predict(X, verbose=0, batch_size=256)
    trends = probs.argmax(axis=1)
    out = [None] * len(M)
    for k, i in enumerate(idx):
        out[i] = int(trends[k])
    return out, probs


def scan_bullish_then_crash(bars, trends, fit_end_ms, top):
    """找"模型看涨 -> 下一根暴跌"的真实事件（这正是用户设定的场景）"""
    closes = [b[4] for b in bars]
    hits = []
    for i in range(60, len(bars) - 2):
        if bars[i][0] < fit_end_ms:
            continue
        if trends[i] != 2:                     # 只关心事前看涨
            continue
        r = (closes[i] - closes[i - 1]) / closes[i - 1]
        if r < 0:
            hits.append((r, i))
    hits.sort()
    return hits[:top]


def tail_stats(bars, trends, fit_end_ms):
    """看涨/看跌各自的下一根收益分布：看涨到底有没有保护作用"""
    import numpy as np

    closes = np.array([b[4] for b in bars], dtype=float)
    ret = np.zeros(len(closes))
    ret[1:] = np.diff(closes) / closes[:-1]
    oos = np.array([b[0] >= fit_end_ms for b in bars])
    oos[:60] = False
    out = {}
    for code, name in ((2, "看涨"), (0, "看跌"), (1, "中性")):
        m = np.array([t == code for t in trends]) & oos
        if m.sum() == 0:
            continue
        r = ret[m]
        out[name] = {
            "n": int(m.sum()),
            "mean": float(r.mean()),
            "p_le_2": float((r <= -0.02).mean()),
            "p_le_5": float((r <= -0.05).mean()),
            "p_ge_2": float((r >= 0.02).mean()),
            "worst": float(r.min()),
            "q01": float(np.quantile(r, 0.01)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?", default="BTCUSDT")
    ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--scan", action="store_true",
                    help="找“模型事前看涨 -> 随后暴跌”的真实事件（默认只找最大跌幅）")
    args = ap.parse_args()
    symbol = args.symbol

    meta = load_train_meta(symbol) or {}
    fit_end_ms = meta.get("train_end_ms")
    if not fit_end_ms:
        print(f"{symbol} 缺少 train_end_ms，无法确定 OOS 区间")
        return 1
    print(f"[{symbol}] 模型 fit 窗口止于 {utc(fit_end_ms):%Y-%m-%d %H:%M} UTC"
          f"（只回放此后的真实历史，避免 in-sample）")

    bars = load_history(symbol)
    print(f"[{symbol}] 取到 {len(bars)} 根 1h K 线，"
          f"{utc(bars[0][0]):%Y-%m-%d} ~ {utc(bars[-1][0]):%Y-%m-%d}")

    predictor = EnhancedLSTMPredictor(symbol=symbol)
    predictor.load_model()

    trends, _ = batch_trends(predictor, bars)

    # ── 事前看涨 vs 事前看跌：谁的尾部风险更大？──
    ts = tail_stats(bars, trends, fit_end_ms)
    print(f"\n{'='*118}")
    print("事前方向 vs 下一根收益的尾部（OOS 区间，即 fit 窗口之后）")
    print(f"{'='*118}")
    print(f"{'事前方向':<10}{'样本':>8}{'均值':>10}{'跌>=2%':>10}{'跌>=5%':>10}"
          f"{'涨>=2%':>10}{'1% 分位':>11}{'最差一根':>11}")
    for name, v in ts.items():
        print(f"{name:<10}{v['n']:>8}{v['mean']:>10.3%}{v['p_le_2']:>10.1%}"
              f"{v['p_le_5']:>10.1%}{v['p_ge_2']:>10.1%}{v['q01']:>11.2%}{v['worst']:>11.2%}")

    if args.scan:
        crashes = scan_bullish_then_crash(bars, trends, fit_end_ms, args.top)
        title = "模型事前【看涨】却随后暴跌"
        if not crashes:
            print(f"\nOOS 区间内没有出现“事前看涨 -> 随后下跌”的样本")
            return 0
    else:
        crashes = find_crashes(bars, fit_end_ms, args.top)
        title = "单根跌幅最大"

    for rank, (r, i) in enumerate(crashes, 1):
        bar_open = utc(bars[i][0])
        print(f"\n{'='*118}")
        print(f"【{title} 第 {rank} 名】{bar_open:%Y-%m-%d %H:%M} ~ "
              f"{bar_open + dt.timedelta(hours=1):%H:%M} UTC 跌 {r:+.2%}   "
              f"{bars[i][1]:,.2f} -> {bars[i][4]:,.2f}")
        print(f"{'='*118}")

        mid = replay(predictor, bars, i - 1, f"① {bar_open:%H:%M} 盘中(暴跌进行中)")
        after = replay(predictor, bars, i,
                       f"② {bar_open + dt.timedelta(hours=1):%H:%M} 该根收盘后")
        nxt = replay(predictor, bars, i + 1,
                     f"③ {bar_open + dt.timedelta(hours=2):%H:%M} 再下一根")

        show([mid], "① 暴跌进行中（这一根未收盘，被 drop_unclosed 丢弃）")
        show([after, nxt], "② 该根收盘后 / ③ 再下一根（此时暴跌已进入模型输入窗口）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
