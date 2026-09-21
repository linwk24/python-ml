"""
状态机端到端验证：真实 OOS 数据上，状态机 vs 原始 argmax

与 `state_machine_pnl.py` 的区别：那个脚本是对**已落盘的 positions 序列**做规则叠加，
只能看到 argmax、拿不到每根的概率；本脚本走完整链路 ——

    K 线 -> 模型批量打分（batch_directional_scores）-> 信号状态机逐根推进 -> 仓位序列

因此它验证的是真东西：状态机、失效线、结构确认全部按生产代码路径跑。

只用模型 fit 窗口**之后**的数据（`*_train_meta.json` 的 train_end_ms），避免 in-sample。

用法：.venv/bin/python experiments/state_machine_validate.py [--symbols BTCUSDT,ETHUSDT,SOLUSDT]
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.main import atr_series, fetch_kline_data, _resolved_params  # noqa: E402
from models.enhanced_lstm import EnhancedLSTMPredictor, SEQUENCE_LENGTH  # noqa: E402
from services.model_meta import load_train_meta  # noqa: E402
from services.signal_state_machine import SignalStateMachine, StateMachineParams  # noqa: E402


def perf(pos, ret, fee):
    turn = np.abs(np.diff(pos, prepend=0.0))
    net = pos * ret - turn * fee
    eq = np.cumprod(1.0 + net)
    dd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    sd = net.std()
    return {
        "total": float(eq[-1] - 1.0), "maxdd": dd, "turnover": float(turn.sum()),
        "sharpe": float(net.mean() / sd * np.sqrt(24 * 365)) if sd else float("nan"),
        "exposure": float((pos != 0).mean()),
        "switches": int((np.diff(np.sign(pos)) != 0).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    ap.add_argument("--bars", type=int, default=12000)
    ap.add_argument("--fee-bp", type=float, default=5.0)
    ap.add_argument("--include-insample", action="store_true",
                    help="连 fit 窗口内的数据一起跑（默认只用 OOS）")
    args = ap.parse_args()
    fee = args.fee_bp / 10000.0

    print(f"口径: 只跑模型 fit 窗口之后的 OOS 数据 | 手续费 {args.fee_bp:.0f}bp/边")
    print("      状态机走生产代码路径（services/signal_state_machine）\n")

    rows = []
    for sym in args.symbols.split(","):
        meta = load_train_meta(sym) or {}
        fit_end = None if args.include_insample else meta.get("train_end_ms")
        bars = fetch_kline_data(sym, "1h", args.bars)
        df = pd.DataFrame(bars, columns=["open_time", "open", "high", "low",
                                         "close", "volume"])
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = df[c].astype(float)

        p = EnhancedLSTMPredictor(symbol=sym)
        p.load_model()

        t0 = time.time()
        idx = list(range(SEQUENCE_LENGTH, len(bars)))
        scores = p.batch_directional_scores(df, idx)
        print(f"[{sym}] 批量打分 {len(idx)} 根完成 [{time.time()-t0:.0f}s]")

        close = df["close"].to_numpy(float)
        ret = np.zeros(len(close))
        ret[1:] = np.diff(close) / close[:-1]
        atrs = atr_series(bars)
        params = StateMachineParams(**_resolved_params(sym))

        # 起点：首个有窗口的 bar，且（默认）在 fit 窗口之后
        start = SEQUENCE_LENGTH
        if fit_end is not None:
            after = [i for i in range(len(bars)) if bars[i][0] >= fit_end]
            start = max(start, after[0] if after else len(bars) - 1)

        sm = SignalStateMachine(params)
        sm_pos = np.zeros(len(bars))
        raw_pos = np.zeros(len(bars))
        for i in range(start, len(bars)):
            if i > start:
                raw_pos[i] = 1.0 if scores.get(i, 50.0) > 50 else (-1.0 if scores.get(i, 50.0) < 50 else 0.0)
            if i > start:
                snap = sm.step(score=scores.get(i, 50.0), close=close[i - 1],
                               high=float(df["high"].iloc[i - 1]),
                               low=float(df["low"].iloc[i - 1]), atr=atrs[i - 1])
                sm_pos[i] = snap.position

        sl = slice(start + 1, len(bars))
        r = ret[sl]
        a = perf(raw_pos[sl], r, fee)
        b = perf(sm_pos[sl], r, fee)
        rows.append((sym, a, b))
        print(f"[{sym}] OOS {len(r)} 根")
        print(f"    原始 argmax : 净 {a['total']:>+8.2%}  毛 {perf(raw_pos[sl], r, 0)['total']:>+8.2%}"
              f"  换手 {a['turnover']:>6.0f}  切换 {a['switches']:>4}  回撤 {a['maxdd']:>7.2%}")
        print(f"    状态机      : 净 {b['total']:>+8.2%}  毛 {perf(sm_pos[sl], r, 0)['total']:>+8.2%}"
              f"  换手 {b['turnover']:>6.0f}  切换 {b['switches']:>4}  回撤 {b['maxdd']:>7.2%}")
        print(f"    切换次数变化: {a['switches']} -> {b['switches']} "
              f"({b['switches']/max(a['switches'],1):.0%})，"
              f"换手 {a['turnover']:.0f} -> {b['turnover']:.0f}")
        print()

    if rows:
        print("=" * 92)
        print(f"{'币种':<10}{'argmax 净':>12}{'状态机 净':>12}{'差':>11}"
              f"{'argmax 切换':>13}{'状态机 切换':>13}")
        print("-" * 92)
        for sym, a, b in rows:
            print(f"{sym:<10}{a['total']:>12.2%}{b['total']:>12.2%}{b['total']-a['total']:>+11.2%}"
                  f"{a['switches']:>13}{b['switches']:>13}")
        an = np.mean([a["total"] for _, a, _ in rows])
        bn = np.mean([b["total"] for _, _, b in rows])
        ac = np.mean([a["switches"] for _, a, _ in rows])
        bc = np.mean([b["switches"] for _, _, b in rows])
        print("-" * 92)
        print(f"{'等权组合':<10}{an:>12.2%}{bn:>12.2%}{bn-an:>+11.2%}{ac:>13.0f}{bc:>13.0f}")
        wins = sum(1 for _, a, b in rows if b["total"] > a["total"])
        print(f"\n状态机优于 argmax 的币种数: {wins}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
