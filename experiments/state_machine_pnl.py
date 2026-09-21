"""
信号状态机 + 移动失效线：用 **P&L 口径** 重测（而不是准确率口径）

背景
----
`services/trend_manager.py` 已经实现了带迟滞的 3 状态机（双阈值 + 绝对阈值 + 价格破位锚点），
但默认关闭。关闭的理由是**准确率**变差：

    原始模型 argmax（无状态机）: 37.70% ~ 40.32%
    状态机 55/45               : 32.20%
    状态机 52/48               : 33.48%
    状态机 51/49               : 34.76%

然而 README 的盈亏结论是：**瓶颈在换手与手续费，不在准确率**
（BTC 毛收益 +138%，5bp 手续费下 -20%；每笔毛边际 3~6bp < 成本 5~10bp）。

所以"牺牲几个百分点准确率、换取换手大幅下降"完全可能是净赚的 —— 当初用准确率否决它，
是在错的指标上做的决定。本脚本在净收益/换手/回撤口径下重测，并加入移动失效线（trailing）。

可测的规则（全部因果，只用第 i 根及之前的信息决定第 i+1 根仓位）：
  raw         : 原始 argmax（基准）
  hold_N      : 最短持有 N 根（换向要等 N 根）
  confirm_N   : 连续 N 根同向才切换（对信号做迟滞/确认）
  stop_X      : 自入场价固定止损 X%（用户设计里的 signal_price*(1-X)）
  trail_K     : 移动失效线：收盘 < 持有期最高收盘 × (1 - K*σ) 则平仓
  confirm+trail: 组合

σ 用过去 24 根收益的滚动标准差近似 ATR（本数据只有收盘价，没有 high/low，
因此无法测"盘中触发"的止损 —— 这也与用户偏好的"收盘确认"一致）。

用法：.venv/bin/python experiments/state_machine_pnl.py [--fee-bp 5]
"""

import argparse
import json
import os

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PNL_FILE = os.path.join(PROJECT_ROOT, "data", "pnl_backtest.json")


def load():
    with open(PNL_FILE) as f:
        d = json.load(f)
    out = {}
    for s in d["symbols"]:
        r = d["results"][s]
        out[s] = (np.array(r["positions"], float),
                  np.array(r["bar_returns"], float))
    return d, out


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
    }


def rule_hold(sig, n):
    """最短持有 n 根：换向需持仓满 n 根"""
    out = np.zeros_like(sig)
    cur, held = 0.0, 0
    for i, s in enumerate(sig):
        if s != cur and held >= n:
            cur, held = s, 1
        elif held == 0:
            cur, held = s, 1
        else:
            held += 1
        out[i] = cur
    return out


def rule_confirm(sig, n):
    """连续 n 根同向才切换（对信号做迟滞）"""
    out = np.zeros_like(sig)
    cur, cnt, cand = 0.0, 0, None
    for i, s in enumerate(sig):
        if s == cur:
            cnt, cand = 0, None
        else:
            if s == cand:
                cnt += 1
            else:
                cand, cnt = s, 1
            if cnt >= n:
                cur, cnt, cand = s, 0, None
        out[i] = cur
    return out


def _decide(idx, close):
    """决策第 idx 根仓位时，能看到的**最新收盘价**是第 idx-1 根的。

    这是本文件最容易出错的地方：out[i] 是"第 i 根持有、赚第 i 根收益"的仓位，
    因此它必须在第 i-1 根收盘时就定好。若用 close[i] 做判断，就是先看到这根
    亏了再躲开 —— 第一版实现正是这么写的，跑出 +950%/Sharpe 4.75 的假结果。
    """
    return close[idx - 1] if idx > 0 else close[0]


def rule_stop(pos, ret, close, pct):
    """自入场价固定止损：可见价格跌破 入场价*(1-pct) 则平仓。

    止损后**锁住**，等模型给出与"被否掉的信号"不同的新信号才允许再入场 ——
    否则会当根立刻重入，等于什么都没做。
    """
    out = np.zeros_like(pos)
    cur, entry, latch = 0.0, None, None
    for i in range(len(pos)):
        c = _decide(i, close)
        sig = pos[i]
        if cur != 0 and entry is not None:
            if (cur > 0 and c < entry * (1 - pct)) or \
               (cur < 0 and c > entry * (1 + pct)):
                latch, cur, entry = cur, 0.0, None
        if latch is not None and sig != latch:
            latch = None
        if latch is None:
            if cur == 0 and sig != 0:
                cur, entry = sig, c
            elif cur != 0 and (sig == 0 or np.sign(sig) != np.sign(cur)):
                cur, entry = (sig, c) if sig != 0 else (0.0, None)
        out[i] = cur
    return out


def rule_trail(pos, ret, close, sigma, k):
    """移动失效线（trailing）：可见价格跌破 持有期最高可见价*(1-k*σ) 则平仓。

    trailing 线随持仓期最高价抬高（做多）/ 降低（做空），即设计稿里
    "失效线跟随提高"的部分。σ 用过去 24 根收益的滚动标准差近似 ATR。
    """
    out = np.zeros_like(pos)
    cur, hwm, latch = 0.0, None, None
    for i in range(len(pos)):
        c = _decide(i, close)
        sig = pos[i]
        if cur != 0 and hwm is not None:
            sd = sigma[i] if not np.isnan(sigma[i]) else 0.0
            if cur > 0 and c < hwm * (1 - k * sd):
                latch, cur, hwm = cur, 0.0, None
            elif cur < 0 and c > hwm * (1 + k * sd):
                latch, cur, hwm = cur, 0.0, None
        if latch is not None and sig != latch:
            latch = None
        if latch is None:
            if cur == 0 and sig != 0:
                cur, hwm = sig, c
            elif cur != 0:
                if sig == 0 or np.sign(sig) != np.sign(cur):
                    cur, hwm = (sig, c) if sig != 0 else (0.0, None)
                else:
                    hwm = max(hwm, c) if cur > 0 else min(hwm, c)
        out[i] = cur
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fee-bp", type=float, default=5.0)
    args = ap.parse_args()
    fee = args.fee_bp / 10000.0

    meta, data = load()
    print(f"数据: data/pnl_backtest.json | 手续费 {args.fee_bp:.0f}bp/边 | "
          f"{meta['results'][meta['symbols'][0]]['bars']} 根 1h（1.37 年 OOS）\n")

    RULES = []
    RULES.append(("raw 原始 argmax", lambda p, r, c, s: p))
    for n in (2, 3, 6, 12, 24):
        RULES.append((f"hold_{n} 最短持有{n}根",
                      (lambda n: lambda p, r, c, s: rule_hold(p, n))(n)))
    for n in (2, 3, 5):
        RULES.append((f"confirm_{n} 连续{n}根确认",
                      (lambda n: lambda p, r, c, s: rule_confirm(p, n))(n)))
    for x in (0.01, 0.02, 0.03, 0.05):
        RULES.append((f"stop_{x:.0%} 入场价止损",
                      (lambda x: lambda p, r, c, s: rule_stop(p, r, c, x))(x)))
    for k in (1.5, 2.5, 4.0):
        RULES.append((f"trail_{k} 移动失效线 k*σ",
                      (lambda k: lambda p, r, c, s: rule_trail(p, r, c, s, k))(k)))
    for n, k in ((2, 2.5), (3, 2.5)):
        RULES.append((f"confirm_{n}+trail_{k}",
                      (lambda n, k: lambda p, r, c, s:
                       rule_trail(rule_confirm(p, n), r, c, s, k))(n, k)))

    rows = {}
    for name, fn in RULES:
        agg = []
        for sym in meta["symbols"]:
            pos, ret = data[sym]
            close = np.cumprod(np.concatenate([[1.0], 1.0 + ret[1:]]))
            close = close * (10000.0 / close[0])
            sigma = np.full(len(ret), np.nan)
            import pandas as pd
            sigma = pd.Series(ret).shift(1).rolling(24).std().to_numpy()
            newpos = fn(pos, ret, close, sigma)
            agg.append(perf(newpos, ret, fee))
        # 同时算 0bp（毛）结果：用来区分"赚得准"与"交易得少省手续费"
        gross = []
        for sym in meta["symbols"]:
            pos, ret = data[sym]
            close = np.cumprod(np.concatenate([[1.0], 1.0 + ret[1:]]))
            close = close * (10000.0 / close[0])
            import pandas as pd
            sigma = pd.Series(ret).shift(1).rolling(24).std().to_numpy()
            newpos = fn(pos, ret, close, sigma)
            gross.append(perf(newpos, ret, 0.0))
        rows[name] = (np.mean([a["total"] for a in agg]),
                      np.mean([a["maxdd"] for a in agg]),
                      np.mean([a["turnover"] for a in agg]),
                      np.mean([a["sharpe"] for a in agg]),
                      np.mean([a["exposure"] for a in agg]),
                      np.mean([g["total"] for g in gross]))

    base = rows["raw 原始 argmax"]
    print(f"{'规则':<28}{'净收益(5bp)':>13}{'毛收益(0bp)':>13}{'省下的手续费':>13}"
          f"{'换手':>8}{'回撤':>9}{'vs 基准':>10}")
    print("-" * 108)
    for name, _ in RULES:
        t, d, to, sh, ex, g = rows[name]
        fee_save = base[2] * fee - to * fee        # 相对基准少付的手续费
        print(f"{name:<28}{t:>13.2%}{g:>13.2%}{fee_save:>13.2%}"
              f"{to:>8.0f}{d:>9.1%}{t-base[0]:>+10.2%}")

    print(f"\n  基准毛收益 {base[5]:.2%} -> 最好规则的毛收益 "
          f"{max(rows.values(), key=lambda v: v[5])[5]:.2%}")
    print("  ★ 关键读法：若某规则的净收益提升 ≈ 它省下的手续费，")
    print("    说明它赚的是'交易得少'，不是'交易得准' —— 毛收益并未提高。")

    best = max(rows.items(), key=lambda kv: kv[1][0])
    print(f"\n净收益最高: {best[0]}  ({best[1][0]:+.2%}, 换手 {best[1][2]:.0f} vs "
          f"基准 {base[2]:.0f})")
    win = [n for n, v in rows.items() if v[0] > base[0]]
    print(f"优于基准的规则: {len(win)}/{len(RULES)}")
    for n in win:
        print(f"  {n:<28} {rows[n][0]:+.2%}  (换手 {rows[n][2]:.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
