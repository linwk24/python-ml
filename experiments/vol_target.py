"""
事前缩仓（波动率目标）实测 —— 唯一不依赖预测、也不被"假触发"惩罚的风控

思路：不试图预测冲击，而是让**敞口**随近期已实现波动率反向缩放。
     scale[i] = min(1, target_vol / realized_vol[i-1])   （只用 i-1 及之前的信息）
暴跌来临时，你已经因为波动率升高而缩小了仓位，同样的百分比跌幅换算成金额更小。

同时如实计入：缩放本身会产生换手，每一次调整都要付手续费。

对比三条曲线（全部在 data/pnl_backtest.json 的 1.37 年 OOS 上）：
  base      原始仓位（±1 / 0）
  voltarget 波动率目标缩仓
  flatgate  冲击后强制空仓 1 根（上一步已证伪，此处作为对照）

用法：.venv/bin/python experiments/vol_target.py
"""

import argparse
import json
import os

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PNL_FILE = os.path.join(PROJECT_ROOT, "data", "pnl_backtest.json")


def realized_vol(ret, window):
    """因果的已实现波动率：用 i-1 及之前的 window 根（pandas 滚动，比 Python 循环快约 100 倍）"""
    import pandas as pd
    return pd.Series(ret).shift(1).rolling(window).std().to_numpy()


def perf(pos, ret, fee):
    turn = np.abs(np.diff(pos, prepend=0.0))
    net = pos * ret - turn * fee
    eq = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak - 1.0)
    ann = net.mean() * 24 * 365
    sd = net.std() * np.sqrt(24 * 365)
    return {
        "total": float(eq[-1] - 1.0),
        "maxdd": float(dd.min()),
        "sharpe": float(ann / sd) if sd else float("nan"),
        "turnover": float(turn.sum()),
        "avg_abs_pos": float(np.abs(pos).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fee-bp", type=float, default=5.0)
    ap.add_argument("--window", type=int, default=24, help="已实现波动率窗口（根）")
    ap.add_argument("--target", type=float, default=0.006,
                    help="目标单根波动率（0.006 ≈ 0.6%/根，接近 BTC 常态）")
    args = ap.parse_args()
    fee = args.fee_bp / 10000.0

    with open(PNL_FILE) as f:
        d = json.load(f)

    print(f"口径: data/pnl_backtest.json | 手续费 {args.fee_bp:.0f}bp 每边 | "
          f"波动率窗口 {args.window} 根 | 目标波动 {args.target:.2%}/根\n")

    rows = {}
    for s in d["symbols"]:
        r = d["results"][s]
        pos = np.array(r["positions"], dtype=float)
        ret = np.array(r["bar_returns"], dtype=float)

        vol = realized_vol(ret, args.window)
        scale = np.ones(len(pos))
        ok = ~np.isnan(vol) & (vol > 0)
        scale[ok] = np.minimum(1.0, args.target / vol[ok])
        vt = pos * scale

        gated = pos.copy()
        for i in range(1, len(ret)):
            if abs(ret[i - 1]) >= 0.02:
                gated[i] = 0.0

        rows[s] = {
            "base": perf(pos, ret, fee),
            "voltarget": perf(vt, ret, fee),
            "flatgate": perf(gated, ret, fee),
        }

    for name, key in (("原始仓位", "base"), ("波动率缩仓", "voltarget"), ("冲击后空仓", "flatgate")):
        print(f"{'='*94}")
        print(f"【{name}】")
        print(f"{'='*94}")
        print(f"{'币种':<10}{'净收益':>12}{'最大回撤':>12}{'Sharpe':>10}{'换手':>12}{'平均|仓位|':>12}")
        for s in d["symbols"]:
            v = rows[s][key]
            print(f"{s:<10}{v['total']:>12.2%}{v['maxdd']:>12.2%}{v['sharpe']:>10.2f}"
                  f"{v['turnover']:>12.0f}{v['avg_abs_pos']:>12.3f}")
        tot = np.mean([rows[s][key]["total"] for s in d["symbols"]])
        dd = np.mean([rows[s][key]["maxdd"] for s in d["symbols"]])
        sh = np.mean([rows[s][key]["sharpe"] for s in d["symbols"]])
        to = np.mean([rows[s][key]["turnover"] for s in d["symbols"]])
        ap_ = np.mean([rows[s][key]["avg_abs_pos"] for s in d["symbols"]])
        print(f"{'等权组合':<10}{tot:>12.2%}{dd:>12.2%}{sh:>10.2f}{to:>12.0f}{ap_:>12.3f}")
        print()

    # 敏感度：目标波动率
    print(f"{'='*94}")
    print("敏感度：目标波动率（等权组合净收益 / 平均回撤）")
    print(f"{'='*94}")
    print(f"{'目标波动':>10}{'净收益':>14}{'最大回撤':>14}{'平均|仓位|':>12}{'换手':>12}")
    for target in (0.004, 0.005, 0.006, 0.008, 0.010, 0.015):
        agg_t, agg_d, agg_p, agg_to = [], [], [], []
        for s in d["symbols"]:
            r = d["results"][s]
            pos = np.array(r["positions"], dtype=float)
            ret = np.array(r["bar_returns"], dtype=float)
            vol = realized_vol(ret, args.window)
            sc = np.ones(len(pos))
            ok = ~np.isnan(vol) & (vol > 0)
            sc[ok] = np.minimum(1.0, target / vol[ok])
            p = perf(pos * sc, ret, fee)
            agg_t.append(p["total"]); agg_d.append(p["maxdd"])
            agg_p.append(p["avg_abs_pos"]); agg_to.append(p["turnover"])
        print(f"{target:>10.2%}{np.mean(agg_t):>14.2%}{np.mean(agg_d):>14.2%}"
              f"{np.mean(agg_p):>12.3f}{np.mean(agg_to):>12.0f}")
    print()
    print("注：原始仓位的平均|仓位| = 0.730（BTC 口径），缩仓后必然更低 ——")
    print("    净收益下降是**必然**的（敞口小了），关键看回撤是否下降得更多。")

    # ── 对准用户的问题：冲击事件上的损失被缩小了多少 ──
    print(f"\n{'='*94}")
    print("冲击事件上的效果：损失没有被'避免'，而是被**缩小**了")
    print(f"{'='*94}")
    print(f"{'目标波动':>10}{'事件数':>9}{'锁死损失(原始)':>16}{'锁死损失(缩仓)':>16}"
          f"{'缩减比例':>11}{'冲击前平均仓位':>16}")
    for target in (0.004, 0.006, 0.010, 0.015):
        tot_b, tot_v, n_ev = 0.0, 0.0, 0
        pos_frac = []
        for s in d["symbols"]:
            r = d["results"][s]
            pos = np.array(r["positions"], dtype=float)
            ret = np.array(r["bar_returns"], dtype=float)
            vol = realized_vol(ret, args.window)
            sc = np.ones(len(pos))
            ok = ~np.isnan(vol) & (vol > 0)
            sc[ok] = np.minimum(1.0, target / vol[ok])
            for i in range(1, len(ret) - 1):
                if abs(ret[i]) >= 0.02 and pos[i] != 0:
                    n_ev += 1
                    tot_b += pos[i] * ret[i]
                    tot_v += pos[i] * sc[i] * ret[i]
                    pos_frac.append(sc[i])
        red = (tot_v - tot_b) / tot_b if tot_b else float("nan")
        print(f"{target:>10.2%}{n_ev:>9}{tot_b:>16.2%}{tot_v:>16.2%}"
              f"{-red:>11.1%}{np.mean(pos_frac):>16.1%}")
    print()
    print("  解读：缩仓后的损失同号且更小 —— 它不改变方向、不择时，只是让暴跌落在更小的敞口上。")
    print("        '冲击前平均仓位'就是暴跌发生那一刻你实际持有的敞口比例。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
