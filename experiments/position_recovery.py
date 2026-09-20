"""
「信号转中性」到底能救回多少仓位损失？—— 把损失拆成"锁死的"和"可避免的"

用户提出的关键质疑：已经持有多仓时，把信号改成中性只能改变**之后**的仓位，
那根暴跌 K 线本身的亏损已经吃掉了，任何"收盘后改信号"的规则都救不回来。

本脚本量化这个拆分：
  锁死部分 = 冲击 K 线当根的盈亏（持仓在冲击发生前就已建立）
  可救部分 = 冲击 K 线收盘之后、模型仍维持原方向期间累计的盈亏

并实测一个"冲击后强制空仓一根"的叠加规则，含手续费。

用法：.venv/bin/python experiments/position_recovery.py
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PNL_FILE = os.path.join(PROJECT_ROOT, "data", "pnl_backtest.json")


def load():
    with open(PNL_FILE) as f:
        d = json.load(f)
    out = {}
    for s in d["symbols"]:
        r = d["results"][s]
        out[s] = {
            "pos": np.array(r["positions"], dtype=float),
            "ret": np.array(r["bar_returns"], dtype=float),
        }
    return d, out


def decompose(data, shock_thr, max_hold=24):
    """把冲击事件的盈亏拆成 锁死 / 可救 两部分"""
    ev = []
    for s, v in data.items():
        pos, ret = v["pos"], v["ret"]
        n = len(ret)
        for i in range(1, n - 1):
            if abs(ret[i]) < shock_thr or pos[i] == 0:
                continue
            d = pos[i]                       # 冲击当根持有的方向（事前已建立）
            locked = d * ret[i]              # 当根盈亏：锁死
            # 之后模型维持同方向期间累计
            j, cont, bars = i + 1, 0.0, 0
            while j < n and pos[j] == d and bars < max_hold:
                cont += d * ret[j]
                bars += 1
                j += 1
            ev.append({
                "symbol": s, "i": i, "dir": int(d),
                "shock_ret": float(ret[i]), "locked": float(locked),
                "cont": float(cont), "cont_bars": bars,
                "flip": bool(j < n and pos[j] != d),
            })
    return ev


def report_structure(ev, shock_thr):
    if not ev:
        print("无样本")
        return
    locked = np.array([e["locked"] for e in ev])
    cont = np.array([e["cont"] for e in ev])
    bars = np.array([e["cont_bars"] for e in ev])

    print(f"\n{'='*100}")
    print(f"冲击事件结构（|单根涨跌| >= {shock_thr:.0%} 且事前已持仓），共 {len(ev)} 个事件")
    print(f"{'='*100}")
    print(f"{'':<26}{'均值':>12}{'中位数':>12}{'合计':>12}")
    print(f"{'锁死部分(冲击当根)':<22}{locked.mean():>12.3%}{np.median(locked):>12.3%}{locked.sum():>12.3%}")
    print(f"{'可救部分(之后维持期间)':<20}{cont.mean():>12.3%}{np.median(cont):>12.3%}{cont.sum():>12.3%}")
    print(f"{'事件平均维持根数':<22}{bars.mean():>12.1f}{np.median(bars):>12.1f}{bars.sum():>12.0f}")
    print()
    print(f"  锁死部分占两者绝对值之和的 {abs(locked).sum()/(abs(locked).sum()+abs(cont).sum()):.1%}")
    print(f"  其中 {int((cont < 0).sum())}/{len(ev)} 个事件在冲击后继续亏（即空仓确实能省）")
    print(f"  另外 {int((cont > 0).sum())}/{len(ev)} 个事件在冲击后反弹（空仓反而少赚）")
    print(f"  按币种拆:")
    for s in sorted({e['symbol'] for e in ev}):
        sub = [e for e in ev if e["symbol"] == s]
        l = np.array([e["locked"] for e in sub])
        c = np.array([e["cont"] for e in sub])
        print(f"    {s:<10} n={len(sub):>4}  锁死合计 {l.sum():>+9.3%}  可救合计 {c.sum():>+9.3%}"
              f"  (继续亏 {sum(1 for e in sub if e['cont']<0)}/{len(sub)})")


def overlay_flat_after_shock(data, shock_thr, fee, cooldown=1):
    """叠加规则：刚收盘的那根若为冲击，则接下来 cooldown 根强制空仓"""
    totals = {}
    for s, v in data.items():
        pos, ret = v["pos"].copy(), v["ret"]
        gated = pos.copy()
        n = len(ret)
        for i in range(1, n):
            if abs(ret[i - 1]) >= shock_thr:
                for k in range(i, min(i + cooldown, n)):
                    gated[k] = 0.0
        def net(p):
            turn = np.abs(np.diff(p, prepend=0.0))
            return float(np.prod(1.0 + p * ret - turn * fee) - 1.0)
        totals[s] = {"base": net(pos), "gated": net(gated),
                     "flat_bars": int((gated == 0).sum() - (pos == 0).sum())}
    return totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shock", type=float, default=0.02)
    ap.add_argument("--fee-bp", type=float, default=5.0)
    args = ap.parse_args()

    meta, data = load()
    print(f"数据: data/pnl_backtest.json  label_mode={meta['label_mode']}  "
          f"每币种 {meta['results'][meta['symbols'][0]]['bars']} 根 1h（1.37 年 OOS）")

    for thr in sorted({0.01, 0.02, args.shock}):
        ev = decompose(data, thr)
        report_structure(ev, thr)

    fee = args.fee_bp / 10000.0
    print(f"\n{'='*100}")
    print(f"叠加规则实测：冲击后强制空仓 1 根（手续费 {args.fee_bp:.0f}bp 每边）")
    print(f"{'='*100}")
    print(f"{'币种':<10}{'冲击阈值':>10}{'原始净收益':>14}{'加空仓规则':>14}{'差':>12}{'多出的空仓根数':>16}")
    for thr in (0.01, 0.02, 0.03):
        t = overlay_flat_after_shock(data, thr, fee)
        for s, v in t.items():
            print(f"{s:<10}{thr:>10.0%}{v['base']:>14.2%}{v['gated']:>14.2%}"
                  f"{v['gated']-v['base']:>+12.2%}{v['flat_bars']:>16}")
        # 等权组合
        base = np.mean([t[s]["base"] for s in t])
        gated = np.mean([t[s]["gated"] for s in t])
        print(f"{'等权组合':<10}{thr:>10.0%}{base:>14.2%}{gated:>14.2%}{gated-base:>+12.2%}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
