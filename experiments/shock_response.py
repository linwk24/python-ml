"""
极端行情（瀑布/暴涨）下模型行为的实测分析

复用 data/pnl_backtest.json 里逐 bar 的 OOS 产物：
  positions[i]    = 用截至第 i-1 根**收盘**的窗口做的模型 argmax（+1 多 / -1 空 / 0 平）
  bar_returns[i]  = (close[i] - close[i-1]) / close[i-1]

因此 positions[i] 回答的是「第 i 根 K 线发生暴跌/暴涨时，模型在这一根开始时站在哪一边」——
模型事前不可能知道这根 K 线会怎么走，所以这是它面对冲击的**事前暴露**。
positions[i+1] 则是冲击**收盘后**模型的重新定位（此时冲击已进入它的输入窗口）。

注意：这里只测纯模型输出，不含 signal_reversal 与 trend_manager
（线上 SIGNAL_REVERSAL_APPLY=false，本来也只是给建议）。

用法：python3 experiments/shock_response.py
"""

import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PNL_FILE = os.path.join(PROJECT_ROOT, "data", "pnl_backtest.json")

# 冲击定义：(标签, 判定函数)  —— 用当期已实现收益，模拟"事后看这是根冲击 K 线"
SHOCK_DEFS = [
    ("极端 0.1% 分位", lambda r, q: np.abs(r) >= q["q999"]),
    ("极端 0.5% 分位", lambda r, q: np.abs(r) >= q["q995"]),
    ("极端 1% 分位", lambda r, q: np.abs(r) >= q["q99"]),
    ("|涨跌| >= 2%", lambda r, q: np.abs(r) >= 0.02),
    ("|涨跌| >= 5%", lambda r, q: np.abs(r) >= 0.05),
]


def load(symbol):
    with open(PNL_FILE) as f:
        d = json.load(f)
    r = d["results"][symbol]
    return (
        np.array(r["positions"], dtype=float),
        np.array(r["bar_returns"], dtype=float),
    )


def event_study(pos, ret, symbol, verbose=True):
    q = {
        "q99": float(np.quantile(np.abs(ret), 0.99)),
        "q995": float(np.quantile(np.abs(ret), 0.995)),
        "q999": float(np.quantile(np.abs(ret), 0.999)),
    }
    dir_mask = pos != 0
    base_hit = float((np.sign(pos[dir_mask]) == np.sign(ret[dir_mask])).mean())
    base_active = float(dir_mask.mean())

    rows = []
    for label, fn in SHOCK_DEFS:
        mask = np.array([bool(fn(r, q)) for r in ret])
        # 冲击 K 线必须不是最后几根，才能看 i+1
        mask[-2:] = False
        n = int(mask.sum())
        if n == 0:
            continue
        idx = np.where(mask)[0]

        pos_into = pos[idx]              # 冲击当根开始时的仓位（事前）
        r_shock = ret[idx]
        active = pos_into != 0
        hit = (np.sign(pos_into[active]) == np.sign(r_shock[active]))
        # 逆势暴露：模型事前方向与冲击方向相反，即"被瀑布打脸"
        against = active & (np.sign(pos_into) != np.sign(r_shock))

        pos_after = pos[idx + 1]         # 冲击收盘后的方向
        r_after = ret[idx + 1]           # 冲击后一根的实际收益

        # 模型在冲击后是否翻向（相对冲击方向）
        flipped = active & (np.sign(pos_after) != np.sign(pos_into))
        # 冲击后一根，模型方向是否押对了
        m_after = pos_after != 0
        hit_after = (np.sign(pos_after[m_after]) == np.sign(r_after[m_after]))

        rows.append({
            "label": label,
            "n": n,
            "abs_ret_mean": float(np.abs(r_shock).mean()),
            "active_share_into": float(active.mean()),
            "flat_share_into": float((~active).mean()),
            "hit_into": float(hit.mean()) if active.sum() else float("nan"),
            "against_share": float(against.mean()),
            "loss_when_against": float((-np.sign(pos_into[against]) * r_shock[against]).mean()) if against.sum() else float("nan"),
            "active_share_after": float((pos_after != 0).mean()),
            "flip_share": float(flipped.mean()) if active.sum() else float("nan"),
            "hit_after": float(hit_after.mean()) if m_after.sum() else float("nan"),
            "ret_after_mean": float(r_after.mean()),
            "ret_after_abs_mean": float(np.abs(r_after).mean()),
            # 反转/延续：冲击后一根是否延续冲击方向
            "continuation_share": float((np.sign(r_after) == np.sign(r_shock)).mean()),
        })

    if verbose:
        print(f"\n{'='*104}")
        print(f"{symbol}  —— 基线：模型有方向占比 {base_active:.1%}，"
              f"全样本方向命中率 {base_hit:.2%}，|收益| 99 分位 {q['q99']:.2%}")
        print(f"{'='*104}")
        print(f"{'冲击定义':<16}{'样本':>6}{'平均|涨跌|':>11}"
              f"{'事前有方向':>11}{'事前命中':>10}{'逆势被套':>10}"
              f"{'事后有方向':>11}{'事后翻向':>10}{'事后命中':>10}{'冲击后延续':>11}")
        for r in rows:
            print(f"{r['label']:<16}{r['n']:>6}{r['abs_ret_mean']:>10.2%}"
                  f"{r['active_share_into']:>11.1%}{r['hit_into']:>10.1%}{r['against_share']:>10.1%}"
                  f"{r['active_share_after']:>11.1%}{r['flip_share']:>10.1%}"
                  f"{r['hit_after']:>10.1%}{r['continuation_share']:>11.1%}")
        print()
        print("  读法：")
        print("    事前有方向 = 冲击 K 线开始时模型没空仓、给了多空方向的比例（空仓=报中性）")
        print("    事前命中   = 事前方向与冲击方向同号的比例；50% 即等于抛硬币")
        print("    逆势被套   = 事前方向与冲击方向**相反**的比例，这些 bar 模型是硬吃整段行情的")
        print("    事后翻向   = 冲击收盘后模型把方向调头（相对事前）的比例")
        print("    冲击后延续 = 冲击后一根继续同向的比例（>50% 为动量延续，<50% 为均值回复）")
    return {"symbol": symbol, "base_hit": base_hit, "base_active": base_active, "rows": rows}


def wilson(k, n, z=1.96):
    """Wilson 区间：小样本比例不能只看点估计"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / d, (c + h) / d)


def pooled_test(symbols_pos_ret, label_fn, qkey="q99"):
    """多币种合并 + 置信区间：单币种极端桶只有 4~120 个样本，功效不够"""
    rows = []
    for symbol, (pos, ret) in symbols_pos_ret.items():
        q = {"q99": float(np.quantile(np.abs(ret), 0.99)),
             "q995": float(np.quantile(np.abs(ret), 0.995)),
             "q999": float(np.quantile(np.abs(ret), 0.999))}
        mask = np.array([bool(label_fn(r, q)) for r in ret])
        mask[-2:] = False
        idx = np.where(mask)[0]
        act = idx[pos[idx] != 0]
        hit = (np.sign(pos[act]) == np.sign(ret[act]))
        base = (np.sign(pos[pos != 0]) == np.sign(ret[pos != 0]))
        rows.append({
            "symbol": symbol, "n": len(idx), "n_active": len(act),
            "hits": int(hit.sum()), "base_hits": int(base.sum()), "base_n": len(base),
            "base_active": float((pos != 0).mean()),
        })

    k = sum(r["hits"] for r in rows)
    n = sum(r["n_active"] for r in rows)
    bk = sum(r["base_hits"] for r in rows)
    bn = sum(r["base_n"] for r in rows)
    # 收手率：冲击时给方向的比例 vs 该币种平时的给方向比例
    act_shock = n / sum(r["n"] for r in rows)
    act_base = sum(r["base_active"] * r["base_n"] for r in rows) / bn
    lo, hi = wilson(k, n)
    blo, bhi = wilson(bk, bn)
    # 两比例 z 检验：冲击桶命中率 vs 基线命中率
    p1, p2 = k / n, bk / bn
    pp = (k + bk) / (n + bn)
    se = (pp * (1 - pp) * (1 / n + 1 / bn)) ** 0.5
    z = (p1 - p2) / se if se else float("nan")
    return {"rows": rows, "n": n, "hit": p1, "ci": (lo, hi),
            "base_hit": p2, "base_ci": (blo, bhi), "base_n": bn, "z": z,
            "act_shock": act_shock, "act_base": act_base}


def main():
    if not os.path.exists(PNL_FILE):
        print(f"缺少 {PNL_FILE}，先跑 experiments/pnl_backtest.py")
        return 1

    with open(PNL_FILE) as f:
        meta = json.load(f)
    print(f"数据来源: data/pnl_backtest.json  label_mode={meta['label_mode']}")
    print("口径: 纯模型 argmax（OOS 逐 fold 重训），不含 signal_reversal / trend_manager")

    all_out = []
    pos_ret = {}
    for symbol in meta["symbols"]:
        pos, ret = load(symbol)
        pos_ret[symbol] = (pos, ret)
        all_out.append(event_study(pos, ret, symbol))

    # ── 合并检验：单币种极端桶样本太少，必须合起来看 ──
    print(f"\n{'='*104}")
    print("合并三币种 + Wilson 95% 置信区间（单币种极端桶 n 只有 4~120，点估计不可信）")
    print(f"{'='*104}")
    print(f"{'冲击定义':<18}{'有效样本':>9}{'给方向占比':>12}{'平时给方向':>12}"
          f"{'事前命中':>10}{'95% 区间':>20}{'基线命中':>10}{'差值':>9}{'z':>7}")
    for label, fn in SHOCK_DEFS:
        t = pooled_test(pos_ret, fn)
        diff = t["hit"] - t["base_hit"]
        ci = f"[{t['ci'][0]:.1%}, {t['ci'][1]:.1%}]"
        print(f"{label:<18}{t['n']:>9}{t['act_shock']:>12.1%}{t['act_base']:>12.1%}"
              f"{t['hit']:>10.1%}{ci:>20}{t['base_hit']:>10.1%}{diff:>+9.1%}{t['z']:>7.2f}")
    print()
    print("  给方向占比 = 模型在冲击 K 线上**没有报中性**（报了多或空）的比例")
    print("  平时给方向 = 同一批模型在全样本上的给方向比例")
    print("  → 若冲击时该比例不降反升，说明模型在极端行情下不会收手，反而更爱给方向")
    print("  基线命中 = 同一批模型在全样本上的方向命中率（约 51%，样本量 ~32000）")
    print("  z 为两比例检验统计量；|z| < 2 时该差值在统计上不可区分")

    # 汇总：极端冲击下的总体表现
    print(f"\n{'='*104}")
    print("汇总 —— 模型在极端行情下的系统性倾向")
    print(f"{'='*104}")
    for o in all_out:
        r = next((x for x in o["rows"] if x["label"] == "极端 1% 分位"), None)
        r5 = next((x for x in o["rows"] if x["label"] == "|涨跌| >= 5%"), None)
        if r:
            print(f"{o['symbol']:<10} 极端1%: 事前命中 {r['hit_into']:.1%} (基线 {o['base_hit']:.1%})，"
                  f"逆势被套 {r['against_share']:.1%}，事后翻向 {r['flip_share']:.1%}")
        if r5:
            print(f"{'':<10} |涨跌|>=5%: n={r5['n']}，事前命中 {r5['hit_into']:.1%}，"
                  f"逆势被套 {r5['against_share']:.1%}，事后命中 {r5['hit_after']:.1%}")

    out_path = os.path.join(PROJECT_ROOT, "data", "shock_response.json")
    with open(out_path, "w") as f:
        json.dump({"results": all_out}, f, indent=2, ensure_ascii=False)
    print(f"\n已保存: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
