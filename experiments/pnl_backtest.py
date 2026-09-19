"""
盈亏（P&L）评估：把预测信号接到最简仓位规则上，扣费后看能不能赚钱

分类指标（准确率/平衡准确率）已经探到底：公允标签下三个币种的平衡准确率优势只有
+1~2.4pp，且主要来自"识别低波动"。这一层要回答的是更实际的问题：
**这点优势扣掉手续费之后还剩多少？**

仓位规则（不含任何调参）：
    看涨 -> +1 单位；看跌 -> -1 单位；中性 -> 空仓
    在每根 K 线收盘时按信号调整仓位，持有到下一根收盘
    turnover = |新仓位 - 旧仓位|，手续费 = turnover * fee_rate * 名义

评估设计：
- **严格样本外**：expanding-window 滚动前向，每个 fold 只用该 fold 之前的数据重训
  （特征权重/统计量/模型全部重拟合），按顺序拼接各 fold 的收益得到一条权益曲线。
- **基准**：买入持有、恒定做多、恒定做空、空仓。
- **置换检验（关键）**：把信号序列做**循环移位**（保持仓位分布与换手结构，只破坏与收益的
  时间对齐），重复 N 次得到零分布 —— 只有当真实收益落在零分布尾部时，才能说这点收益
  来自"对齐"而不是来自"经常做空碰上跌市"这类结构性偏差。
- **手续费敏感性**：0 / 5 / 10 bps 每边（现货约 10bps，合约 taker 约 5bps）。

用法::

    .venv/bin/python experiments/pnl_backtest.py
    .venv/bin/python experiments/pnl_backtest.py --symbols BTCUSDT,ETHUSDT --folds 3
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

import models.enhanced_lstm as EL  # noqa: E402
from config import fetch_klines  # noqa: E402
from models.enhanced_lstm import EnhancedLSTMPredictor  # noqa: E402
from services.labeling import get_label_fn  # noqa: E402
from services.model_meta import MODEL_SUBDIR  # noqa: E402

SCRATCH = "PNLEXP"
COLS = ["open_time", "open", "high", "low", "close", "volume"]
WARMUP = 120
EPOCHS = 6
BATCH = 64
FEE_RATES = [0.0, 0.0005, 0.0010]   # 每边手续费
N_PERMUTATIONS = 200
BARS_PER_YEAR = 24 * 365


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_bars(symbol, total, cache):
    if os.path.exists(cache):
        bars = json.load(open(cache))
        if len(bars) >= total:
            return bars[:total]
    log(f"下载 {symbol} 1h x{total} ...")
    cursor = int(time.time() * 1000)
    chunks, got = [], 0
    while got < total:
        raw = None
        for attempt in range(5):
            try:
                raw = fetch_klines(symbol, "1h", cursor - 1000 * 3600 * 1000, cursor, 1000)
                break
            except Exception as e:
                wait = 2 ** attempt
                log(f"  取数失败({type(e).__name__}), {wait}s 后重试 ({attempt+1}/5)")
                time.sleep(wait)
        if not raw:
            log("  多次重试仍失败，中止本次下载")
            raise RuntimeError(f"{symbol} 数据下载失败")
        ch = [[float(r[i]) for i in range(6)] for r in raw]
        chunks.append(ch)
        got += len(ch)
        cursor = int(ch[0][0]) - 1
        time.sleep(0.2)
    bars = [r for c in reversed(chunks) for r in c]
    seen, out = set(), []
    for r in bars:
        if r[0] not in seen:
            seen.add(r[0])
            out.append(r)
    out.sort(key=lambda r: r[0])
    out = out[:total]
    json.dump(out, open(cache, "w"))
    return out


def to_df(bars):
    df = pd.DataFrame(bars, columns=COLS)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["money_flow"] = df["typical_price"] * df["volume"]
    return df.dropna().reset_index(drop=True)


# ------------------------------------------------------------------ 绩效

def perf(step_returns: np.ndarray, positions: np.ndarray) -> Dict:
    """由逐根净收益与仓位算绩效"""
    step_returns = np.asarray(step_returns, dtype=float)
    equity = np.cumprod(1.0 + step_returns)
    n = len(step_returns)
    total = float(equity[-1] - 1.0) if n else 0.0
    years = n / BARS_PER_YEAR
    if n > 1 and np.std(step_returns, ddof=1) > 0:
        sharpe = float(np.mean(step_returns) / np.std(step_returns, ddof=1) * math.sqrt(BARS_PER_YEAR))
    else:
        sharpe = 0.0
    peak = np.maximum.accumulate(equity) if n else np.array([1.0])
    dd = float(np.max(1.0 - equity / peak)) if n else 0.0
    active = np.abs(positions) > 0
    return {
        "bars": n, "years": round(years, 2),
        "total_return": total,
        "annualized": float((1.0 + total) ** (1 / years) - 1) if years > 0 and total > -1 else float("nan"),
        "sharpe": sharpe,
        "max_drawdown": dd,
        "hit_rate": float((step_returns[active] > 0).mean()) if active.any() else 0.0,
        "exposure": float(active.mean()),
        "turnover": float(np.abs(np.diff(positions, prepend=0)).sum()),
    }


def strategy_returns(positions: np.ndarray, bar_returns: np.ndarray, fee: float) -> np.ndarray:
    """position[i] 是持有到第 i 根收盘的仓位；turnover 在换仓时收费"""
    turnover = np.abs(np.diff(positions, prepend=0.0))
    return positions * bar_returns - turnover * fee


def circular_shift_pvalue(positions, bar_returns, fee, actual_total, n_perm=N_PERMUTATIONS,
                          seed=0) -> Dict:
    """循环移位置换检验：保持仓位分布与换手结构，只破坏与收益的时间对齐"""
    rng = np.random.default_rng(seed)
    n = len(positions)
    if n < 100:
        return {"p_value": float("nan"), "null_mean": float("nan"), "null_std": float("nan"), "n_perm": 0}
    totals = []
    lo, hi = max(int(n * 0.05), 1), min(int(n * 0.95), n - 1)
    for _ in range(n_perm):
        shift = int(rng.integers(lo, hi))
        shifted = np.roll(positions, shift)
        totals.append(float(np.prod(1.0 + strategy_returns(shifted, bar_returns, fee)) - 1.0))
    totals = np.array(totals)
    p = float((totals >= actual_total).mean())
    return {"p_value": p, "null_mean": float(totals.mean()), "null_std": float(totals.std(ddof=1)),
            "n_perm": len(totals)}


# ------------------------------------------------------------------ 主流程

def run_symbol(symbol, bars, train0, fold, n_folds, label_mode, out: Dict):
    fn = get_label_fn(label_mode)
    closes = np.array([float(b[4]) for b in bars])
    all_pos, all_ret, all_idx = [], [], []

    for k in range(n_folds):
        a = train0 + k * fold
        b = a + fold
        if b > len(bars):
            break
        ctx_start = max(a - WARMUP, 0)
        ctx_df = to_df(bars[ctx_start:b])
        idxs = [r - ctx_start for r in range(a + 1, b)]
        idxs = [i for i in idxs if EL.SEQUENCE_LENGTH <= i < len(ctx_df) - 1]

        t0 = time.time()
        p = EnhancedLSTMPredictor(symbol=SCRATCH)
        p.train(bars[:a], epochs=EPOCHS, batch_size=BATCH, is_fine_tune=False, label_fn=fn)
        M = p.build_input_matrix(p._compute_features(ctx_df))
        X = np.array([M[i - EL.SEQUENCE_LENGTH:i] for i in idxs])
        pred = p.model.predict(X, verbose=0).argmax(axis=1)
        # 2 -> +1 做多, 0 -> -1 做空, 1 -> 空仓
        pos = np.where(pred == 2, 1.0, np.where(pred == 0, -1.0, 0.0))

        c = ctx_df["close"].to_numpy(dtype=float)
        # 持有期收益：position 在第 i-1 根收盘建立，赚第 i 根的收益
        ret = np.array([(c[i] - c[i - 1]) / c[i - 1] for i in idxs])
        all_pos.append(pos)
        all_ret.append(ret)
        all_idx.append(idxs)
        log(f"{symbol} fold{k} 训练/预测完成 [{time.time()-t0:.0f}s] 信号: "
            f"多 {int((pred==2).sum())} 空 {int((pred==0).sum())} 平 {int((pred==1).sum())}")

    if not all_pos:
        return
    positions = np.concatenate(all_pos)
    rets = np.concatenate(all_ret)

    row = {"symbol": symbol, "label_mode": label_mode, "bars": len(positions),
           "long_share": float((positions > 0).mean()), "short_share": float((positions < 0).mean()),
           "flat_share": float((positions == 0).mean())}

    for fee in FEE_RATES:
        tag = f"{fee*10000:.0f}bp"
        net = strategy_returns(positions, rets, fee)
        p_ = perf(net, positions)
        row[f"net_{tag}_total"] = p_["total_return"]
        row[f"net_{tag}_sharpe"] = p_["sharpe"]
        row[f"net_{tag}_maxdd"] = p_["max_drawdown"]
        row[f"net_{tag}_annualized"] = p_["annualized"]
        if fee == 0.0005:
            row["hit_rate"] = p_["hit_rate"]
            row["exposure"] = p_["exposure"]
            row["turnover"] = p_["turnover"]
            perm = circular_shift_pvalue(positions, rets, fee, p_["total_return"])
            row["perm_p_value"] = perm["p_value"]
            row["perm_null_mean"] = perm["null_mean"]
            row["perm_null_std"] = perm["null_std"]
            # 保存逐根净收益与仓位，供事后做组合/合并显著性检验
            row["net_returns_5bp"] = [round(float(x), 8) for x in net]
            row["positions"] = [int(x) for x in positions]
            row["bar_returns"] = [round(float(x), 8) for x in rets]

    # 基准
    bh = perf(rets, np.ones_like(rets))
    row["buy_hold_total"] = bh["total_return"]
    row["buy_hold_maxdd"] = bh["max_drawdown"]
    row["buy_hold_sharpe"] = bh["sharpe"]
    row["always_long_5bp"] = perf(strategy_returns(np.ones_like(rets), rets, 0.0005),
                                  np.ones_like(rets))["total_return"]
    row["always_short_5bp"] = perf(strategy_returns(-np.ones_like(rets), rets, 0.0005),
                                   -np.ones_like(rets))["total_return"]
    out[symbol] = row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    ap.add_argument("--bars", type=int, default=24000)
    ap.add_argument("--train0", type=int, default=12000)
    ap.add_argument("--fold", type=int, default=3000)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--label-mode", default="A_fixed_0.1pct")
    ap.add_argument("--out", default="data/pnl_backtest.json")
    args = ap.parse_args()

    symbols = args.symbols.split(",")
    log(f"标签模式 {args.label_mode} | 币种 {symbols} | 每币种 {args.bars} 根 1h "
        f"| 初始训练段 {args.train0} | {args.folds} 折 × {args.fold} 根")
    log(f"手续费敏感性: {[f'{f*10000:.0f}bp' for f in FEE_RATES]} 每边; 置换检验 {N_PERMUTATIONS} 次")

    out: Dict[str, Dict] = {}
    for symbol in symbols:
        ckpt = f"data/pnl_ckpt_{symbol}.json"
        if os.path.exists(ckpt):
            out[symbol] = json.load(open(ckpt))
            log(f"{symbol}: 复用已完成结果（{ckpt}）")
            continue
        bars = load_bars(symbol, args.bars, f"data/pnl_{symbol}.json")
        run_symbol(symbol, bars, args.train0, args.fold, args.folds, args.label_mode, out)
        os.remove(f"data/pnl_{symbol}.json")
        with open(ckpt, "w") as f:
            json.dump(out[symbol], f)
        log(f"{symbol}: 结果已落盘（断点续跑用）")

    # ---- 报告 ----
    print()
    print("=" * 132)
    print("盈亏评估（严格样本外，逐折重训；仓位 = 看涨+1 / 看跌-1 / 中性空仓；持有 1 根）")
    print("=" * 132)
    print(f"{'币种':<9}{'样本外':>8}{'多/空/平':>16}{'净收益@5bp':>11}{'年化':>9}{'Sharpe':>8}"
          f"{'最大回撤':>9}{'换手':>8}{'胜率':>8}{'置换检验p':>11}{'零分布均值':>12}")
    for s, r in out.items():
        mix = f"{r['long_share']*100:.0f}/{r['short_share']*100:.0f}/{r['flat_share']*100:.0f}%"
        print(f"{s:<9}{r['bars']:>8}{mix:>16}{r['net_5bp_total']*100:>10.2f}%"
              f"{r['net_5bp_annualized']*100:>8.2f}%{r['net_5bp_sharpe']:>8.2f}"
              f"{r['net_5bp_maxdd']*100:>8.2f}%{r['turnover']:>8.0f}{r['hit_rate']*100:>7.1f}%"
              f"{r['perm_p_value']:>11.3f}{r['perm_null_mean']*100:>11.2f}%")
    print("-" * 132)
    print(f"{'基准':<9}")
    for s, r in out.items():
        print(f"  {s:<9} 买入持有 {r['buy_hold_total']*100:+7.2f}% (回撤 {r['buy_hold_maxdd']*100:5.1f}%)"
              f" | 恒定做多@5bp {r['always_long_5bp']*100:+7.2f}%"
              f" | 恒定做空@5bp {r['always_short_5bp']*100:+7.2f}%")
    print("-" * 132)
    print(f"{'手续费敏感性（净收益）':<20}" + "".join(f"{f'{f*10000:.0f}bp':>14}" for f in FEE_RATES))
    for s, r in out.items():
        print(f"  {s:<18}" + "".join(f"{r[f'net_{f*10000:.0f}bp_total']*100:>13.2f}%" for f in FEE_RATES))

    os.makedirs("data", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"symbols": symbols, "label_mode": args.label_mode,
                   "fee_rates": FEE_RATES, "results": out}, f, indent=2, ensure_ascii=False)
    log(f"结果已写入 {args.out}")

    for fn in os.listdir(MODEL_SUBDIR):
        if SCRATCH in fn:
            os.remove(os.path.join(MODEL_SUBDIR, fn))
    log("已清理训练中间产物")


if __name__ == "__main__":
    main()
