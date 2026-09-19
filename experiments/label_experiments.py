"""
标签定义对比实验（A 固定阈值 / B 波动归一化 / C 分位数）

设计要点（保持原策略不被替换）：

- **Experiment A** ``fixed``：固定 ±0.1%，生产默认口径，作为对照基准。
- **Experiment B** ``volatility``：``threshold = k * sigma``（sigma 取过去 60 根收益标准差），
  k ∈ {0.5, 0.8, 1.0}；另附相对 ATR 口径。目的：让不同币种的任务难度可比。
- **Experiment C** ``quantile``：用**过去 500 根**收益的分位数当阈值（因果实现），
  类别天然均衡；代价是失去绝对收益含义。

三者都只是**实验标签**，不动生产默认（``config.LABEL_THRESHOLD``）。每个 (标签 × 币种 × 折)
都重新训练模型（标签变了，模型必须重训），评估用同一套标签。

指标：类分布、多数类基线、总准确率、**平衡准确率**（跨币种主指标）、逐类精度提升，
并为准确率给出 95% 置信区间 —— 样本量决定能分辨多大的差异（n=4000 时约 ±1.5pp）。

用法::

    .venv/bin/python experiments/label_experiments.py
    .venv/bin/python experiments/label_experiments.py --modes A_fixed_0.1pct,C_quantile
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
from scipy import stats  # noqa: E402

import models.enhanced_lstm as EL  # noqa: E402
from config import fetch_klines  # noqa: E402
from models.enhanced_lstm import EnhancedLSTMPredictor  # noqa: E402
from services.labeling import LABEL_MODES, get_label_fn, label_distribution  # noqa: E402
from services.model_meta import MODEL_SUBDIR  # noqa: E402

SCRATCH = "LBEXP"
COLS = ["open_time", "open", "high", "low", "close", "volume"]
WARMUP = 120
EPOCHS = 6
BATCH = 64
DEFAULT_MODES = ["A_fixed_0.1pct", "B_vol_k0.5", "B_vol_k0.8", "B_vol_k1.0", "C_quantile"]


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
        raw = fetch_klines(symbol, "1h", cursor - 1000 * 3600 * 1000, cursor, 1000)
        if not raw:
            break
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


def metrics(labels: np.ndarray, pred: np.ndarray) -> Dict:
    n = len(labels)
    cnt = np.bincount(labels, minlength=3)
    mr = cnt.max() / n
    correct = int((pred == labels).sum())
    acc = correct / n
    rec = [float((pred[labels == k] == k).mean()) for k in range(3) if (labels == k).sum()]
    prec = [float((labels[pred == k] == k).mean()) if (pred == k).sum() else float("nan")
            for k in range(3)]
    base = [float((labels == k).mean()) for k in range(3)]
    se = math.sqrt(acc * (1 - acc) / n)
    return {
        "n": n,
        "accuracy": acc,
        "accuracy_ci95": [acc - 1.96 * se, acc + 1.96 * se],
        "se": se,
        "majority_rate": float(mr),
        "edge_vs_majority_pp": round((acc - mr) * 100, 2),
        "balanced_accuracy": float(np.mean(rec)) if rec else 0.0,
        "balanced_ci95": [float(np.mean(rec)) - 1.96 * math.sqrt(np.mean(rec) * (1 - np.mean(rec)) / n),
                          float(np.mean(rec)) + 1.96 * math.sqrt(np.mean(rec) * (1 - np.mean(rec)) / n)]
        if rec else [0.0, 0.0],
        "class_dist": {"down": base[0], "neutral": base[1], "up": base[2]},
        "precision": {"down": prec[0], "neutral": prec[1], "up": prec[2]},
        "precision_lift_pp": {k: round((prec[i] - base[i]) * 100, 2)
                              for i, k in enumerate(("down", "neutral", "up"))},
        "p_value_vs_majority": float(stats.binomtest(correct, n, mr, alternative="greater").pvalue),
    }


def pool(per: List[Dict]) -> Dict:
    """合并多次运行（种子 × 折）。

    不确定性有两个来源，必须分开看：
      - **训练随机性**：同配置换种子重训，平衡准确率实测 σ≈1.9pp（这是主要来源）；
      - **抽样噪声**：测试样本数决定的二项标准误（n=2000 时约 1.1pp）。
    这里用"各种子均值 ± 种子间标准误"表示前者，另附抽样标准误。
    """
    n = sum(r["n"] for r in per)
    runs = len(per)
    acc = sum(r["accuracy"] * r["n"] for r in per) / n
    bal = sum(r["balanced_accuracy"] * r["n"] for r in per) / n
    mr = sum(r["majority_rate"] * r["n"] for r in per) / n
    run_bals = [r["balanced_accuracy"] for r in per]
    run_accs = [r["accuracy"] for r in per]
    se_seed = float(np.std(run_bals, ddof=1) / math.sqrt(runs)) if runs > 1 else float("nan")
    se_seed_acc = float(np.std(run_accs, ddof=1) / math.sqrt(runs)) if runs > 1 else float("nan")
    se = math.sqrt(acc * (1 - acc) / n)
    se_b = math.sqrt(bal * (1 - bal) / n)
    lift = {}
    for k in ("down", "neutral", "up"):
        vals = [r["precision_lift_pp"][k] for r in per if not math.isnan(r["precision_lift_pp"][k])]
        lift[k] = round(float(np.mean(vals)), 2) if vals else float("nan")
    dist = {k: float(np.mean([r["class_dist"][k] for r in per])) for k in ("down", "neutral", "up")}
    return {
        "runs": runs, "n": n, "accuracy": acc, "majority_rate": mr,
        "edge_vs_majority_pp": round((acc - mr) * 100, 2),
        "accuracy_ci95": [acc - 1.96 * se, acc + 1.96 * se],
        "balanced_accuracy": bal,
        "balanced_ci95": [bal - 1.96 * se_b, bal + 1.96 * se_b],
        # 训练随机性带来的不确定度（种子间标准误）与抽样标准误分开报告
        "balanced_seed_sem": se_seed,
        "accuracy_seed_sem": se_seed_acc,
        "balanced_sampling_se": se_b,
        "per_run_balanced": [round(b, 4) for b in run_bals],
        "balanced_edge_vs_chance_pp": round((bal - 1 / 3) * 100, 2),
        "class_dist": dist,
        "precision_lift_pp": lift,
        "p_value_vs_majority": float(stats.binomtest(int(round(acc * n)), n, mr,
                                                     alternative="greater").pvalue),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    ap.add_argument("--modes", default=",".join(DEFAULT_MODES))
    ap.add_argument("--bars", type=int, default=16000)
    ap.add_argument("--train0", type=int, default=12000)
    ap.add_argument("--fold", type=int, default=2000)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=1, help="每个单元格重复训练的随机种子数")
    ap.add_argument("--out", default="data/label_experiments.json")
    args = ap.parse_args()

    symbols = args.symbols.split(",")
    modes = args.modes.split(",")
    log(f"标签模式: {modes}")
    log(f"币种: {symbols} | 每币种 {args.bars} 根, 初始训练段 {args.train0}, {args.folds} 折 × {args.fold} 根")

    results: Dict[str, Dict[str, List[Dict]]] = {s: {m: [] for m in modes} for s in symbols}

    for symbol in symbols:
        bars = load_bars(symbol, args.bars, f"data/lb_{symbol}.json")
        df_all = to_df(bars)

        # 先看标签本身：各模式在不同币种上的类分布（与模型无关）
        log(f"--- {symbol} 标签分布（全段 {len(df_all)} 根）---")
        for mode in modes:
            d = label_distribution(get_label_fn(mode)(df_all))
            log(f"    {mode:<16} 跌 {d['down']*100:4.1f}%  平 {d['neutral']*100:4.1f}%  涨 {d['up']*100:4.1f}%")

        for k in range(args.folds):
            a = args.train0 + k * args.fold
            b = a + args.fold
            if b > len(bars):
                break
            ctx_start = max(a - WARMUP, 0)
            ctx_df = to_df(bars[ctx_start:b])
            idxs = [r - ctx_start for r in range(a + 1, b)]
            idxs = [i for i in idxs if EL.SEQUENCE_LENGTH <= i < len(ctx_df)]

            for mode in modes:
                fn = get_label_fn(mode)
                labels_ctx = fn(ctx_df)
                y = labels_ctx[idxs]
                for seed in range(args.seeds):
                    tf.keras.utils.set_random_seed(seed)
                    t0 = time.time()
                    p = EnhancedLSTMPredictor(symbol=SCRATCH)
                    p.train(bars[:a], epochs=EPOCHS, batch_size=BATCH, is_fine_tune=False, label_fn=fn)
                    M = p.build_input_matrix(p._compute_features(ctx_df))
                    X = np.array([M[i - EL.SEQUENCE_LENGTH:i] for i in idxs])
                    pred = p.model.predict(X, verbose=0).argmax(axis=1)
                    r = metrics(y, pred)
                    r["train_secs"] = round(time.time() - t0)
                    r["seed"] = seed
                    results[symbol][mode].append(r)
                    log(f"{symbol} fold{k} seed{seed} [{mode:<16}] 总准确率 {r['accuracy']*100:5.2f}% "
                        f"(基线 {r['majority_rate']*100:5.2f}%) 平衡 {r['balanced_accuracy']*100:5.2f}% "
                        f"中性占比 {r['class_dist']['neutral']*100:4.1f}% [{r['train_secs']}s]")

        os.remove(f"data/lb_{symbol}.json")

    # ---- 汇总 ----
    print()
    print("=" * 118)
    print("标签定义对比汇总（合并各折；平衡准确率是跨币种主指标，随机=33.33%）")
    print("=" * 118)
    print(f"{'币种':<9}{'标签模式':<16}{'类别分布 跌/平/涨':>18}{'基线':>7}"
          f"{'平衡准确率':>11}{'±种子SEM':>15}{'对随机':>9}{'运行数':>7}{'中性精度提升':>13}")
    summary = {}
    for symbol in symbols:
        for mode in modes:
            per = results[symbol][mode]
            if not per:
                continue
            s = pool(per)
            summary[f"{symbol}|{mode}"] = s
            d = s["class_dist"]
            dist = f"{d['down']*100:.0f}/{d['neutral']*100:.0f}/{d['up']*100:.0f}%"
            sem = s["balanced_seed_sem"]
            sem_txt = f"{sem*100:.2f}pp" if not math.isnan(sem) else "n/a"
            print(f"{symbol:<9}{mode:<16}{dist:>18}{s['majority_rate']*100:>6.1f}%"
                  f"{s['balanced_accuracy']*100:>10.2f}%{sem_txt:>15}"
                  f"{s['balanced_edge_vs_chance_pp']:>+8.2f}pp{s['runs']:>7d}"
                  f"{s['precision_lift_pp']['neutral']:>+12.2f}pp")
        print("-" * 118)

    # ---- 功效分析：要分辨观察到的差异需要多少种子 ----
    print()
    print("=" * 118)
    print("功效分析：分辨 A 与其它标签的差异需要多少种子")
    print("=" * 118)
    for symbol in symbols:
        base = summary.get(f"{symbol}|{modes[0]}")
        if not base:
            continue
        for mode in modes[1:]:
            cur = summary.get(f"{symbol}|{mode}")
            if not cur:
                continue
            delta = (cur["balanced_accuracy"] - base["balanced_accuracy"]) * 100
            sd = float(np.std(cur["per_run_balanced"] + base["per_run_balanced"], ddof=1) * 100) \
                if len(cur["per_run_balanced"]) + len(base["per_run_balanced"]) > 2 else float("nan")
            if math.isnan(sd) or abs(delta) < 1e-9:
                need = float("nan")
            else:
                # 两侧各 n 个种子，要求 2*SE_diff <= |delta|
                need = 2 * (2 * sd ** 2) / (delta ** 2)
            need_txt = f"{need:.1f}" if not math.isnan(need) else "n/a"
            print(f"  {symbol:<9} {modes[0]} vs {mode:<16} 差异 {delta:+5.2f}pp  "
                  f"运行间 σ≈{sd:4.2f}pp  ->  每侧需要约 {need_txt} 个种子才能以 2σ 分辨")

    os.makedirs("data", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"symbols": symbols, "modes": modes, "results": results, "summary": summary},
                  f, indent=2, ensure_ascii=False)
    log(f"结果已写入 {args.out}")

    for fn in os.listdir(MODEL_SUBDIR):
        if SCRATCH in fn:
            os.remove(os.path.join(MODEL_SUBDIR, fn))
    log("已清理训练中间产物")


if __name__ == "__main__":
    main()
