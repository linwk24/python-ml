"""
多币种 + 特征加权方式对比（实验脚本，不属于生产路径）

两个问题一起回答：

1. **这套方法是不是多币种通用的？** 同一套管线在 BTC / ETH / SOL 上分别做滚动前向验证，
   看是否都能显著超过各自的常量基线。
2. **特征加权该怎么加？** 对比：
     - ``none``        : 只用训练期统计量做标准化（当前默认）
     - ``mutual_info`` : 标准化**之后**再按训练段互信息缩放各特征（真正生效的那一种）

背景：早期实现把权重乘在标准化**之前**，会被下游 StandardScaler 逐列精确抵消
（实测极端权重与均匀权重的输入差异仅 3e-15）——那一版加权对模型完全没有影响。
本脚本验证"放到标准化之后"是否真的有用。

用法::

    .venv/bin/python experiments/multisymbol_folds.py
    .venv/bin/python experiments/multisymbol_folds.py --symbols BTCUSDT,ETHUSDT
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import models.enhanced_lstm as EL  # noqa: E402
from config import fetch_klines, LABEL_THRESHOLD  # noqa: E402
from models.enhanced_lstm import EnhancedLSTMPredictor  # noqa: E402
from services.model_meta import MODEL_SUBDIR  # noqa: E402

SCRATCH = "MSFOLD"
COLS = ["open_time", "open", "high", "low", "close", "volume"]
WARMUP = 120
EPOCHS = 6
BATCH = 64
VARIANTS = ["none", "mutual_info"]


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


def labels_for(closes, idxs):
    base = closes[np.array(idxs) - 1]
    nxt = closes[np.array(idxs)]
    ch = (nxt - base) / base
    return np.where(ch > LABEL_THRESHOLD, 2, np.where(ch < -LABEL_THRESHOLD, 0, 1))


def score(y_true, y_pred):
    n = len(y_true)
    cnt = np.bincount(y_true, minlength=3)
    mr = cnt.max() / n
    correct = int((y_pred == y_true).sum())
    acc = correct / n
    rec = [float((y_pred[y_true == k] == k).mean()) for k in range(3) if (y_true == k).sum()]
    pv = float(stats.binomtest(correct, n, mr, alternative="greater").pvalue)
    return {"n": n, "accuracy": acc, "majority_rate": float(mr), "edge_pp": round((acc - mr) * 100, 2),
            "balanced_accuracy": float(np.mean(rec)) if rec else 0.0, "p_value": pv}


def run_symbol(symbol, bars, train0, fold, n_folds, results):
    folds = []
    t = train0
    for _ in range(n_folds):
        if t + fold > len(bars):
            break
        folds.append((t, t + fold))
        t += fold

    for k, (a, b) in enumerate(folds):
        ctx_start = max(a - WARMUP, 0)
        ctx_df = to_df(bars[ctx_start:b])
        idxs = [r - ctx_start for r in range(a + 1, b)]
        idxs = [i for i in idxs if EL.SEQUENCE_LENGTH <= i < len(ctx_df)]
        closes = ctx_df["close"].to_numpy()
        y = labels_for(closes, idxs)
        const = np.full(len(y), int(np.bincount(y, minlength=3).argmax()))
        results[symbol]["常量基线"].append(score(y, const))

        for variant in VARIANTS:
            EL.FEATURE_WEIGHTING = variant
            t0 = time.time()
            p = EnhancedLSTMPredictor(symbol=SCRATCH)
            p.train(bars[:a], epochs=EPOCHS, batch_size=BATCH, is_fine_tune=False)
            # 一次性算全量特征再切片，与生产 predict 的输入路径完全一致
            feats = p._compute_features(ctx_df)
            matrix = p.build_input_matrix(feats)
            X = np.array([matrix[i - EL.SEQUENCE_LENGTH:i] for i in idxs])
            pred = p.model.predict(X, verbose=0).argmax(axis=1)
            r = score(y, pred)
            r["train_secs"] = round(time.time() - t0)
            r["mi_top"] = sorted(p.weight_analyzer.feature_mi.items(), key=lambda kv: -kv[1])[:5]
            results[symbol][variant].append(r)
            log(f"{symbol} fold{k} [{variant:<11}] 准确率 {r['accuracy']*100:5.2f}% "
                f"vs 基线 {r['majority_rate']*100:5.2f}% ({r['edge_pp']:+5.2f}pp, p={r['p_value']:.1e}) "
                f"[{r['train_secs']}s]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    ap.add_argument("--bars", type=int, default=14000)
    ap.add_argument("--train0", type=int, default=9000)
    ap.add_argument("--fold", type=int, default=2500)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--out", default="data/multisymbol.json")
    args = ap.parse_args()

    symbols = args.symbols.split(",")
    results: Dict[str, Dict[str, List[Dict]]] = {
        s: {"常量基线": [], **{v: [] for v in VARIANTS}} for s in symbols
    }

    for s in symbols:
        cache = f"data/ms_{s}.json"
        bars = load_bars(s, args.bars, cache)
        log(f"{s}: {len(bars)} 根 1h；fold 数 {args.folds} × {args.fold} 根")
        run_symbol(s, bars, args.train0, args.fold, args.folds, results)
        os.remove(cache)

    print()
    print("=" * 104)
    print("多币种 × 特征加权方式  合并结果（样本外，逐折重新训练）")
    print("=" * 104)
    print(f"{'币种':<10}{'方案':<14}{'合并准确率':>11}{'基线':>9}{'优势':>9}{'p值':>11}"
          f"{'平衡准确率':>11}{'样本数':>8}{'各折占优':>10}")
    summary = {}
    for s in symbols:
        for name in ["常量基线"] + VARIANTS:
            per = results[s][name]
            if not per:
                continue
            n = sum(r["n"] for r in per)
            acc = sum(r["accuracy"] * r["n"] for r in per) / n
            mr = sum(r["majority_rate"] * r["n"] for r in per) / n
            bal = sum(r["balanced_accuracy"] * r["n"] for r in per) / n
            correct = int(round(acc * n))
            pv = float(stats.binomtest(correct, n, mr, alternative="greater").pvalue)
            wins = sum(1 for r in per if r["accuracy"] > r["majority_rate"])
            summary[f"{s}|{name}"] = {"accuracy": acc, "majority_rate": mr,
                                      "edge_pp": round((acc - mr) * 100, 2), "p_value": pv,
                                      "balanced_accuracy": bal, "n": n,
                                      "folds_beating_baseline": wins, "folds": len(per)}
            print(f"{s:<10}{name:<14}{acc*100:10.2f}%{mr*100:8.2f}%{(acc-mr)*100:+8.2f}pp"
                  f"{pv:11.1e}{bal*100:10.2f}%{n:8d}{wins:6d}/{len(per)}")

    os.makedirs("data", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"symbols": symbols, "results": results, "summary": summary},
                  f, indent=2, ensure_ascii=False, default=str)
    log(f"结果已写入 {args.out}")

    for fn in os.listdir(MODEL_SUBDIR):
        if SCRATCH in fn:
            os.remove(os.path.join(MODEL_SUBDIR, fn))
    log("已清理训练中间产物")


if __name__ == "__main__":
    main()
