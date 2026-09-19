"""
滚动前向验证 + 模型复杂度对比（实验脚本，不属于生产路径）

回答两个问题：

② **+8pp 的优势是不是单段行情的运气？**
   用 expanding-window 滚动前向验证：每个 fold 只用该 fold **之前**的数据重新训练
   （特征权重、标准化统计量、模型全部重拟合），在该 fold 上评估，再汇总所有 fold。

③ **非要这套 17 维 × 60 步的 LSTM 吗？**
   在**同一批 fold、同一套标签**上并行跑：
     - 逻辑回归（窗口最后一根的 17 维特征）
     - 逻辑回归（只用训练 fold 上选出的 top-k 特征）
     - 梯度提升树（同样只用最后一根特征）
     - LSTM（生产架构，17 维 × 60 步）
   谁能超过常量基线，谁就够用；简单模型能打平就没必要上深度模型。

所有评估都对齐模型的标签口径：窗口 = 行 [i-60, i-1]，标签 = 行 i-1 -> 行 i。
对比基线 = 该 fold 的多数类（最强的常量预测器），用单侧二项检验判显著。

用法::

    .venv/bin/python experiments/walk_forward.py                       # 默认 26000 根
    .venv/bin/python experiments/walk_forward.py --bars 20000 --folds 3
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
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.feature_selection import mutual_info_classif  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from tensorflow.keras.callbacks import EarlyStopping  # noqa: E402

from config import fetch_klines, LABEL_THRESHOLD  # noqa: E402
from models.enhanced_lstm import (  # noqa: E402
    FEATURE_NAMES,
    SEQUENCE_LENGTH,
    EnhancedLSTMPredictor,
)
from services.model_meta import MODEL_SUBDIR  # noqa: E402

SYMBOL = "BTCUSDT"
SCRATCH_SYMBOL = "WFOLD"          # 训练中间产物落盘用，跑完清理
COLS = ["open_time", "open", "high", "low", "close", "volume"]
WARMUP = 120                       # 每个 fold 评估所需的前置上下文
EPOCHS = 10
BATCH = 64
TOP_K = 5


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ 数据

def load_bars(total: int, cache: str) -> List[List[float]]:
    if os.path.exists(cache):
        bars = json.load(open(cache))
        if len(bars) >= total:
            return bars[:total] if len(bars) > total else bars
    log(f"下载 {SYMBOL} 1h K线 x{total} ...")
    cursor = int(time.time() * 1000)
    chunks, got = [], 0
    while got < total:
        raw = fetch_klines(SYMBOL, "1h", cursor - 1000 * 3600 * 1000, cursor, 1000)
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
    json.dump(out, open(cache, "w"))
    return out[:total]


def to_df(bars) -> pd.DataFrame:
    df = pd.DataFrame(bars, columns=COLS)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["money_flow"] = df["typical_price"] * df["volume"]
    return df.dropna().reset_index(drop=True)


def three_class_labels(closes, idxs):
    base = closes[np.array(idxs) - 1]
    nxt = closes[np.array(idxs)]
    ch = (nxt - base) / base
    return np.where(ch > LABEL_THRESHOLD, 2, np.where(ch < -LABEL_THRESHOLD, 0, 1))


# ------------------------------------------------------------------ 指标

def metrics(y_true, y_pred) -> Dict:
    n = len(y_true)
    cnt = np.bincount(y_true, minlength=3)
    mc = int(cnt.argmax())
    mr = cnt[mc] / n
    correct = int((y_pred == y_true).sum())
    acc = correct / n
    rec = [float((y_pred[y_true == k] == k).mean()) for k in range(3) if (y_true == k).sum()]
    return {
        "n": n,
        "accuracy": acc,
        "majority_rate": mr,
        "edge_pp": round((acc - mr) * 100, 2),
        "balanced_accuracy": float(np.mean(rec)) if rec else 0.0,
        "p_value": float(stats.binomtest(correct, n, mr, alternative="greater").pvalue),
        "beats_baseline": bool(acc > mr and correct > 0
                               and stats.binomtest(correct, n, mr, alternative="greater").pvalue < 0.05),
    }


# ------------------------------------------------------------------ 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=26000)
    ap.add_argument("--train0", type=int, default=14000)
    ap.add_argument("--fold", type=int, default=3000)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--cache", default="data/wf_btc.json")
    ap.add_argument("--out", default="data/walk_forward.json")
    args = ap.parse_args()

    bars = load_bars(args.bars, args.cache)
    folds = []
    t0 = args.train0
    for k in range(args.folds):
        if t0 + args.fold > len(bars):
            break
        folds.append((t0, t0 + args.fold))
        t0 += args.fold
    log(f"数据 {len(bars)} 根; 初始训练段 {args.train0}; fold 数 {len(folds)}; "
        f"每折 {args.fold} 根; 总样本外 {sum(b-a for a,b in folds)} 根")
    for k, (a, b) in enumerate(folds):
        log(f"  fold{k}: 训练 bars[:{a}] -> 评估 ({a}, {b}]")

    results: Dict[str, List[Dict]] = {
        "常量基线": [], "LSTM(17维x60步)": [], "逻辑回归(17维x1)": [],
        f"逻辑回归(top{TOP_K}特征)": [], "梯度提升树(17维x1)": [],
    }
    fold_rows_out = []

    for k, (a, b) in enumerate(folds):
        ctx_start = max(a - WARMUP, 0)
        ctx_df = to_df(bars[ctx_start:b])
        ctx_idx = [r - ctx_start for r in range(a + 1, b)]
        ctx_idx = [i for i in ctx_idx if SEQUENCE_LENGTH <= i < len(ctx_df)]
        closes = ctx_df["close"].to_numpy()
        y = three_class_labels(closes, ctx_idx)

        # ---- LSTM：完整重训（权重/统计量/scaler 都只用训练段）----
        t = time.time()
        p = EnhancedLSTMPredictor(symbol=SCRATCH_SYMBOL)
        p.train(bars[:a], epochs=EPOCHS, batch_size=BATCH, is_fine_tune=False)
        feats = p._compute_features(ctx_df)
        weighted = p.weight_analyzer.get_weighted_features(feats)
        scaled = p.scaler.transform(weighted)
        Xte = np.array([scaled[i - SEQUENCE_LENGTH:i] for i in ctx_idx])
        yte = y
        lstm_pred = p.model.predict(Xte, verbose=0).argmax(axis=1)
        train_secs = time.time() - t

        # ---- 简单模型的训练集：训练段尾部（同样只用训练段）----
        tr_ctx = to_df(bars[max(a - 8000, 0):a])
        tr_idx = list(range(SEQUENCE_LENGTH, len(tr_ctx)))
        tr_closes = tr_ctx["close"].to_numpy()
        ytr = three_class_labels(tr_closes, tr_idx)
        tr_feats = p._compute_features(tr_ctx)
        tr_w = p.weight_analyzer.get_weighted_features(tr_feats)
        tr_scaled = p.scaler.transform(tr_w)
        Xtr = np.array([tr_scaled[i - SEQUENCE_LENGTH:i] for i in tr_idx])
        last_tr, last_te = Xtr[:, -1, :], Xte[:, -1, :]

        lr = LogisticRegression(max_iter=2000).fit(last_tr, ytr)
        lr_pred = lr.predict(last_te)

        mi = mutual_info_classif(last_tr, ytr, random_state=0)
        top = list(np.argsort(mi)[::-1][:TOP_K])
        lr_k = LogisticRegression(max_iter=2000).fit(last_tr[:, top], ytr)
        lr_k_pred = lr_k.predict(last_te[:, top])

        gb = HistGradientBoostingClassifier(random_state=0).fit(last_tr, ytr)
        gb_pred = gb.predict(last_te)

        const = np.full(len(yte), int(np.bincount(yte, minlength=3).argmax()))
        for name, pred in [
            ("常量基线", const),
            ("LSTM(17维x60步)", lstm_pred),
            ("逻辑回归(17维x1)", lr_pred),
            (f"逻辑回归(top{TOP_K}特征)", lr_k_pred),
            ("梯度提升树(17维x1)", gb_pred),
        ]:
            results[name].append(metrics(yte, pred))

        row = {"fold": k, "train_rows": a, "test_rows": len(yte), "train_secs": round(train_secs),
               "top_features": [FEATURE_NAMES[i] for i in top]}
        for name in results:
            row[name] = results[name][-1]["accuracy"]
        row["baseline"] = results["常量基线"][-1]["majority_rate"]
        fold_rows_out.append(row)
        log(f"fold{k} 完成 ({train_secs:.0f}s) | " + " ".join(
            f"{n}={results[n][-1]['accuracy']*100:.1f}%" for n in results))

    # ---- 汇总 ----
    print()
    print("=" * 96)
    print(f"滚动前向验证汇总  ({len(folds)} 折, 共 {sum(r['n'] for r in results['常量基线'])} 个样本外预测点)")
    print("=" * 96)
    print(f"{'模型':<26}{'合并准确率':>10}{'基线':>9}{'优势':>9}{'p值':>12}{'平衡准确率':>11}{'各折占优':>10}")
    summary = {}
    for name in results:
        per = results[name]
        n = sum(r["n"] for r in per)
        acc = sum(r["accuracy"] * r["n"] for r in per) / n
        mr = sum(r["majority_rate"] * r["n"] for r in per) / n
        correct = int(round(acc * n))
        pv = stats.binomtest(correct, n, mr, alternative="greater").pvalue
        bal = sum(r["balanced_accuracy"] * r["n"] for r in per) / n
        wins = sum(1 for r in per if r["accuracy"] > r["majority_rate"])
        summary[name] = {"accuracy": acc, "majority_rate": mr, "edge_pp": round((acc - mr) * 100, 2),
                         "p_value": float(pv), "balanced_accuracy": bal, "n": n,
                         "folds_beating_baseline": wins, "folds": len(per)}
        print(f"{name:<26}{acc*100:9.2f}%{mr*100:8.2f}%{(acc-mr)*100:+8.2f}pp{pv:12.2e}"
              f"{bal*100:10.2f}%{wins:6d}/{len(per)}")

    os.makedirs("data", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"symbol": SYMBOL, "folds": fold_rows_out, "summary": summary},
                  f, indent=2, ensure_ascii=False)
    log(f"结果已写入 {args.out}")

    # 清理中间产物
    for fn in os.listdir(MODEL_SUBDIR):
        if SCRATCH_SYMBOL in fn:
            os.remove(os.path.join(MODEL_SUBDIR, fn))
    log("已清理训练中间产物")


if __name__ == "__main__":
    main()
