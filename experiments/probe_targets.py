"""
预测目标可学性探针（实验脚本，不属于生产路径）

问题：现有"1h 方向三分类"目标在样本外完全跑不赢常量基线（见 README 口径与风险）。
本脚本用**同一套特征管线**、**同一套样本外方法**，把目标换成别的候选，回答一个问题：

    换目标之后，模型能显著超过「最强常量基线」吗？

判定标准（严格）：
    - 样本外准确率 > 多数类基线，且单侧二项检验 p < 0.05；
    - 同时报告 balanced accuracy（类别不均衡时准确率会被多数类主导，单看准确率会骗人）；
    - 附"打乱标签"对照：打乱后若仍明显高于基线，说明评价管线本身有泄漏。

防泄漏设计：
    - 特征权重 / 标准化统计量只在训练段上拟合；
    - 序列按"标签结束行"划分训练/留出，中间留 H 根缓冲，训练标签绝不越界进留出段；
    - 早停只用训练段内部切出的验证集，留出集从头到尾不参与任何训练决策。

生产代码不受影响：本脚本只调用 `_compute_features` / 权重分析器 / StandardScaler，
不写任何模型产物（不调用 save_model / train）。

用法::

    .venv/bin/python experiments/probe_targets.py            # 全部候选目标
    .venv/bin/python experiments/probe_targets.py --targets dir_h1,vol_regime
"""

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from tensorflow.keras.callbacks import EarlyStopping  # noqa: E402
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input  # noqa: E402
from tensorflow.keras.models import Sequential  # noqa: E402

from config import fetch_klines  # noqa: E402
from models.enhanced_lstm import (  # noqa: E402
    FEATURE_NAMES,
    SEQUENCE_LENGTH,
    TRAIN_LEARNING_RATE,
    EnhancedLSTMPredictor,
)

SYMBOL = "BTCUSDT"
CACHE = "data/probe_btc_1h.json"
TOTAL_BARS = 4000
FIT_RATIO = 0.80
CHUNKS = 4          # 训练段比例（按行）
VAL_RATIO = 0.15          # 训练段内部再切出的验证集比例
EPOCHS = 8
BATCH = 64
BAND = 0.001              # 方向标签阈值 ±0.1%（与生产一致）
COLS = ["open_time", "open", "high", "low", "close", "volume"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------- 数据

def load_bars(total: int = None, cache: str = None) -> List[List[float]]:
    total = total or TOTAL_BARS
    cache = cache or CACHE
    if os.path.exists(cache):
        bars = json.load(open(cache))
        if len(bars) >= total:
            return bars
    log(f"下载 {SYMBOL} 1h 真实 K 线 x{total} ...")
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
    os.makedirs("data", exist_ok=True)
    json.dump(out, open(cache, "w"))
    return out


def to_df(bars) -> pd.DataFrame:
    df = pd.DataFrame(bars, columns=COLS)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["money_flow"] = df["typical_price"] * df["volume"]
    return df.dropna().reset_index(drop=True)


# --------------------------------------------------------------------- 目标定义
# 统一约定（与生产一致）：窗口 = 行 [i-SEQUENCE_LENGTH, i-1]，标签从行 i-1 起算。

def _ret(closes, i, H):
    """行 i-1 -> 行 i-1+H 的收益率"""
    base, fut = closes[i - 1], closes[i - 1 + H]
    return (fut - base) / base


def label_dir_3c(df, i, H):
    ch = _ret(df["close"].to_numpy(), i, H)
    return 2 if ch > BAND else (0 if ch < -BAND else 1)


def label_dir_bin(df, i, H):
    ch = _ret(df["close"].to_numpy(), i, H)
    if abs(ch) <= BAND:
        return None          # 丢弃"平"的样本，任务变成纯方向
    return 1 if ch > 0 else 0


def label_vol_regime(df, i, H=12, lookback_windows=20):
    """未来 H 根已实现波动率 是否高于 过去 lookback_windows 个 H 窗口的中位数"""
    closes = df["close"].to_numpy()
    rets = np.diff(np.log(closes))
    end = i - 1
    if end + H > len(rets) or end - H * lookback_windows < 0:
        return None
    fut_vol = rets[end:end + H].std()
    past = [rets[end - H * (k + 1):end - H * k].std() for k in range(lookback_windows)]
    ref = float(np.median(past))
    return 1 if fut_vol > ref else 0


def label_breakout(df, i, H=12):
    """未来 H 根的最高价 是否突破 前 H 根的最高价"""
    high = df["high"].to_numpy()
    end = i - 1
    if end + H >= len(high) or end - H + 1 < 0:
        return None
    ref_high = high[end - H + 1:end + 1].max()
    fut_max = high[end + 1:end + 1 + H].max()
    return 1 if fut_max > ref_high else 0


TARGETS: Dict[str, dict] = {
    "dir_h1_3c": {"fn": lambda d, i: label_dir_3c(d, i, 1), "n": 3, "H": 1,
                  "desc": "H=1 三分类(涨/平/跌 ±0.1%) [生产现状]"},
    "dir_h1": {"fn": lambda d, i: label_dir_bin(d, i, 1), "n": 2, "H": 1,
               "desc": "H=1 二分类 涨/跌"},
    "dir_h4": {"fn": lambda d, i: label_dir_bin(d, i, 4), "n": 2, "H": 4,
               "desc": "H=4 二分类 涨/跌"},
    "dir_h12": {"fn": lambda d, i: label_dir_bin(d, i, 12), "n": 2, "H": 12,
                "desc": "H=12 二分类 涨/跌"},
    "dir_h24": {"fn": lambda d, i: label_dir_bin(d, i, 24), "n": 2, "H": 24,
                "desc": "H=24 二分类 涨/跌"},
    "vol_regime": {"fn": label_vol_regime, "n": 2, "H": 12,
                   "desc": "H=12 波动率状态(高于历史中位数?)"},
    "breakout": {"fn": label_breakout, "n": 2, "H": 12,
                 "desc": "H=12 突破(未来最高价 > 前12根最高价?)"},
}


# --------------------------------------------------------------------- 特征

def build_features(bars):
    """用生产同款管线算特征；权重与标准化只在训练段拟合"""
    pre = EnhancedLSTMPredictor(symbol="PROBE")   # 仅借用特征管线，不写产物
    df = to_df(bars)
    features = pre._compute_features(df)
    fit_rows = int(len(features) * FIT_RATIO)

    future_returns = pre.compute_future_returns(df)
    train_features = features.iloc[:fit_rows]
    pre.weight_analyzer.update_weights(train_features, future_returns[:fit_rows], FEATURE_NAMES)
    pre.weight_analyzer.fit_normalization(train_features, FEATURE_NAMES)
    weighted = pre.weight_analyzer.get_weighted_features(features)

    scaler = StandardScaler().fit(weighted[:fit_rows])
    return df, scaler.transform(weighted), fit_rows


def build_model(n_classes: int, input_dim: int):
    """与生产同构的 LSTM，只改输出维度；不加 class_weight，让模型直接优化准确率"""
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, input_dim)),
        LSTM(128, return_sequences=True), Dropout(0.3),
        LSTM(64, return_sequences=True), Dropout(0.3),
        LSTM(32, return_sequences=False), Dropout(0.2),
        Dense(32, activation="relu"), Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=__import__("tensorflow").keras.optimizers.Adam(learning_rate=TRAIN_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_sequences(df, scaled, target_fn, H):
    """按'标签结束行 < fit_rows' 划训练、'标签起始行 >= fit_rows' 划留出，中间留 H 根缓冲"""
    n = len(df)
    X, y, rows = [], [], []
    for i in range(SEQUENCE_LENGTH, n - H + 1):
        lab = target_fn(df, i)
        if lab is None:
            continue
        X.append(scaled[i - SEQUENCE_LENGTH:i])
        y.append(lab)
        rows.append(i)
    return np.array(X), np.array(y), np.array(rows)


# --------------------------------------------------------------------- 评估

def evaluate(y_true, y_pred, n_classes, rows=None, H=1):
    """指标；H>1 时多根视界的标签在相邻样本间**重叠**，名义样本数被高估，
    因此额外报告"非重叠子样本"（每 H 根取一个）上的结果 —— 只有它才配得上二项检验。"""
    n = len(y_true)
    counts = np.bincount(y_true, minlength=n_classes)
    maj_class = int(np.argmax(counts))
    maj_rate = counts[maj_class] / n
    correct = int((y_pred == y_true).sum())
    acc = correct / n

    recalls = []
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() > 0:
            recalls.append(float((y_pred[mask] == c).mean()))
    balanced = float(np.mean(recalls)) if recalls else 0.0

    p_value = stats.binomtest(correct, n, maj_rate, alternative="greater").pvalue

    out = {
        "n": n,
        "accuracy": acc,
        "majority_class": maj_class,
        "majority_rate": maj_rate,
        "balanced_accuracy": balanced,
        "class_counts": counts.tolist(),
        "recalls": [round(r, 4) for r in recalls],
        "p_value_vs_majority": float(p_value),
        "beats_baseline": bool(acc > maj_rate and p_value < 0.05),
        "edge_pp": round((acc - maj_rate) * 100, 2),
    }

    if rows is not None and H > 1:
        keep = ((rows - rows[0]) % H) == 0
        if keep.sum() >= 20 and len(np.unique(y_true[keep])) > 1:
            yt, yp = y_true[keep], y_pred[keep]
            m = len(yt)
            c = np.bincount(yt, minlength=n_classes)
            mc = int(np.argmax(c))
            mr = c[mc] / m
            corr = int((yp == yt).sum())
            a = corr / m
            rec = [float((yp[yt == k] == k).mean()) for k in range(n_classes) if (yt == k).sum()]
            out["nonoverlap"] = {
                "n": m,
                "accuracy": a,
                "majority_rate": mr,
                "balanced_accuracy": float(np.mean(rec)) if rec else 0.0,
                "p_value_vs_majority": float(stats.binomtest(corr, m, mr, alternative="greater").pvalue),
                "edge_pp": round((a - mr) * 100, 2),
            }
            out["nonoverlap"]["beats_baseline"] = bool(a > mr and out["nonoverlap"]["p_value_vs_majority"] < 0.05)

    return out


def run_target(name, cfg, df, scaled, fit_rows, shuffle_control=False, seed=0):
    target_fn, n_classes, H = cfg["fn"], cfg["n"], cfg["H"]
    X, y, rows = make_sequences(df, scaled, target_fn, H)
    if len(X) == 0:
        return None

    # 切分：标签结束行 < fit_rows -> 训练；标签起始行 >= fit_rows -> 留出
    label_end = rows - 1 + H
    label_start = rows - 1
    tr = label_end < fit_rows
    ho = label_start >= fit_rows
    X_tr, y_tr = X[tr], y[tr]
    X_ho, y_ho = X[ho], y[ho]

    # 训练段内部再切验证集（早停只用它，绝不碰留出集）
    n_val = max(int(len(X_tr) * VAL_RATIO), BATCH)
    X_fit, y_fit = X_tr[:-n_val], y_tr[:-n_val]
    X_val, y_val = X_tr[-n_val:], y_tr[-n_val:]

    if shuffle_control:
        rng = np.random.default_rng(seed)
        y_fit = rng.permutation(y_fit)   # 只打乱训练标签，留出标签保持真实

    model = build_model(n_classes, X.shape[2])
    model.fit(
        X_fit, y_fit,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
    )
    pred = model.predict(X_ho, verbose=0).argmax(axis=1)

    res = evaluate(y_ho, pred, n_classes, rows=rows[ho], H=H)

    # 分块一致性：把留出段按时间切成 CHUNKS 段，看优势是否只来自某一段（单一行情）
    idx = np.array_split(np.arange(len(y_ho)), CHUNKS)
    res["chunks"] = [
        {
            "n": int(len(ix)),
            "accuracy": round(float((pred[ix] == y_ho[ix]).mean()), 4),
            "majority_rate": round(float(np.bincount(y_ho[ix], minlength=n_classes).max() / len(ix)), 4),
        }
        for ix in idx if len(ix) > 0
    ]
    res["train_seq"] = int(len(X_fit))
    res["holdout_seq"] = int(len(X_ho))
    res["desc"] = cfg["desc"]
    res["H"] = H
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=None, help="逗号分隔的目标名，默认全部")
    ap.add_argument("--shuffle-control", action="store_true", help="附打乱标签对照")
    ap.add_argument("--bars", type=int, default=None, help="使用多少根 K 线（默认 4000）")
    ap.add_argument("--cache", default=None, help="数据缓存文件")
    ap.add_argument("--fit-ratio", type=float, default=None, help="训练段比例（默认 0.80）")
    ap.add_argument("--out", default="data/probe_results.json", help="结果 JSON 路径")
    args = ap.parse_args()

    global FIT_RATIO
    if args.fit_ratio:
        FIT_RATIO = args.fit_ratio

    bars = load_bars(args.bars, args.cache)
    df, scaled, fit_rows = build_features(bars)
    log(f"数据 {len(df)} 行 1h K线; 训练段 {fit_rows} 行 / 留出段 {len(df)-fit_rows} 行; 特征 {scaled.shape[1]} 维")

    names = args.targets.split(",") if args.targets else list(TARGETS)
    results = {}
    for name in names:
        cfg = TARGETS[name]
        t0 = time.time()
        res = run_target(name, cfg, df, scaled, fit_rows)
        results[name] = res
        flag = "✔ 超过基线" if res["beats_baseline"] else "✘ 未超过基线"
        log(f"{name:<12} 留出 {res['holdout_seq']:>4} | 全部样本: 准确率 {res['accuracy']*100:5.2f}% "
            f"vs 多数类 {res['majority_rate']*100:5.2f}% ({res['edge_pp']:+5.2f}pp, p={res['p_value_vs_majority']:.3f}) "
            f"| 平衡 {res['balanced_accuracy']*100:5.2f}% | {flag}")
        if res.get("chunks"):
            ch = " ".join(f"{c['accuracy']*100:.1f}/{c['majority_rate']*100:.1f}" for c in res["chunks"])
            log(f"{'':<12} 分块(准确率/基线): {ch}   <- 应各段都占优，而非单段拉高")
        if "nonoverlap" in res:
            nv = res["nonoverlap"]
            flag2 = "✔ 仍超过" if nv["beats_baseline"] else "✘ 不显著"
            log(f"{'':<12} 非重叠子样本 n={nv['n']:>4}: 准确率 {nv['accuracy']*100:5.2f}% "
                f"vs 多数类 {nv['majority_rate']*100:5.2f}% ({nv['edge_pp']:+5.2f}pp, p={nv['p_value_vs_majority']:.3f}) "
                f"| 平衡 {nv['balanced_accuracy']*100:5.2f}% | {flag2}  [{time.time()-t0:.0f}s]")

    if args.shuffle_control:
        name = "dir_h1_3c"
        t0 = time.time()
        ctrl = run_target(name, TARGETS[name], df, scaled, fit_rows, shuffle_control=True)
        log(f"[对照] 打乱标签后: 准确率 {ctrl['accuracy']*100:5.2f}% vs 基线 "
            f"{ctrl['majority_rate']*100:5.2f}% (差 {ctrl['edge_pp']:+5.2f}pp, p={ctrl['p_value_vs_majority']:.3f}) "
            f"-> 应接近基线，否则管线有泄漏  [{time.time()-t0:.0f}s]")
        results["_shuffle_control_" + name] = ctrl

    os.makedirs("data", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"symbol": SYMBOL, "rows": len(df), "fit_rows": fit_rows,
                   "fit_ratio": FIT_RATIO, "results": results}, f, indent=2, ensure_ascii=False)
    log(f"结果已写入 {args.out}")


if __name__ == "__main__":
    main()
