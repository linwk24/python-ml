"""
极端行情把模型输入推到训练分布之外多远（OOD 诊断）

思路：模型对输入做过 StandardScaler（fit 在训练窗口上，mean_/scale_ 存在 scaler.pkl）。
如果某根 K 线的特征 z 分数远超训练时的量级，说明模型见到的是训练中几乎没出现过的输入，
其输出（softmax 方向）不具备训练时那种可信度。

这里不做"模型对不对"的判断，只量化"输入有多陌生"。

用法：.venv/bin/python experiments/shock_ood.py [SYMBOL]
"""

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.enhanced_lstm import EnhancedLSTMPredictor  # noqa: E402


def fetch(symbol, interval="1h", limit=2000, pages=6):
    """复用 app.main 的分页抓取，保证与线上口径一致"""
    from app.main import fetch_kline_data
    return fetch_kline_data(symbol, interval, limit * pages)


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    print(f"[{symbol}] 抓取历史 K 线 ...")
    bars = fetch(symbol)
    # fetch_kline_data 统一输出前 6 列（与线上预测口径一致）
    df = pd.DataFrame(bars, columns=["open_time", "open", "high", "low",
                                     "close", "volume"])
    df = df.astype({"open": float, "high": float, "low": float,
                    "close": float, "volume": float})
    print(f"[{symbol}] 拿到 {len(df)} 根 {symbol} 1h K 线")

    p = EnhancedLSTMPredictor(symbol=symbol)
    p.load_model()
    feats = p._compute_features(df)
    # 必须走 build_input_matrix（生产同一条路径）：_compute_features 给 20 列，
    # 模型实际用 17 列，且标准化用的是训练段拟合的 scaler。
    # 该函数返回的 matrix 已经是 StandardScaler 的输出，即 z 分数本身。
    Z = p.build_input_matrix(feats)
    names = list(p.weight_analyzer.feature_order) or [f"f{j}" for j in range(Z.shape[1])]
    print(f"[{symbol}] 特征维度 {Z.shape[1]}，input_pipeline="
          f"{p.weight_analyzer.input_pipeline}，样本 {Z.shape[0]}")
    col_std = np.nanstd(Z, axis=0)
    print(f"[{symbol}] 模型输入各列 std 中位数 {np.median(col_std):.3f}"
          f"（≈1 说明可按 σ 解读；若远小于 1 则只做相对比较）")

    z = Z
    max_abs_z = np.nanmax(np.abs(z), axis=1)          # 每根 K 线最"陌生"的那个特征
    n_over3 = (np.abs(z) > 3).sum(axis=1)             # 超过 3σ 的特征个数
    n_over5 = (np.abs(z) > 5).sum(axis=1)

    close = df["close"].to_numpy(dtype=float)
    ret = np.zeros(len(close))
    ret[1:] = np.diff(close) / close[:-1]
    valid = ~np.isnan(max_abs_z)
    valid[:1] = False

    shock = valid & (np.abs(ret) >= np.quantile(np.abs(ret[valid]), 0.99))
    normal = valid & ~shock

    def stat(mask):
        return {
            "n": int(mask.sum()),
            "maxz_median": float(np.median(max_abs_z[mask])),
            "maxz_p90": float(np.quantile(max_abs_z[mask], 0.90)),
            "maxz_max": float(max_abs_z[mask].max()),
            "over3_share": float((n_over3[mask] > 0).mean()),
            "over5_share": float((n_over5[mask] > 0).mean()),
            "mean_over3": float(n_over3[mask].mean()),
        }

    s, nm = stat(shock), stat(normal)
    print(f"\n{'':<22}{'普通 K 线':>14}{'极端 1% K 线':>16}")
    print(f"{'样本数':<22}{nm['n']:>14}{s['n']:>16}")
    print(f"{'max|z| 中位数':<22}{nm['maxz_median']:>14.2f}{s['maxz_median']:>16.2f}")
    print(f"{'max|z| 90 分位':<22}{nm['maxz_p90']:>14.2f}{s['maxz_p90']:>16.2f}")
    print(f"{'max|z| 最大':<22}{nm['maxz_max']:>14.2f}{s['maxz_max']:>16.2f}")
    print(f"{'至少 1 个特征 >3σ':<22}{nm['over3_share']:>13.1%}{s['over3_share']:>16.1%}")
    print(f"{'至少 1 个特征 >5σ':<22}{nm['over5_share']:>13.1%}{s['over5_share']:>16.1%}")
    print(f"{'平均超 3σ 特征数':<22}{nm['mean_over3']:>14.2f}{s['mean_over3']:>16.2f}")

    # 找出冲击时最极端的几个特征
    print(f"\n极端 K 线上被推得最远的特征（按 |z| 中位数排序）:")
    order = np.argsort(-np.nanmedian(np.abs(z[shock]), axis=0))[:6]
    for j in order:
        print(f"  {names[j]:<22} 普通 |z| 中位 {np.median(np.abs(z[normal, j])):>6.2f}"
              f"   极端 |z| 中位 {np.median(np.abs(z[shock, j])):>6.2f}"
              f"   极端最大 {np.abs(z[shock, j]).max():>7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
