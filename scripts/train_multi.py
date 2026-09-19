"""
多币种批量训练（同一套配方，供"多币种通用方法"使用）

用法::

    .venv/bin/python scripts/train_multi.py --symbols BTCUSDT,ETHUSDT,SOLUSDT
    .venv/bin/python scripts/train_multi.py --symbols BNBUSDT --bars 12000 --epochs 10

每个币种用完全相同的配方：同一周期（1h）、同一根数、同一套特征与标签口径、
同样的时序切分（前 80% 拟合 / 后 20% 验证）。训练完成后逐个打印指标，
并可用 `evaluate_model.py --symbol X --start ... --end ...` 做样本外评估。
"""

import argparse
import json
import os
import sys
import time

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

from config import fetch_klines, FEATURE_WEIGHTING  # noqa: E402
from models.enhanced_lstm import EnhancedLSTMPredictor  # noqa: E402
from services.model_meta import describe_meta, load_train_meta  # noqa: E402


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
    os.makedirs("data", exist_ok=True)
    json.dump(out, open(cache, "w"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help="逗号分隔，如 BTCUSDT,ETHUSDT")
    ap.add_argument("--bars", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    log(f"特征加权模式: {FEATURE_WEIGHTING}")
    summary = []
    for symbol in args.symbols.split(","):
        symbol = symbol.strip()
        cache = f"data/train_{symbol}_1h.json"
        try:
            bars = load_bars(symbol, args.bars, cache)
            if len(bars) < 1000:
                log(f"{symbol}: 数据不足（{len(bars)} 根），跳过")
                continue
            log(f"{symbol}: {len(bars)} 根 1h，开始训练（{args.epochs} epochs）...")
            t0 = time.time()
            predictor = EnhancedLSTMPredictor(symbol=symbol)
            history = predictor.train(bars, epochs=args.epochs, batch_size=args.batch,
                                     is_fine_tune=False)
            hist = history.history
            meta = load_train_meta(symbol) or {}
            log(f"{symbol} 完成 ({time.time()-t0:.0f}s) | "
                f"acc {hist['accuracy'][-1]:.4f} val_acc {hist['val_accuracy'][-1]:.4f} | "
                f"训练序列 {meta.get('train_sequences')} 验证序列 {meta.get('test_sequences')}")
            summary.append({
                "symbol": symbol, "bars": len(bars),
                "train_accuracy": hist["accuracy"][-1],
                "val_accuracy": hist["val_accuracy"][-1],
                "seconds": round(time.time() - t0),
            })
            os.remove(cache)
        except Exception as e:
            log(f"{symbol} 训练失败: {type(e).__name__}: {e}")

    print()
    print("=" * 78)
    print("多币种训练汇总（同一套配方）")
    print("=" * 78)
    for s in summary:
        log(f"  {s['symbol']:<10} {s['bars']:>6} 根 | "
            f"train_acc {s['train_accuracy']:.4f} | val_acc {s['val_accuracy']:.4f} | {s['seconds']}s")
    for s in summary:
        print(f"  {describe_meta(s['symbol'])}")


if __name__ == "__main__":
    main()
