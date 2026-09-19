"""
模型训练元数据（训练窗口 / 切分 / 指标）

存在的意义：让"这个模型是在哪段数据上训练的"变成可查的事实，从而让离线评估与回测
能够**自动发现 in-sample 评估**。

没有它的时候，``evaluate_model.py`` 和两个回测脚本可以对着模型训练过的同一段行情
跑评估并打印出漂亮但毫无意义的准确率 —— 报出来的数字比实盘好得多，而使用者无从察觉。
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

from config import PROJECT_ROOT

MODEL_SUBDIR = os.path.join(PROJECT_ROOT, "models", "models")


def train_meta_path(symbol: str) -> str:
    """训练元数据文件路径"""
    return os.path.join(MODEL_SUBDIR, f"{symbol}_train_meta.json")


def save_train_meta(symbol: str, meta: Dict[str, Any]) -> str:
    """写入训练元数据（失败不影响训练主流程）"""
    path = train_meta_path(symbol)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = dict(meta)
        payload.setdefault("symbol", symbol)
        payload.setdefault("created_at", datetime.now().isoformat())
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception as e:  # pragma: no cover - 磁盘异常
        print(f"保存训练元数据失败: {e}")
    return path


def load_train_meta(symbol: str) -> Optional[Dict[str, Any]]:
    """读取训练元数据；不存在或损坏时返回 None"""
    path = train_meta_path(symbol)
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def window_overlap(
    start_ms: float, end_ms: float, train_start_ms: float, train_end_ms: float
) -> float:
    """返回两个时间窗的重叠时长（毫秒），不重叠为 0"""
    latest_start = max(start_ms, train_start_ms)
    earliest_end = min(end_ms, train_end_ms)
    return max(0.0, earliest_end - latest_start)


def check_window_overlap(symbol: str, start_ms: float, end_ms: float) -> Dict[str, Any]:
    """检查评估/回测窗口是否与模型训练窗口重叠。

    返回::

        {
          "known": 是否有训练元数据,
          "overlaps": 是否重叠,
          "overlap_days": 重叠天数,
          "train_window": "2025-09-19 ~ 2026-09-19" 或 None,
          "message": 供直接打印的说明,
        }
    """
    meta = load_train_meta(symbol)
    if not meta or meta.get("train_start_ms") is None or meta.get("train_end_ms") is None:
        return {
            "known": False,
            "overlaps": False,
            "overlap_days": 0.0,
            "train_window": None,
            "message": (
                f"模型 {symbol} 缺少训练窗口元数据，无法自动判断评估区间是否 in-sample。"
                f"请自行确认评估区间在训练窗口之外（重新训练一次即可写入该元数据）。"
            ),
        }

    train_start = float(meta["train_start_ms"])
    train_end = float(meta["train_end_ms"])
    overlap_ms = window_overlap(start_ms, end_ms, train_start, train_end)
    overlap_days = overlap_ms / 86400000.0

    fmt = lambda ms: datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")  # noqa: E731
    train_window = f"{fmt(train_start)} ~ {fmt(train_end)}"

    if overlap_ms <= 0:
        message = f"评估窗口与训练窗口（{train_window}）无重叠，属 out-of-sample 结果。"
    else:
        message = (
            f"评估窗口与训练窗口（{train_window}）重叠 {overlap_days:.1f} 天。"
            f"重叠部分的准确率是 in-sample 结果，会明显偏乐观，不能代表实盘表现。"
        )

    return {
        "known": True,
        "overlaps": overlap_ms > 0,
        "overlap_days": overlap_days,
        "train_window": train_window,
        "train_rows": meta.get("rows"),
        "train_metrics": {
            k: meta.get(k) for k in ("val_loss", "val_accuracy", "epochs_run")
        },
        "message": message,
    }


def describe_meta(symbol: str) -> str:
    """一行式描述模型训练信息（用于日志/CLI 输出）"""
    meta = load_train_meta(symbol)
    if not meta:
        return f"{symbol}: 无训练元数据"
    fmt = lambda ms: datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")  # noqa: E731
    parts = [
        f"{symbol}: 训练窗口 {fmt(meta['train_start_ms'])} ~ {fmt(meta['train_end_ms'])}",
        f"{meta.get('rows', '?')} 行",
    ]
    if meta.get("val_accuracy") is not None:
        parts.append(f"val_acc {meta['val_accuracy']:.4f}")
    if meta.get("created_at"):
        parts.append(f"训练于 {meta['created_at'][:19]}")
    return " | ".join(parts)


def now_ms() -> float:
    return time.time() * 1000.0


def guard_in_sample_window(
    symbol: str, start_date: str, end_date: str, allow_in_sample: bool = False
) -> bool:
    """离线评估/回测的统一守卫。

    打印口径结论；若评估窗口与训练窗口重叠且未显式放行，返回 False（调用方应立即中止）。
    这样"对着训练过的行情跑回测、再把结果当实盘预期"这件事不会再悄无声息地发生。
    """
    border = "=" * 70
    try:
        start_ms = datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000
        end_ms = datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000
    except ValueError:
        print(f"[口径检查] 无法解析日期 {start_date} ~ {end_date}，跳过 in-sample 检查。")
        return True

    info = check_window_overlap(symbol, start_ms, end_ms)

    print(border)
    print(f"[口径检查] {describe_meta(symbol)}")
    print(f"[口径检查] {info['message']}")

    if not info["known"]:
        print("[!] 建议重新训练一次以写入训练窗口元数据，此后该检查会自动生效。")
        print(border)
        return True

    if info["overlaps"]:
        if not allow_in_sample:
            print("[!] 已中止。确认要在此区间评估请加 --allow-in-sample；"
                  "结果必须标注为 in-sample，不能当作实盘预期。")
            print(border)
            return False
        print("[!] 已按 --allow-in-sample 继续：以下指标含 in-sample 成分，不可作为实盘预期。")

    print(border)
    return True
