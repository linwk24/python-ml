"""
预测记录与核对模块（自我学习闭环的数据层）

单条记录格式::

    {
        "timestamp": "2026-09-19T01:00:00",  # 预测时间（本地时间，兼容旧消费方）
        "ts_ms": 1789743600000,              # 预测时间 epoch 毫秒（核对用，避免时区歧义）
        "symbol": "BTCUSDT",
        "price": 80000.0,                    # 预测时的入场价
        "trend_code": 2,                     # 0 看跌 / 1 中性 / 2 看涨
        "confidence": 55.0,
        "interval": "1h",                    # 预测所用 K 线周期（可为 null）
        "horizon_minutes": 60,               # 预测视界：未来一根 K 线的时长
        "actual": null,                      # 核对后的真实标签 0/1/2
        "actual_price": null,                # 核对所用的真实价格
        "actual_change": null,               # 真实涨跌幅
        "actual_bar_ms": null,               # 真实价格所属 K 线的开盘时间
        "verify_lag_minutes": null,          # 核对时点相对目标时点的滞后
        "verified_at": null,
        "status": "pending"                  # pending / verified / expired
    }

核对三原则（本模块存在的意义）:

1. **不用当前价审判陈旧预测**：核对价必须取「预测时点 + 视界」之后的真实 K 线收盘价，
   而不是巡检时刻的价格。否则一条 3 个月前的预测会被拿今天的价格判定对错。
2. **不可核对的标记 expired**：回看窗口已不覆盖目标 K 线、或 K 线长时间缺失时，
   标记为 expired 并排除在准确率统计之外，保证统计口径诚实。
3. **未到核对时点保持 pending**：预测后还没走完一根 K 线时不做任何判定。

阈值统一来自 ``config``（LABEL_THRESHOLD / VERIFY_THRESHOLD / VERIFY_MAX_AGE_BARS），
禁止在本文件内再写一套魔法数字 —— 训练标签与核对阈值必须一致，否则准确率不可比。
"""

import json
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config import (
    VERIFY_MAX_AGE_BARS,
    VERIFY_THRESHOLD,
    prediction_file_path,
)

# 预测记录文件的读改写需要互斥（多线程/多请求并发时防止互相覆盖）
_FILE_LOCK = threading.RLock()

DEFAULT_HORIZON_MINUTES = 60  # 无法从 K 线推断时的兜底视界


def _bar_open_ms(row: Sequence[Any]) -> Optional[float]:
    """从 K 线行中取出开盘时间（毫秒）。兼容 6 列 / 7 列、字符串或数字。"""
    try:
        return float(row[0])
    except (TypeError, ValueError, IndexError):
        return None


def _bar_close(row: Sequence[Any]) -> Optional[float]:
    """从 K 线行中取出收盘价。统一格式为 [open_time, open, high, low, close, volume, ...]"""
    try:
        return float(row[4])
    except (TypeError, ValueError, IndexError):
        return None


def infer_horizon_minutes(klines: Optional[List[Sequence[Any]]]) -> int:
    """从 K 线时间戳推断单根 K 线的时长（分钟），即预测视界的分钟数。

    取相邻开盘时间差的中位数，对缺失/乱序 K 线不敏感。无法推断时返回兜底值。
    """
    if not klines or len(klines) < 3:
        return DEFAULT_HORIZON_MINUTES

    stamps = sorted({ms for ms in (_bar_open_ms(r) for r in klines) if ms is not None})
    if len(stamps) < 3:
        return DEFAULT_HORIZON_MINUTES

    diffs = sorted(b - a for a, b in zip(stamps, stamps[1:]) if b > a)
    if not diffs:
        return DEFAULT_HORIZON_MINUTES

    median_ms = diffs[len(diffs) // 2]
    minutes = int(round(median_ms / 60000.0))
    return minutes if minutes > 0 else DEFAULT_HORIZON_MINUTES


def _record_ms(entry: Dict[str, Any]) -> Optional[float]:
    """取预测时点的 epoch 毫秒。优先 ts_ms，其次解析 timestamp（兼容历史记录）。"""
    ts_ms = entry.get("ts_ms")
    if ts_ms is not None:
        try:
            return float(ts_ms)
        except (TypeError, ValueError):
            pass

    raw = entry.get("timestamp")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).timestamp() * 1000.0
    except (TypeError, ValueError):
        return None


def label_from_change(change: float, threshold: float = VERIFY_THRESHOLD) -> int:
    """把涨跌幅映射为标签：2 看涨 / 1 中性 / 0 看跌（与训练标签口径一致）"""
    if change > threshold:
        return 2
    if change < -threshold:
        return 0
    return 1


class PredictionTracker:
    """预测记录器：落盘、按视界核对、统计。"""

    def __init__(self, symbol: str = "BTCUSDT", file_path: Optional[str] = None):
        self.symbol = symbol
        # 默认锚定到项目根目录，避免因工作目录不同而读写到不同文件
        self.file_path = file_path or prediction_file_path(symbol)
        self._init_file()

    # ------------------------------------------------------------------ 存储

    def _init_file(self) -> None:
        with _FILE_LOCK:
            if not os.path.exists(self.file_path):
                os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
                self._write([])

    def _read(self) -> List[Dict[str, Any]]:
        with _FILE_LOCK:
            try:
                with open(self.file_path, "r") as f:
                    data = json.load(f)
            except FileNotFoundError:
                return []
            except (json.JSONDecodeError, ValueError):
                # 文件损坏时先留档再重来，避免直接覆盖掉可能还有价值的数据
                backup = f"{self.file_path}.corrupt-{int(time.time())}"
                try:
                    os.replace(self.file_path, backup)
                    print(f"预测记录文件损坏，已备份到 {backup}")
                except OSError:
                    pass
                return []

            if not isinstance(data, list):
                return []
            return [e for e in data if isinstance(e, dict)]

    def _write(self, data: List[Dict[str, Any]]) -> None:
        """原子写：先写临时文件再 rename，避免中断时留下半个 JSON。"""
        with _FILE_LOCK:
            tmp_path = f"{self.file_path}.tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.file_path)

    # ------------------------------------------------------------------ 记录

    def record_prediction(
        self,
        symbol: str,
        current_price: float,
        trend_code: int,
        confidence: float,
        interval: Optional[str] = None,
        horizon_minutes: Optional[int] = None,
        ts_ms: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """记录一次预测。

        仅在 ``trend_code`` 发生变化时落盘（方向未变不重复记录）。
        返回新写入的记录；方向未变化时返回 None。
        """
        now = datetime.now()
        entry = {
            "timestamp": now.isoformat(),
            "ts_ms": float(ts_ms) if ts_ms is not None else now.timestamp() * 1000.0,
            "symbol": symbol,
            "price": float(current_price),
            "trend_code": int(trend_code),
            "confidence": float(confidence),
            "interval": interval,
            "horizon_minutes": int(horizon_minutes or DEFAULT_HORIZON_MINUTES),
            "actual": None,
            "actual_price": None,
            "actual_change": None,
            "actual_bar_ms": None,
            "verify_lag_minutes": None,
            "verified_at": None,
            "status": "pending",
        }

        with _FILE_LOCK:
            data = self._read()
            if data and data[-1].get("trend_code") == entry["trend_code"]:
                return None  # 趋势未变化，不记录快照
            data.append(entry)
            self._write(data)

        return entry

    # ------------------------------------------------------------------ 核对

    @staticmethod
    def _prepare_bars(klines: Optional[List[Sequence[Any]]]) -> List[Tuple[float, float]]:
        """把 K 线整理为按时间升序的 (开盘时间ms, 收盘价) 列表。"""
        if not klines:
            return []
        bars = []
        for row in klines:
            ms = _bar_open_ms(row)
            close = _bar_close(row)
            if ms is None or close is None:
                continue
            bars.append((ms, close))
        bars.sort(key=lambda b: b[0])
        return bars

    def verify_predictions(
        self,
        current_price: Optional[float] = None,
        klines: Optional[List[Sequence[Any]]] = None,
    ) -> Dict[str, Any]:
        """核对所有待核对的预测，返回本轮核对结果。

        参数:
            klines: 当前可用的 K 线（形如 [open_time, open, high, low, close, volume]）。
                    推荐传入 —— 只有它才能取到「预测时点 + 视界」之后的真实价格。
            current_price: 仅在拿不到 K 线时作为兜底近似，且受 VERIFY_MAX_AGE_BARS 限制。

        返回::

            {
              "verified": 本轮新核对数量,
              "failed": 本轮新核对中预测错误的数量,
              "expired": 本轮新标记过期的数量,
              "failed_samples": [预测错误的记录, ...]   # 供微调使用
            }
        """
        bars = self._prepare_bars(klines)
        now_ms = time.time() * 1000.0

        result: Dict[str, Any] = {"verified": 0, "failed": 0, "expired": 0, "failed_samples": []}

        with _FILE_LOCK:
            data = self._read()
            changed = False

            for entry in data:
                if entry.get("status") == "expired" or entry.get("actual") is not None:
                    continue

                rec_ms = _record_ms(entry)
                entry_price = entry.get("price")
                if rec_ms is None or not entry_price:
                    entry["status"] = "expired"
                    result["expired"] += 1
                    changed = True
                    continue

                horizon_min = int(entry.get("horizon_minutes") or DEFAULT_HORIZON_MINUTES)
                target_ms = rec_ms + horizon_min * 60000.0

                # 历史记录没有 status 字段，顺手补齐，避免统计口径含糊
                if entry.get("status") is None:
                    entry["status"] = "pending"
                    changed = True

                actual_price: Optional[float] = None
                actual_bar_ms: Optional[float] = None
                lag_minutes: Optional[float] = None

                if bars:
                    # 目标 K 线必须落在回看窗口内，否则该预测已无法诚实核对
                    if target_ms < bars[0][0]:
                        entry["status"] = "expired"
                        result["expired"] += 1
                        changed = True
                        continue

                    candidate = next(((ms, close) for ms, close in bars if ms >= target_ms), None)
                    waited_min = (now_ms - target_ms) / 60000.0

                    # 只使用「已经收盘」的 K 线：刚开盘那根的收盘价还在变，
                    # 用它核对等于偷看未完成的行情。
                    bar_closed = (
                        candidate is not None
                        and (candidate[0] + horizon_min * 60000.0) <= now_ms
                    )
                    if not bar_closed:
                        # 目标 K 线还没出现或还没走完；久等不到（数据缺口）则过期
                        if waited_min > horizon_min * (1 + VERIFY_MAX_AGE_BARS):
                            entry["status"] = "expired"
                            result["expired"] += 1
                            changed = True
                        continue

                    actual_bar_ms, actual_price = candidate
                    lag_minutes = (actual_bar_ms - target_ms) / 60000.0
                else:
                    # 兜底路径：用当前价近似，但绝不允许无限期滞后
                    if current_price is None:
                        continue
                    elapsed_min = (now_ms - target_ms) / 60000.0
                    if elapsed_min < 0:
                        continue  # 还没到核对时点
                    if elapsed_min > horizon_min * VERIFY_MAX_AGE_BARS:
                        entry["status"] = "expired"
                        result["expired"] += 1
                        changed = True
                        continue
                    actual_price = float(current_price)
                    lag_minutes = elapsed_min

                change = (actual_price - float(entry_price)) / float(entry_price)
                actual = label_from_change(change)

                entry["actual"] = actual
                entry["actual_price"] = actual_price
                entry["actual_change"] = change
                entry["actual_bar_ms"] = actual_bar_ms
                entry["verify_lag_minutes"] = lag_minutes
                entry["verified_at"] = datetime.now().isoformat()
                entry["status"] = "verified"

                result["verified"] += 1
                changed = True
                if actual != entry.get("trend_code"):
                    result["failed"] += 1
                    result["failed_samples"].append(entry)

            if changed:
                self._write(data)

        return result

    # ------------------------------------------------------------------ 统计

    def _verified_entries(self) -> List[Dict[str, Any]]:
        entries = [e for e in self._read() if e.get("actual") is not None]
        entries.sort(key=lambda e: (_record_ms(e) or 0.0))
        return entries

    def stats(self) -> Dict[str, Any]:
        """汇总预测记录统计。准确率只基于已核对(verified)样本计算。"""
        data = self._read()
        entries = [e for e in data if e.get("actual") is not None]

        correct = sum(1 for e in entries if e.get("trend_code") == e.get("actual"))
        verified = len(entries)
        incorrect = verified - correct
        accuracy = (correct / verified) if verified else None

        lags = [
            float(e["verify_lag_minutes"])
            for e in entries
            if e.get("verify_lag_minutes") is not None
        ]

        return {
            "total": len(data),
            "verified": verified,
            "correct": correct,
            "incorrect": incorrect,
            # 待核对 = 尚未有核对结论且未过期（历史记录可能没有 status 字段）
            "pending": sum(
                1 for e in data if e.get("actual") is None and e.get("status") != "expired"
            ),
            "expired": sum(1 for e in data if e.get("status") == "expired"),
            "accuracy": accuracy,
            "accuracy_pct": f"{accuracy:.2%}" if accuracy is not None else None,
            "avg_verify_lag_minutes": (sum(lags) / len(lags)) if lags else None,
        }

    def recent_verified(self, limit: int = 10) -> List[Dict[str, Any]]:
        """最近 limit 条已核对样本（按预测时间排序）。"""
        entries = self._verified_entries()
        return entries[-limit:] if limit > 0 else entries

    def recent_accuracy(self, limit: int = 10) -> Optional[float]:
        """最近 limit 条已核对样本的准确率；样本为空时返回 None。"""
        entries = self.recent_verified(limit)
        if not entries:
            return None
        correct = sum(1 for e in entries if e.get("trend_code") == e.get("actual"))
        return correct / len(entries)
