"""
自我学习闭环的回归测试

只用标准库（unittest），不依赖 TensorFlow / pandas —— 闭环的数据层与编排层
不应该因为模型框架缺失而无法验证。

运行::

    python3 -m unittest discover -s tests -t . -v
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (  # noqa: E402
    LABEL_THRESHOLD,
    SELF_LEARNING_ACCURACY_THRESHOLD,
    SELF_LEARNING_MIN_SAMPLES,
    VERIFY_THRESHOLD,
)
from services.prediction_tracker import (  # noqa: E402
    PredictionTracker,
    infer_horizon_minutes,
)
from services.self_learning_manager import SelfLearningManager  # noqa: E402

HOUR_MS = 3600 * 1000


def make_bars(start_ms, closes, interval_ms=HOUR_MS):
    """构造 K 线行 [open_time, open, high, low, close, volume]"""
    return [
        [start_ms + i * interval_ms, c, c, c, c, 1.0] for i, c in enumerate(closes)
    ]


def build_path(correct_flags, start_close=100.0, move=0.01):
    """构造价格路径: 让第 i 条预测的实际走势符合 correct_flags[i]。

    预测代码交替为 2(看涨)/0(看跌)，从而保证每条记录都会落盘
    （record_prediction 只在方向变化时记录）。
    返回 (codes, closes, bars)。
    """
    codes = [2 if i % 2 == 0 else 0 for i in range(len(correct_flags))]
    closes = [start_close]
    for code, correct in zip(codes, correct_flags):
        want_up = (code == 2) == correct  # 想让实际走势与预测一致 -> 该涨就涨
        closes.append(closes[-1] * (1 + move if want_up else 1 - move))
    return codes, closes, closes  # bars 在测试里用 make_bars 生成


class FakePredictor:
    """假模型：只验证闭环编排逻辑，不加载 TensorFlow"""

    def __init__(self, trend_code=2, confidence=55.0, price=100.0, error=None):
        self.trend_code = trend_code
        self.confidence = confidence
        self.price = price
        self.error = error
        self.train_calls = []

    def predict(self, recent_klines):
        if self.error:
            return {"error": self.error, "symbol": "BTCUSDT"}
        return {
            "symbol": "BTCUSDT",
            "current_price": self.price,
            "prediction": {
                "trend_code": self.trend_code,
                "confidence": self.confidence,
                "trend": "看涨" if self.trend_code == 2 else "看跌",
            },
        }

    def train(self, klines, epochs=100, batch_size=32, is_fine_tune=False):
        self.train_calls.append(
            {"n_klines": len(klines), "is_fine_tune": is_fine_tune, "epochs": epochs}
        )
        return {"loss": [0.5]}


class TrackerTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.symbol = "BTCUSDT"
        self.tracker = PredictionTracker(
            symbol=self.symbol,
            file_path=os.path.join(self._tmp.name, f"predictions_{self.symbol}.json"),
        )


class TestHorizonInference(unittest.TestCase):
    def test_infers_from_hourly_bars(self):
        bars = make_bars(0, [1, 2, 3, 4])
        self.assertEqual(infer_horizon_minutes(bars), 60)

    def test_infers_from_minute_bars(self):
        bars = make_bars(0, [1, 2, 3, 4], interval_ms=60 * 1000)
        self.assertEqual(infer_horizon_minutes(bars), 1)

    def test_defaults_when_unavailable(self):
        self.assertEqual(infer_horizon_minutes(None), 60)
        self.assertEqual(infer_horizon_minutes([]), 60)
        self.assertEqual(infer_horizon_minutes([[0, 1, 1, 1, 1, 1]]), 60)


class TestVerifyUsesRealBarPrice(TrackerTestBase):
    """核心回归：必须用「预测时点 + 视界」之后的真实 K 线价格核对"""

    def test_uses_horizon_bar_not_current_price(self):
        ts0 = int(time.time() * 1000) - 20 * HOUR_MS
        # 预测看涨，但下一根 K 线实际跌了 1%
        bars = make_bars(ts0, [100.0, 99.0, 105.0, 105.0])
        self.tracker.record_prediction(
            self.symbol, 100.0, 2, 55.0, interval="1h", horizon_minutes=60, ts_ms=bars[0][0]
        )

        result = self.tracker.verify_predictions(current_price=105.0, klines=bars)

        entry = self.tracker._read()[0]
        self.assertEqual(entry["actual"], 0, "必须按视界后那根 K 线判定为下跌")
        self.assertEqual(entry["actual_price"], 99.0)
        self.assertEqual(result["verified"], 1)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["failed_samples"][0]["trend_code"], 2)

    def test_correct_prediction_is_counted(self):
        ts0 = int(time.time() * 1000) - 20 * HOUR_MS
        bars = make_bars(ts0, [100.0, 101.0, 101.0])
        self.tracker.record_prediction(
            self.symbol, 100.0, 2, 55.0, interval="1h", horizon_minutes=60, ts_ms=bars[0][0]
        )

        result = self.tracker.verify_predictions(klines=bars)

        self.assertEqual(result["verified"], 1)
        self.assertEqual(result["failed"], 0)
        stats = self.tracker.stats()
        self.assertEqual(stats["correct"], 1)
        self.assertEqual(stats["accuracy"], 1.0)

    def test_pending_before_target_bar_closes(self):
        """目标 K 线存在但尚未收盘时，必须保持 pending（不得偷看未完成行情）"""
        now = int(time.time() * 1000)
        bars = make_bars(now, [100.0, 100.5])  # bars[1] 恰好刚开盘
        self.tracker.record_prediction(
            self.symbol, 100.0, 2, 55.0, interval="1h", horizon_minutes=60, ts_ms=now
        )

        result = self.tracker.verify_predictions(klines=bars)

        entry = self.tracker._read()[0]
        self.assertIsNone(entry["actual"])
        self.assertEqual(entry["status"], "pending")
        self.assertEqual(result["verified"], 0)
        self.assertEqual(result["expired"], 0)

    def test_verifies_once_target_bar_has_closed(self):
        now = int(time.time() * 1000)
        # 目标 K 线在 1 分钟前收盘 -> 可以核对
        target_open = now - HOUR_MS - 60 * 1000
        bars = make_bars(target_open - HOUR_MS, [100.0, 101.0])
        self.tracker.record_prediction(
            self.symbol, 100.0, 2, 55.0, interval="1h", horizon_minutes=60,
            ts_ms=bars[0][0],
        )

        result = self.tracker.verify_predictions(klines=bars)

        self.assertEqual(result["verified"], 1)
        self.assertEqual(self.tracker._read()[0]["actual"], 2)

    def test_expires_when_target_bar_never_closes(self):
        """数据断流：目标 K 线迟迟不收盘，超过宽限期后应过期而不是永远挂起"""
        now = int(time.time() * 1000)
        # 行情在 9.5 小时前断了：中间只剩一根刚开盘、迟迟不收盘的 K 线
        bars = [
            [now - 10 * HOUR_MS, 100.0, 100.0, 100.0, 100.0, 1.0],
            [now - 30 * 60 * 1000, 100.0, 100.0, 100.0, 100.0, 1.0],
        ]
        self.tracker.record_prediction(
            self.symbol, 100.0, 2, 55.0, interval="1h", horizon_minutes=60,
            ts_ms=now - 10 * HOUR_MS,
        )

        result = self.tracker.verify_predictions(klines=bars)

        self.assertEqual(result["expired"], 1)
        self.assertEqual(self.tracker._read()[0]["status"], "expired")

    def test_expired_when_window_no_longer_covers_prediction(self):
        now = int(time.time() * 1000)
        # 预测发生在 100 根 K 线之前，回看窗口已覆盖不到
        self.tracker.record_prediction(
            self.symbol, 100.0, 2, 55.0, interval="1h", horizon_minutes=60,
            ts_ms=now - 100 * HOUR_MS,
        )
        bars = make_bars(now - 10 * HOUR_MS, [100.0] * 11)

        result = self.tracker.verify_predictions(klines=bars)

        entry = self.tracker._read()[0]
        self.assertEqual(entry["status"], "expired")
        self.assertIsNone(entry["actual"], "过期样本不得被填入 actual")
        self.assertEqual(result["expired"], 1)
        stats = self.tracker.stats()
        self.assertEqual(stats["verified"], 0)
        self.assertIsNone(stats["accuracy"])

    def test_verify_lag_is_recorded(self):
        ts0 = int(time.time() * 1000) - 20 * HOUR_MS
        # 提供的 K 线比目标时点晚了 2 小时（周期比预测周期更粗）
        bars = make_bars(ts0, [100.0, 100.0, 100.0, 101.0])
        self.tracker.record_prediction(
            self.symbol, 100.0, 2, 55.0, interval="1h", horizon_minutes=60, ts_ms=bars[0][0]
        )
        bars = bars[0:1] + bars[3:4]  # 只留下 ts0 与 ts0+3h 两根

        self.tracker.verify_predictions(klines=bars)

        entry = self.tracker._read()[0]
        self.assertEqual(entry["status"], "verified")
        self.assertAlmostEqual(entry["verify_lag_minutes"], 120.0, places=1)


class TestThresholds(TrackerTestBase):
    def test_verify_threshold_matches_label_threshold(self):
        self.assertEqual(VERIFY_THRESHOLD, LABEL_THRESHOLD)

    def test_move_within_threshold_is_neutral(self):
        ts0 = int(time.time() * 1000) - 20 * HOUR_MS
        bars = make_bars(ts0, [100.0, 100.0 * (1 + LABEL_THRESHOLD / 2)])
        self.tracker.record_prediction(
            self.symbol, 100.0, 2, 55.0, interval="1h", horizon_minutes=60, ts_ms=bars[0][0]
        )

        self.tracker.verify_predictions(klines=bars)

        self.assertEqual(self.tracker._read()[0]["actual"], 1, "小于阈值应为中性")

    def test_move_beyond_threshold_is_bullish(self):
        ts0 = int(time.time() * 1000) - 20 * HOUR_MS
        bars = make_bars(ts0, [100.0, 100.0 * (1 + LABEL_THRESHOLD * 3)])
        self.tracker.record_prediction(
            self.symbol, 100.0, 2, 55.0, interval="1h", horizon_minutes=60, ts_ms=bars[0][0]
        )

        self.tracker.verify_predictions(klines=bars)

        self.assertEqual(self.tracker._read()[0]["actual"], 2)


class TestRecordDedup(TrackerTestBase):
    def test_records_only_when_trend_changes(self):
        first = self.tracker.record_prediction(self.symbol, 100.0, 2, 55.0)
        second = self.tracker.record_prediction(self.symbol, 101.0, 2, 60.0)
        third = self.tracker.record_prediction(self.symbol, 102.0, 0, 40.0)

        self.assertIsNotNone(first)
        self.assertIsNone(second, "方向未变化不应落盘")
        self.assertIsNotNone(third)
        self.assertEqual(len(self.tracker._read()), 2)

    def test_new_record_carries_verification_metadata(self):
        entry = self.tracker.record_prediction(
            self.symbol, 100.0, 2, 55.0, interval="1h", horizon_minutes=60
        )
        self.assertEqual(entry["status"], "pending")
        self.assertEqual(entry["horizon_minutes"], 60)
        self.assertEqual(entry["interval"], "1h")
        self.assertIsNone(entry["actual"])
        self.assertIn("ts_ms", entry)


class TestLegacyRecords(TrackerTestBase):
    """历史记录（无 ts_ms / 无 horizon / 无 status）必须仍可核对"""

    def _legacy_entry(self, ts_ms, price=100.0, trend_code=2):
        import datetime

        return {
            "timestamp": datetime.datetime.fromtimestamp(ts_ms / 1000).isoformat(),
            "symbol": self.symbol,
            "price": price,
            "trend_code": trend_code,
            "confidence": 55.0,
            "actual": None,
        }

    def test_legacy_record_without_new_fields(self):
        ts0 = int(time.time() * 1000) - 20 * HOUR_MS
        bars = make_bars(ts0, [100.0, 101.0])
        self.tracker._write([self._legacy_entry(ts0)])

        result = self.tracker.verify_predictions(klines=bars)

        entry = self.tracker._read()[0]
        self.assertEqual(result["verified"], 1)
        self.assertEqual(entry["actual"], 2, "兼容按 timestamp 解析的旧记录")
        self.assertEqual(entry["status"], "verified")

    def test_legacy_pending_record_gets_status_and_is_counted(self):
        now = int(time.time() * 1000)
        bars = make_bars(now, [100.0, 100.5])  # 目标 K 线尚未收盘
        self.tracker._write([self._legacy_entry(now)])

        self.tracker.verify_predictions(klines=bars)

        entry = self.tracker._read()[0]
        self.assertEqual(entry["status"], "pending", "历史记录应被补齐 status")
        self.assertIsNone(entry["actual"])
        stats = self.tracker.stats()
        self.assertEqual(stats["pending"], 1, "待核对记录必须计入 pending")
        self.assertEqual(stats["verified"], 0)
        self.assertIsNone(stats["accuracy"])


class TestSelfLearningCycle(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.symbol = "BTCUSDT"
        self.tracker = PredictionTracker(
            symbol=self.symbol,
            file_path=os.path.join(self._tmp.name, f"predictions_{self.symbol}.json"),
        )
        self.predictor = FakePredictor()

    def make_manager(self):
        return SelfLearningManager(
            symbol=self.symbol, predictor=self.predictor, tracker=self.tracker
        )

    def seed_scenario(self, flags):
        """落盘前 n-1 条并核对，最后一条留待本轮核对。

        时间轴刻意整体后移一根 K 线：最后一条预测的目标 K 线必须已经收盘，
        否则它会（正确地）保持 pending。
        返回本轮巡检用的 klines。
        """
        start_ms = int(time.time() * 1000) - (len(flags) + 1) * HOUR_MS
        codes, closes, _ = build_path(flags)
        bars = make_bars(start_ms, closes)

        for i, code in enumerate(codes[:-1]):
            self.tracker.record_prediction(
                self.symbol, closes[i], code, 55.0, interval="1h",
                horizon_minutes=60, ts_ms=bars[i][0],
            )
        self.tracker.verify_predictions(klines=bars)

        last = len(flags) - 1
        self.tracker.record_prediction(
            self.symbol, closes[last], codes[last], 55.0, interval="1h",
            horizon_minutes=60, ts_ms=bars[last][0],
        )
        return bars

    def test_cycle_runs_end_to_end_and_learns_when_accuracy_low(self):
        flags = [False] * 12  # 全部预测错误
        bars = self.seed_scenario(flags)

        result = self.make_manager().run_self_learning_cycle(bars)

        self.assertEqual(result["status"], "learned")
        self.assertEqual(len(self.predictor.train_calls), 1)
        self.assertTrue(self.predictor.train_calls[0]["is_fine_tune"])
        self.assertEqual(result["stats"]["newly_verified"], 1)
        self.assertEqual(result["stats"]["accuracy"], "0.00%")

    def test_cycle_is_stable_when_accuracy_is_high(self):
        flags = [True] * 12
        bars = self.seed_scenario(flags)

        result = self.make_manager().run_self_learning_cycle(bars)

        self.assertEqual(result["status"], "stable")
        self.assertEqual(self.predictor.train_calls, [])
        self.assertGreaterEqual(result["accuracy"], SELF_LEARNING_ACCURACY_THRESHOLD)

    def test_cycle_skips_when_not_enough_verified_samples(self):
        flags = [False] * 2
        bars = self.seed_scenario(flags)

        result = self.make_manager().run_self_learning_cycle(bars)

        self.assertEqual(result["status"], "skipped")
        self.assertIn("Insufficient", result["message"])
        self.assertEqual(self.predictor.train_calls, [])
        self.assertLess(result["stats"]["verified_count"], SELF_LEARNING_MIN_SAMPLES)

    def test_cycle_does_not_retrain_without_new_evidence(self):
        flags = [False] * 12
        bars = self.seed_scenario(flags)
        manager = self.make_manager()

        first = manager.run_self_learning_cycle(bars)
        second = manager.run_self_learning_cycle(bars)

        self.assertEqual(first["status"], "learned")
        self.assertEqual(second["status"], "skipped")
        self.assertIn("No newly verified", second["message"])
        self.assertEqual(len(self.predictor.train_calls), 1, "无新样本时不得重复微调")

    def test_cycle_reports_error_when_training_fails(self):
        class BrokenPredictor(FakePredictor):
            def train(self, *a, **kw):
                raise RuntimeError("boom")

        flags = [False] * 12
        bars = self.seed_scenario(flags)
        manager = SelfLearningManager(
            symbol=self.symbol, predictor=BrokenPredictor(), tracker=self.tracker
        )

        result = manager.run_self_learning_cycle(bars)

        self.assertEqual(result["status"], "error")
        self.assertIn("boom", result["message"])

    def test_wrap_predict_records_with_inferred_horizon(self):
        manager = self.make_manager()
        bars = make_bars(int(time.time() * 1000) - 3 * HOUR_MS, [100.0] * 4)
        manager.predictor.price = 100.0

        manager.wrap_predict(bars, interval="1h")

        entry = self.tracker._read()[0]
        self.assertEqual(entry["horizon_minutes"], 60)
        self.assertEqual(entry["interval"], "1h")
        self.assertEqual(entry["price"], 100.0)

    def test_wrap_predict_does_not_record_on_error(self):
        manager = SelfLearningManager(
            symbol=self.symbol, predictor=FakePredictor(error="模型未训练"), tracker=self.tracker
        )
        bars = make_bars(int(time.time() * 1000) - 3 * HOUR_MS, [100.0] * 4)

        manager.wrap_predict(bars, interval="1h")

        self.assertEqual(self.tracker._read(), [])


class TestImportability(unittest.TestCase):
    """闭环模块必须能在没有 TensorFlow、且工作目录任意的情况下导入"""

    def test_manager_imports_with_tensorflow_unavailable(self):
        """把 tensorflow 置为不可导入后，闭环模块仍须能导入且不加载模型模块（延迟导入）"""
        code = (
            "import sys\n"
            "sys.modules['tensorflow'] = None  # 令 import tensorflow 直接失败\n"
            f"sys.path.insert(0, {PROJECT_ROOT!r})\n"
            "import services.self_learning_manager\n"
            "assert 'models.enhanced_lstm' not in sys.modules, '闭环模块不应在导入期加载模型'\n"
            "print('ok')\n"
        )
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ok", proc.stdout)

    def test_import_from_unrelated_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, PYTHONPATH=PROJECT_ROOT, PYTHONDONTWRITEBYTECODE="1")
            proc = subprocess.run(
                [sys.executable, "-c", "import services.self_learning_manager; print('ok')"],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ok", proc.stdout)

    def test_log_dir_created_on_import(self):
        from config import LOG_DIR

        self.assertTrue(os.path.isdir(LOG_DIR))


class TestAtomicWrite(TrackerTestBase):
    def test_no_temp_file_left_behind(self):
        self.tracker.record_prediction(self.symbol, 100.0, 2, 55.0)
        leftovers = [f for f in os.listdir(self._tmp.name) if f.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_corrupt_file_is_quarantined(self):
        with open(self.tracker.file_path, "w") as f:
            f.write("{not json")
        self.assertEqual(self.tracker._read(), [])
        backups = [f for f in os.listdir(self._tmp.name) if ".corrupt-" in f]
        self.assertEqual(len(backups), 1)

    def test_records_are_valid_json_after_many_writes(self):
        for i in range(20):
            self.tracker.record_prediction(self.symbol, 100.0 + i, i % 3, 50.0)
        with open(self.tracker.file_path) as f:
            data = json.load(f)
        self.assertEqual(len(data), 20)
        self.assertEqual(len(self.tracker._read()), 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
