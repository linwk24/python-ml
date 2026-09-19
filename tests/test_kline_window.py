"""
取数窗口计算的单元测试

覆盖的回归：``app.main.fetch_kline_data`` 原先把窗口硬编码成
``end - limit * 60 * 60 * 1000``（永远按小时算），配合"Binance 从 startTime 起向后返回
limit 根"的语义，两头都错：

    interval=1m -> 窗口 100 小时，只取到 100 根 1m K 线 -> 数据滞后约 4 天（实测 98.3 小时）
    interval=4h -> 窗口仅覆盖 25 根 -> 少于模型所需，无法预测
    interval=1d -> 窗口仅覆盖 4 根  -> 完全不可用

只用标准库。
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import interval_to_ms, kline_window_ms  # noqa: E402

MINUTE = 60 * 1000
HOUR = 60 * MINUTE
DAY = 24 * HOUR
WEEK = 7 * DAY

# 模型推理所需最小根数（与 models/enhanced_lstm.MIN_KLINES_FOR_PREDICT 对齐）
MIN_KLINES_FOR_PREDICT = 120


class TestIntervalToMs(unittest.TestCase):
    def test_known_intervals(self):
        cases = {
            "1m": MINUTE, "5m": 5 * MINUTE, "15m": 15 * MINUTE, "30m": 30 * MINUTE,
            "1h": HOUR, "4h": 4 * HOUR, "1d": DAY, "1w": WEEK,
        }
        for interval, expected in cases.items():
            with self.subTest(interval=interval):
                self.assertEqual(interval_to_ms(interval), expected)

    def test_rejects_unknown_unit(self):
        with self.assertRaises(ValueError):
            interval_to_ms("1x")


class TestKlineWindow(unittest.TestCase):
    def test_window_grows_with_bars(self):
        self.assertEqual(kline_window_ms("1h", 100), 100 * HOUR)
        self.assertEqual(kline_window_ms("1m", 100), 100 * MINUTE)

    def test_window_is_interval_aware(self):
        """同样的根数，不同周期必须得到不同的时间跨度（这正是原实现丢失的信息）"""
        spans = {iv: kline_window_ms(iv, 100) for iv in ("1m", "5m", "1h", "4h", "1d")}

        self.assertEqual(spans["1m"], 100 * MINUTE)
        self.assertEqual(spans["4h"], 400 * HOUR)
        self.assertEqual(spans["1d"], 100 * DAY)
        self.assertLess(spans["1m"], spans["1h"])
        self.assertLess(spans["1h"], spans["1d"])

    def test_old_hour_hardcoded_window_would_starve_coarse_intervals(self):
        """文档化原缺陷：按小时算的窗口对粗周期只能覆盖很少几根"""
        hours_assumed = 100 * HOUR
        for interval, expected_bars in (("4h", 25), ("1d", 4)):
            with self.subTest(interval=interval):
                self.assertEqual(hours_assumed // interval_to_ms(interval), expected_bars)
                self.assertLess(hours_assumed // interval_to_ms(interval), MIN_KLINES_FOR_PREDICT)

    def test_interval_aware_window_covers_requested_bars_for_every_interval(self):
        """修好之后，每个周期请求 N 根都能覆盖到 N 根的时间跨度"""
        for interval in ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"):
            with self.subTest(interval=interval):
                bars = 300
                self.assertEqual(kline_window_ms(interval, bars) // interval_to_ms(interval), bars)
                self.assertGreaterEqual(bars, MIN_KLINES_FOR_PREDICT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
