"""
信号反转（前置动量校验）的单元测试

覆盖的核心回归：基准价必须来自**本次预测所用的 K 线**，不能是预测记录文件里那条可能
陈旧数月的记录。原实现线上实测复现过"5 月记录价 64266 vs 现价 80624 → +25.45% 应反转"
这种结论。

只用标准库，不依赖 TensorFlow / pandas。
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.signal_reversal import (  # noqa: E402
    check_signal_reversal,
    effective_trend,
)

HOUR_MS = 3600 * 1000
START_MS = 1_700_000_000_000


def bars(closes, interval_ms=HOUR_MS):
    """构造 K 线: [open_time, open, high, low, close, volume]"""
    return [
        [START_MS + i * interval_ms, c, c, c, c, 1.0] for i, c in enumerate(closes)
    ]


class TestSignalReversal(unittest.TestCase):
    def test_returns_none_when_data_insufficient(self):
        self.assertIsNone(check_signal_reversal(None, 2))
        self.assertIsNone(check_signal_reversal([], 2))
        # lookback=1 需要至少 2 根
        self.assertIsNone(check_signal_reversal(bars([100.0]), 2))

    def test_no_trigger_within_threshold(self):
        result = check_signal_reversal(bars([100.0, 100.0]), 0)

        self.assertFalse(result["triggered"])
        self.assertFalse(result["applied"])
        self.assertAlmostEqual(result["price_change"], 0.0)
        self.assertIn("未超过阈值", result["message"])

    def test_triggers_reverse_to_bullish_when_price_rises_against_bearish_model(self):
        result = check_signal_reversal(bars([100.0, 102.0]), current_trend=0)

        self.assertTrue(result["triggered"])
        self.assertEqual(result["reason"], "reverse_to_bullish")
        self.assertEqual(result["corrected_trend_code"], 2)
        self.assertEqual(result["baseline_price"], 100.0)
        self.assertEqual(result["current_price"], 102.0)

    def test_triggers_reverse_to_bearish_when_price_falls_against_bullish_model(self):
        result = check_signal_reversal(bars([100.0, 98.0]), current_trend=2)

        self.assertTrue(result["triggered"])
        self.assertEqual(result["reason"], "reverse_to_bearish")
        self.assertEqual(result["corrected_trend_code"], 0)

    def test_no_trigger_when_price_move_agrees_with_model(self):
        result = check_signal_reversal(bars([100.0, 102.0]), current_trend=2)

        self.assertFalse(result["triggered"])
        self.assertIn("一致", result["message"])

    def test_neutral_model_trend_is_never_reversed(self):
        """中性(1) 没有可反转的方向，不应触发"""
        for move in (102.0, 98.0, 130.0):
            with self.subTest(move=move):
                result = check_signal_reversal(bars([100.0, move]), current_trend=1)
                self.assertFalse(result["triggered"])

    def test_baseline_is_recent_bar_not_a_stale_reference(self):
        """核心回归：历史涨幅不能被当成"当前动量"

        构造：很久以前价格 50，最近 200 根稳定在 100 附近（最后一根 100.05）。
        用"最近 1 根"作基准（默认）时价差只有 +0.05%，不该触发任何反转 ——
        尽管相对远古价格这里已经"涨了 100%"。只有显式把窗口放大到远古那根才会看到它。
        """
        closes = [50.0] + [100.0] * 200 + [100.05]
        klines = bars(closes)

        recent = check_signal_reversal(klines, current_trend=0)
        self.assertFalse(recent["triggered"], "默认只看最近一根，价差不足以触发")
        self.assertEqual(recent["baseline_price"], 100.0, "基准必须是上一根，而不是历史记录价")

        old_window = check_signal_reversal(klines, current_trend=0, lookback_bars=201)
        self.assertTrue(old_window["triggered"], "显式放大窗口才会看到那段远古涨幅")
        self.assertEqual(old_window["baseline_price"], 50.0)
        self.assertGreater(old_window["price_change"], 0.9)

    def test_lookback_bars_is_respected(self):
        klines = bars([100.0, 101.0, 102.0, 103.0])
        result = check_signal_reversal(klines, current_trend=0, lookback_bars=3)

        self.assertEqual(result["lookback_bars"], 3)
        self.assertEqual(result["baseline_price"], 100.0)
        self.assertAlmostEqual(result["price_change"], 0.03)

    def test_input_order_does_not_matter(self):
        klines = bars([100.0, 101.0, 102.0])
        shuffled = [klines[2], klines[0], klines[1]]

        ordered_result = check_signal_reversal(klines, current_trend=0)
        shuffled_result = check_signal_reversal(shuffled, current_trend=0)

        self.assertEqual(ordered_result["baseline_price"], shuffled_result["baseline_price"])
        self.assertEqual(ordered_result["current_price"], shuffled_result["current_price"])

    def test_threshold_is_configurable(self):
        klines = bars([100.0, 100.5])  # +0.5%

        loose = check_signal_reversal(klines, current_trend=0, threshold=0.001)
        strict = check_signal_reversal(klines, current_trend=0, threshold=0.01)

        self.assertTrue(loose["triggered"])
        self.assertFalse(strict["triggered"])

    def test_advisory_by_default_does_not_override_model(self):
        """默认只给建议、不覆盖模型输出（未经回测验证的动量规则不静默生效）"""
        result = check_signal_reversal(bars([100.0, 102.0]), current_trend=0, apply_reversal=False)

        self.assertTrue(result["triggered"])
        self.assertFalse(result["applied"])
        self.assertEqual(result["effective_trend_code"], 0, "未启用时不覆盖模型方向")
        self.assertEqual(result["corrected_trend_code"], 2, "但建议仍然给出")
        self.assertIn("不覆盖模型输出", result["message"])

    def test_applied_when_explicitly_enabled(self):
        result = check_signal_reversal(bars([100.0, 102.0]), current_trend=0, apply_reversal=True)

        self.assertTrue(result["applied"])
        self.assertEqual(result["effective_trend_code"], 2)
        self.assertEqual(result["model_trend_code"], 0)

    def test_price_change_is_numeric_with_formatted_companion(self):
        """原实现混用字符串/数值，这里明确两者都给"""
        result = check_signal_reversal(bars([100.0, 102.0]), current_trend=0)

        self.assertIsInstance(result["price_change"], float)
        self.assertEqual(result["price_change_pct"], "+2.00%")


class TestEffectiveTrend(unittest.TestCase):
    def test_uses_model_trend_when_not_applied(self):
        signal = effective_trend(0, {"applied": False, "effective_trend_code": 2})

        self.assertEqual(signal["trend_code"], 0)
        self.assertEqual(signal["trend"], "看跌")
        self.assertEqual(signal["source"], "model")

    def test_uses_reversal_when_applied(self):
        signal = effective_trend(0, {"applied": True, "effective_trend_code": 2})

        self.assertEqual(signal["trend_code"], 2)
        self.assertEqual(signal["trend"], "看涨")
        self.assertEqual(signal["source"], "signal_reversal")

    def test_handles_missing_reversal(self):
        signal = effective_trend(2, None)

        self.assertEqual(signal["trend_code"], 2)
        self.assertEqual(signal["source"], "model")


if __name__ == "__main__":
    unittest.main(verbosity=2)
