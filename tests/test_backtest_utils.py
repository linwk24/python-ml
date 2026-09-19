"""
回测口径工具的单元测试

被钉住的核心事实：模型的输入窗口是 df[i-lookback:i]（最后一根 i-1），而训练标签
（prepare_data，已实测确认）取的是"行 i -> 行 i+1"这一根的涨跌。
评价基准必须是 df.iloc[i]，不是窗口最后一根 df.iloc[i-1] ——
用错基准会把跨度变成 2 根 K 线，并让相邻样本重叠，实测滞后自相关被人为抬到 +0.51。

需要 pandas（不需要 TensorFlow）。
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.backtest_utils import (  # noqa: E402  (纯函数，无第三方依赖)
    flat_band_share,
    is_prediction_correct,
    next_bar_label,
)

try:
    import pandas as pd

    HAS_PANDAS = True
    SKIP_REASON = ""
except Exception as e:  # pragma: no cover
    pd = None
    HAS_PANDAS = False
    SKIP_REASON = f"pandas 不可用: {e}"


def frame(closes):
    return pd.DataFrame(
        {
            "open_time": [i * 3600000 for i in range(len(closes))],
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1.0] * len(closes),
        }
    )


@unittest.skipUnless(HAS_PANDAS, SKIP_REASON)
class TestNextBarLabel(unittest.TestCase):
    """窗口 = df[i-lookback:i]（最后一根 i-1），标签 = 行 i-1 -> 行 i"""

    # 构造让 (i-2 -> i-1) 与 (i-1 -> i) 方向相反，能区分基准取哪一根
    CLOSES = [100.0, 100.0, 100.0, 101.0, 99.0, 99.0, 99.0, 99.0]

    def test_base_is_the_windows_last_bar(self):
        df = frame(self.CLOSES)

        base, change, trend = next_bar_label(df, 4)

        self.assertEqual(base, 101.0, "基准必须是窗口最后一根 df.iloc[i-1]")
        self.assertAlmostEqual(change, (99.0 - 101.0) / 101.0 * 100)
        self.assertEqual(trend, "跌")

    def test_old_gapped_convention_would_give_a_different_bar(self):
        """反证：旧的"隔一根"口径取 df.iloc[i] -> df.iloc[i+1]，是另一根 K 线"""
        df = frame(self.CLOSES)

        _, new_change, new_trend = next_bar_label(df, 4)          # i-1 -> i
        old_change = (99.0 - 99.0) / 99.0 * 100                    # i -> i+1（均为 99）
        self.assertEqual(new_trend, "跌")
        self.assertAlmostEqual(old_change, 0.0)
        self.assertNotAlmostEqual(new_change, old_change, places=3)

    def test_base_price_equals_window_last_close(self):
        """基准价必须等于预测时可见的最后一根收盘价（线上闭环同口径）"""
        df = frame(self.CLOSES)

        base, _, _ = next_bar_label(df, 4)

        self.assertEqual(base, float(df.iloc[3]["close"]))

    def test_threshold_band_marks_flat(self):
        df = frame([100.0, 100.05])

        _, change, trend = next_bar_label(df, 1)

        self.assertAlmostEqual(change, 0.05)
        self.assertEqual(trend, "平", "0.05% 未超过 ±0.1% 阈值")

    def test_rise_beyond_threshold(self):
        df = frame([100.0, 100.5])

        _, _, trend = next_bar_label(df, 1)

        self.assertEqual(trend, "涨")

    def test_custom_threshold(self):
        df = frame([100.0, 101.0])

        _, _, trend = next_bar_label(df, 1, threshold=0.02)

        self.assertEqual(trend, "平")

    def test_out_of_range_index_raises(self):
        df = frame([100.0, 101.0])

        for bad in (0, -1, 2, 5):
            with self.subTest(i=bad):
                with self.assertRaises(IndexError):
                    next_bar_label(df, bad)


class TestCorrectness(unittest.TestCase):
    def test_three_class_matching(self):
        self.assertTrue(is_prediction_correct("看涨", "涨"))
        self.assertTrue(is_prediction_correct("看跌", "跌"))
        self.assertTrue(is_prediction_correct("中性", "平"))
        self.assertFalse(is_prediction_correct("看涨", "跌"))
        self.assertFalse(is_prediction_correct("中性", "涨"))

    def test_none_actual_is_not_scored(self):
        self.assertIsNone(is_prediction_correct("看涨", None))

    def test_unknown_trend_is_not_scored(self):
        self.assertIsNone(is_prediction_correct("横盘", "平"))


@unittest.skipUnless(HAS_PANDAS, SKIP_REASON)
class TestFlatBandShare(unittest.TestCase):
    def test_share_within_neutral_band(self):
        changes = [0.0, 0.05, -0.09, 0.2, -0.5, 1.0]

        self.assertAlmostEqual(flat_band_share(changes), 3 / 6)

    def test_ignores_missing(self):
        self.assertAlmostEqual(flat_band_share([0.0, None, 5.0]), 0.5)

    def test_empty(self):
        self.assertEqual(flat_band_share([]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
