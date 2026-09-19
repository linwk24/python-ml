"""
未收盘 K 线剔除的单元测试

训练只见过完整 K 线（收盘后入库），而取数接口会把当前正在走的那根一并返回 ——
它的 close/high/low/volume 都还在变。拿它做推理会让输入的最后一step与训练分布不同，
模型的标签定义（"窗口最后一根已收盘 → 下一根"）也不成立。

只用标准库。
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import drop_unclosed_klines, interval_to_ms  # noqa: E402

HOUR = interval_to_ms("1h")
BASE = 1_700_000_000_000


def bars(count, interval_ms=HOUR, start=BASE):
    return [[start + i * interval_ms, 100.0, 100.0, 100.0, 100.0, 1.0] for i in range(count)]


class TestDropUnclosedKlines(unittest.TestCase):
    def test_drops_forming_last_bar(self):
        """最后一根还没收盘（开盘+周期 > now）-> 丢掉"""
        k = bars(5)
        now = k[-1][0] + HOUR // 2  # 最后一根才走了半根

        result, dropped = drop_unclosed_klines(k, HOUR, now_ms=now)

        self.assertTrue(dropped)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[-1][0], k[-2][0])

    def test_keeps_last_bar_when_closed(self):
        k = bars(5)
        now = k[-1][0] + HOUR  # 最后一根刚好收盘

        result, dropped = drop_unclosed_klines(k, HOUR, now_ms=now)

        self.assertFalse(dropped)
        self.assertEqual(len(result), 5)

    def test_keeps_when_now_is_well_after_close(self):
        k = bars(3)
        now = k[-1][0] + 10 * HOUR

        result, dropped = drop_unclosed_klines(k, HOUR, now_ms=now)

        self.assertFalse(dropped)
        self.assertEqual(len(result), 3)

    def test_interval_aware(self):
        """周期越细，越容易把最后一根判为未收盘"""
        k = bars(3, interval_ms=interval_to_ms("1m"))
        now = k[-1][0] + interval_to_ms("1m") // 2

        _, dropped = drop_unclosed_klines(k, interval_to_ms("1m"), now_ms=now)
        self.assertTrue(dropped)

        _, dropped_hour = drop_unclosed_klines(k, interval_to_ms("1m"), now_ms=k[-1][0] + 2 * HOUR)
        self.assertFalse(dropped_hour)

    def test_empty_input(self):
        result, dropped = drop_unclosed_klines([], HOUR, now_ms=BASE)
        self.assertEqual(result, [])
        self.assertFalse(dropped)

    def test_unparsable_timestamp_is_kept(self):
        """时间戳解析失败时保守不丢，避免误删数据"""
        k = [["not-a-number", 100.0, 100.0, 100.0, 100.0, 1.0]]

        result, dropped = drop_unclosed_klines(k, HOUR, now_ms=BASE)

        self.assertFalse(dropped)
        self.assertEqual(len(result), 1)

    def test_single_bar_can_be_dropped_to_empty(self):
        k = bars(1)
        now = k[-1][0] + 1

        result, dropped = drop_unclosed_klines(k, HOUR, now_ms=now)

        self.assertTrue(dropped)
        self.assertEqual(result, [])

    def test_does_not_mutate_input(self):
        k = bars(4)
        before = [list(r) for r in k]

        drop_unclosed_klines(k, HOUR, now_ms=k[-1][0] + 1)

        self.assertEqual(k, before, "不应原地修改调用方的数据")


if __name__ == "__main__":
    unittest.main(verbosity=2)
