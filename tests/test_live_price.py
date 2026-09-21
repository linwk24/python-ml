"""
实时价通路（services/live_price.py）的单元测试

覆盖两类风险：
  1. 实时价获取失败时**不能**让 /predict 崩掉，也不能静默返回 0 价格；
  2. "未收盘那根"必须被硬隔离在模型输入之外 —— 这是本模块存在的全部意义。
"""

import os
import sys
import unittest
from unittest import mock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402
from services import live_price  # noqa: E402


HOUR_MS = 3600 * 1000

# 一根正在走的 1h K 线（open_time 用"现在往前推 27 分钟"以便 elapsed 可控）
def forming_kline(open_ms, o=100.0, h=102.0, l=99.0, c=101.0, v=5.0):
    return [open_ms, o, h, l, c, v]


class TestFetchLive(unittest.TestCase):
    def _patch(self, payload, now_ms=None):
        """把 _get_json 换成返回固定 payload，并把当前时间固定住"""
        now_ms = now_ms if now_ms is not None else (1_700_000_000_000 + 27 * 60 * 1000)
        return mock.patch.object(live_price, "_get_json", return_value=payload), \
            mock.patch.object(live_price.time, "time", return_value=now_ms / 1000.0)

    def test_parses_price_and_forming_bar(self):
        open_ms = 1_700_000_000_000
        p, t = self._patch([forming_kline(open_ms), forming_kline(open_ms, c=101.5)])
        with p, t:
            out = live_price.fetch_live("BTCUSDT", "1h")

        self.assertIsNone(out["error"])
        self.assertEqual(out["price"], 101.5, "取最后一根（正在走的）的 close")
        bar = out["forming_bar"]
        self.assertEqual(bar["open_ms"], open_ms)
        self.assertEqual(bar["high"], 102.0)
        self.assertEqual(bar["low"], 99.0)
        self.assertFalse(bar["complete"], "只走了 27 分钟，不该标记为已收盘")
        self.assertAlmostEqual(bar["elapsed_minutes"], 27.0, places=1)

    def test_marks_for_model_false(self):
        """硬约束：这个块必须自带 'for_model': False"""
        open_ms = 1_700_000_000_000
        p, t = self._patch([forming_kline(open_ms)])
        with p, t:
            out = live_price.fetch_live("BTCUSDT", "1h")
        self.assertIs(out["for_model"], False)

    def test_network_failure_returns_error_not_raise(self):
        with mock.patch.object(live_price, "_get_json",
                               side_effect=OSError("connection reset")):
            out = live_price.fetch_live("BTCUSDT", "1h")

        self.assertIsNone(out["price"], "失败时价格必须是 None，不能是 0")
        self.assertIn("实时价获取失败", out["error"])
        self.assertIs(out["for_model"], False)

    def test_malformed_payload_returns_error(self):
        for bad in (None, {}, [], "oops"):
            with self.subTest(bad=bad):
                p, t = self._patch(bad)
                with p, t:
                    out = live_price.fetch_live("BTCUSDT", "1h")
                self.assertIsNotNone(out["error"])
                self.assertIsNone(out["price"])

    def test_short_row_returns_error(self):
        p, t = self._patch([[1_700_000_000_000, 1.0, 2.0]])
        with p, t:
            out = live_price.fetch_live("BTCUSDT", "1h")
        self.assertIn("字段解析失败", out["error"])

    def test_complete_flag_when_past_interval(self):
        open_ms = 1_700_000_000_000
        # 当前时间已过 61 分钟 -> 这根其实已经收盘了
        p, t = self._patch([forming_kline(open_ms)],
                           now_ms=open_ms + 61 * 60 * 1000)
        with p, t:
            out = live_price.fetch_live("BTCUSDT", "1h")
        self.assertTrue(out["forming_bar"]["complete"])
        self.assertEqual(out["forming_bar"]["progress"], 1.0)


class TestLiveFromFormingBar(unittest.TestCase):
    """零延迟路径：复用 fetch_kline_data 已经取到的未收盘那根"""

    def test_computes_price_and_progress(self):
        open_ms = 1_700_000_000_000
        bar = forming_kline(open_ms, o=100.0, h=102.0, l=99.0, c=101.5, v=7.0)
        out = live_price.live_from_forming_bar(bar, "1h", now_ms=open_ms + 27 * 60 * 1000)

        self.assertIsNone(out["error"])
        self.assertEqual(out["price"], 101.5)
        self.assertEqual(out["forming_bar"]["low"], 99.0)
        self.assertEqual(out["forming_bar"]["volume"], 7.0)
        self.assertAlmostEqual(out["forming_bar"]["elapsed_minutes"], 27.0, places=1)
        self.assertAlmostEqual(out["forming_bar"]["progress"], 0.45, places=2)
        self.assertFalse(out["forming_bar"]["complete"])
        self.assertIs(out["for_model"], False)

    def test_none_forming_bar_yields_error_not_crash(self):
        """恰好卡在收盘边界时没有半根 —— 必须是 None 而不是抛异常或 0"""
        out = live_price.live_from_forming_bar(None, "1h")
        self.assertIsNone(out["price"])
        self.assertIn("未取到", out["error"])
        self.assertIs(out["for_model"], False)

    def test_does_not_make_http_request(self):
        """首选路径必须零出网"""
        open_ms = 1_700_000_000_000
        with mock.patch.object(live_price, "_get_json",
                               side_effect=AssertionError("不该出网")):
            out = live_price.live_from_forming_bar(forming_kline(open_ms), "1h",
                                                   now_ms=open_ms + 60_000)
        self.assertEqual(out["price"], 101.0)

    def test_build_live_block_prefers_given_bar(self):
        open_ms = 1_700_000_000_000
        with mock.patch.object(live_price, "_get_json",
                               side_effect=AssertionError("不该出网")):
            block = live_price.build_live_block("BTCUSDT", "1h",
                                                forming_kline(open_ms, c=55.5))
        self.assertEqual(block["live"]["price"], 55.5)
        self.assertTrue(block["live"]["enabled"])

    def test_malformed_forming_bar(self):
        # 空列表是 falsy，语义上等同"没取到"，走的是同一条分支
        empty = live_price.live_from_forming_bar([], "1h")
        self.assertIsNone(empty["price"])
        self.assertIn("未取到", empty["error"])

        # 字段不足才是解析失败
        short = live_price.live_from_forming_bar([1.0, 2.0], "1h")
        self.assertIsNone(short["price"])
        self.assertIn("字段解析失败", short["error"])


class TestPositionGuard(unittest.TestCase):
    def test_rejects_none_and_zero(self):
        """拿不到价格时禁止做仓位决策 —— 静默当 0 会让止损算出错误盈亏"""
        for bad in (None, 0, 0.0, -1.0):
            with self.subTest(bad=bad):
                self.assertFalse(live_price.position_guard(bad))

    def test_accepts_positive(self):
        self.assertTrue(live_price.position_guard(0.00001))


class TestModelPathIsolation(unittest.TestCase):
    """核心不变量：未收盘那根绝不能进模型输入"""

    def test_drop_unclosed_removes_forming_bar(self):
        """fetch_kline_data 默认丢弃未收盘那根 —— 模型路径的硬约束"""
        now_ms = 1_700_000_000_000 + 27 * 60 * 1000
        open_ms = 1_700_000_000_000
        klines = [
            [open_ms - HOUR_MS, 1, 1, 1, 100.0, 1],
            [open_ms, 1, 1, 1, 101.0, 1],          # 正在走
        ]
        kept, dropped = config.drop_unclosed_klines(klines, HOUR_MS, now_ms=now_ms)
        self.assertTrue(dropped)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[-1][0], open_ms - HOUR_MS, "留下的必须是已收盘那根")

    def test_live_block_is_not_klines_shaped(self):
        """实时价块的结构不能直接当 klines 用（防止被顺手拼回模型输入）"""
        open_ms = 1_700_000_000_000
        with mock.patch.object(live_price, "_get_json",
                               return_value=[forming_kline(open_ms)]), \
             mock.patch.object(live_price.time, "time",
                               return_value=(open_ms + 600_000) / 1000.0):
            out = live_price.fetch_live("BTCUSDT", "1h")

        self.assertIsInstance(out, dict)
        self.assertNotIsInstance(out, list)
        # 它不是一个可以直接 append 进 klines 的 bar
        self.assertNotIn(0, out, "键是字符串，不是 K 线的整数下标")

    def test_disabled_flag(self):
        """LIVE_PRICE_ENABLED=False 时不出网"""
        with mock.patch.object(config, "LIVE_PRICE_ENABLED", False), \
             mock.patch.object(live_price, "_get_json",
                               side_effect=AssertionError("不该出网")):
            block = live_price.build_live_block("BTCUSDT", "1h")
        self.assertFalse(block["live"]["enabled"])
        self.assertIsNone(block["live"]["price"])

    def test_prediction_failure_isolated_from_live_failure(self):
        """实时价挂了，预测结果本身仍应完整返回"""
        with mock.patch.object(live_price, "fetch_live",
                               return_value={"price": None, "for_model": False,
                                             "error": "boom"}):
            block = live_price.build_live_block("BTCUSDT", "1h")
        self.assertEqual(block["live"]["error"], "boom")
        self.assertIn("live", block)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestHorizonDeclaration(unittest.TestCase):
    """响应必须声明视界 —— 否则 trend 字段会被读成多日趋势"""

    @classmethod
    def setUpClass(cls):
        import config
        src = open(os.path.join(PROJECT_ROOT, "app", "main.py")).read()
        fn = src[src.index("def horizon_block"):src.index("# current_price 的口径声明")]
        cls._ns = {"interval_to_ms": config.interval_to_ms,
                   "PREDICTION_HORIZON_BARS": config.PREDICTION_HORIZON_BARS}
        exec(fn, cls._ns)

    def test_reports_bars_and_minutes(self):
        r = self._ns["horizon_block"]("1h")
        self.assertEqual(r["prediction_horizon_bars"], 1)
        self.assertEqual(r["prediction_horizon_minutes"], 60)

    def test_scales_with_interval(self):
        for iv, minutes in (("1m", 1), ("15m", 15), ("1h", 60), ("4h", 240), ("1d", 1440)):
            with self.subTest(interval=iv):
                r = self._ns["horizon_block"](iv)
                self.assertEqual(r["prediction_horizon_minutes"], minutes)

    def test_scope_text_says_it_is_not_a_trend_view(self):
        """文案必须明确排除"多日趋势"的误读"""
        text = self._ns["horizon_block"]("1h")["prediction_scope"]
        self.assertIn("下一根 K 线", text)
        self.assertIn("不是多日趋势", text)
        self.assertIn("60 分钟", text)

    def test_bad_interval_does_not_crash(self):
        r = self._ns["horizon_block"]("not-an-interval")
        self.assertEqual(r["prediction_horizon_bars"], 1)
        self.assertGreater(r["prediction_horizon_minutes"], 0)
