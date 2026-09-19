"""
趋势状态机的单元测试

覆盖的回归：中性档曾被彻底吃掉 ——
``base_score = p_bull/(p_bull+p_bear)`` 把中性概率约掉再归一化，于是"强烈说不清方向"的
预测 ``[0.05, 0.85, 0.10]`` 算出 66.7 分、放大后越过 70 分阈值被判成**看涨**；
加上状态机只有两态，``trend_code`` 永远只能是 0 或 2，中性从未出现在任何输出里。

只用标准库（trend_manager 已去除 numpy 依赖）。
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.trend_manager import (  # noqa: E402
    STATE_TO_TREND_CODE,
    TREND_CODE_TO_STATE,
    TREND_LABELS,
    TrendManager,
)

PRICE = 80000.0

# 典型概率向量
STRONG_BULL = [0.05, 0.05, 0.90]
STRONG_BEAR = [0.90, 0.05, 0.05]
STRONG_NEUTRAL = [0.05, 0.85, 0.10]  # 模型 85% 说"中性"
MILD_BULL = [0.0909, 0.5455, 0.3636]  # 方向分 ~63.6 -> 放大后 ~70.4（越过阈值但不触发绝对重置）
MILD_BEAR = [0.3636, 0.5455, 0.0909]
DEAD_EVEN = [0.30, 0.40, 0.30]  # 完全无方向


class TestDirectionalScore(unittest.TestCase):
    def setUp(self):
        self.tm = TrendManager()

    def test_neutral_mass_shrinks_the_score(self):
        """核心回归：中性概率大时方向分必须回到 50 附近，而不是被归一化放大"""
        score = self.tm.directional_score(STRONG_NEUTRAL)

        self.assertAlmostEqual(score, 52.5, places=6)

        # 旧公式在同一输入下会算出 66.7 —— 正是"中性被吃成看涨"的来源
        old_score = (0.10 / (0.10 + 0.05)) * 100
        self.assertAlmostEqual(old_score, 66.666, places=2)
        self.assertLess(score, old_score)

    def test_strong_direction_scores_high(self):
        # 50 + 50*(0.90-0.05) = 92.5
        self.assertAlmostEqual(self.tm.directional_score(STRONG_BULL), 92.5, places=6)
        self.assertAlmostEqual(self.tm.directional_score(STRONG_BEAR), 7.5, places=6)

    def test_uniform_probabilities_score_50(self):
        self.assertAlmostEqual(
            self.tm.directional_score([1 / 3, 1 / 3, 1 / 3]), 50.0, places=6
        )

    def test_score_is_scale_invariant(self):
        self.assertAlmostEqual(
            self.tm.directional_score([1.0, 17.0, 2.0]),
            self.tm.directional_score([0.05, 0.85, 0.10]),
            places=6,
        )

    def test_degenerate_input_is_neutral(self):
        self.assertAlmostEqual(self.tm.directional_score([0.0, 0.0, 0.0]), 50.0)


class TestNeutralIsRepresentable(unittest.TestCase):
    def test_first_call_can_land_on_neutral(self):
        tm = TrendManager()

        res = tm.update(PRICE, STRONG_NEUTRAL)

        self.assertEqual(res["final_trend"], "Neutral")
        self.assertEqual(res["trend_code"], 1)

    def test_strongly_neutral_model_never_reports_a_direction(self):
        """回归：以前第一次调用就会把 [0.05,0.85,0.10] 判成 Bullish"""
        tm = TrendManager()
        states = set()

        for _ in range(30):
            res = tm.update(PRICE, STRONG_NEUTRAL)
            states.add(res["final_trend"])

        self.assertEqual(states, {"Neutral"}, f"不应出现方向信号，实际: {states}")

    def test_dead_even_stays_neutral(self):
        tm = TrendManager()
        for _ in range(5):
            res = tm.update(PRICE, DEAD_EVEN)

        self.assertEqual(res["final_trend"], "Neutral")

    def test_neutral_entry_price_tracks_price(self):
        tm = TrendManager()
        tm.update(PRICE, DEAD_EVEN)
        res = tm.update(PRICE * 1.001, DEAD_EVEN)

        self.assertEqual(res["entry_price"], PRICE * 1.001)


class TestStateMachine(unittest.TestCase):
    def test_strong_signal_enters_direction(self):
        tm = TrendManager()

        res = tm.update(PRICE, STRONG_BULL)

        self.assertEqual(res["final_trend"], "Bullish")
        self.assertEqual(res["trend_code"], 2)

    def test_exit_to_neutral_before_reversing(self):
        """滞回：看涨支撑减弱先退到中性，而不是立刻反手做空"""
        tm = TrendManager()
        self.assertEqual(tm.update(PRICE, MILD_BULL)["final_trend"], "Bullish")

        mid = tm.update(PRICE, DEAD_EVEN)
        self.assertEqual(mid["final_trend"], "Neutral", "应先退出到中性")
        self.assertTrue(mid["state_changed"])

        strong = tm.update(PRICE, STRONG_BEAR)
        self.assertEqual(strong["final_trend"], "Bearish")

    def test_absolute_threshold_can_cross_directly(self):
        tm = TrendManager()
        self.assertEqual(tm.update(PRICE, STRONG_BULL)["final_trend"], "Bullish")

        res = tm.update(PRICE, STRONG_BEAR)

        self.assertEqual(res["final_trend"], "Bearish", "绝对强度的信号允许直接跨方向")

    def test_price_break_with_model_agreement_reverses(self):
        tm = TrendManager(bullish_threshold=70.0, price_break_pct=0.01)
        tm.update(PRICE, MILD_BULL)
        self.assertEqual(tm.current_state, "Bullish")

        # 模型转空 + 价格实质下跌 2%（但分数未达绝对阈值）
        res = tm.update(PRICE * 0.98, MILD_BEAR)

        self.assertEqual(res["final_trend"], "Bearish")

    def test_state_changed_flag(self):
        tm = TrendManager()
        first = tm.update(PRICE, STRONG_BULL)
        second = tm.update(PRICE, STRONG_BULL)

        self.assertTrue(first["state_changed"], "首次进入状态视为变化")
        self.assertFalse(second["state_changed"], "状态未变")

    def test_return_shape(self):
        tm = TrendManager()

        res = tm.update(PRICE, STRONG_BULL)

        for key in (
            "final_trend", "trend_code", "trend_score", "directional_score",
            "entry_price", "model_bias", "state_changed", "is_state_locked",
        ):
            self.assertIn(key, res)

    def test_is_state_locked_when_machine_disagrees_with_model(self):
        """is_state_locked = 状态机当前报告的方向得不到模型当前倾向的支持"""
        tm = TrendManager()
        tm.update(PRICE, STRONG_BULL)  # -> Bullish

        # 模型转空、但价格尚未破位：先退到中性观望，此时机器方向与模型倾向不一致
        res = tm.update(PRICE, MILD_BEAR)

        self.assertEqual(res["final_trend"], "Neutral")
        self.assertEqual(res["model_bias"], "Bearish")
        self.assertTrue(res["is_state_locked"])

        # 模型继续保持强空 -> 绝对重置为 Bearish，锁解除
        resolved = tm.update(PRICE, STRONG_BEAR)
        self.assertEqual(resolved["final_trend"], "Bearish")
        self.assertFalse(resolved["is_state_locked"])


class TestThresholds(unittest.TestCase):
    def test_defaults_come_from_config(self):
        """阈值只有 config 一个事实来源，不允许在 TrendManager 里再写一套"""
        from config import (
            TREND_ABSOLUTE_THRESHOLD,
            TREND_BEARISH_THRESHOLD,
            TREND_BULLISH_THRESHOLD,
            TREND_PRICE_BREAK_PCT,
        )

        tm = TrendManager()

        self.assertEqual(tm.bullish_threshold, TREND_BULLISH_THRESHOLD)
        self.assertEqual(tm.bearish_threshold, TREND_BEARISH_THRESHOLD)
        self.assertEqual(tm.absolute_threshold, TREND_ABSOLUTE_THRESHOLD)
        self.assertEqual(tm.price_break_pct, TREND_PRICE_BREAK_PCT)

    def test_default_bands_are_symmetric(self):
        tm = TrendManager()

        self.assertAlmostEqual(
            tm.bullish_threshold - 50.0, 50.0 - tm.bearish_threshold, places=6
        )

    def test_sensitive_mode_is_symmetric_too(self):
        from config import (
            TREND_SENSITIVE_BEARISH_THRESHOLD,
            TREND_SENSITIVE_BULLISH_THRESHOLD,
        )

        self.assertAlmostEqual(
            TREND_SENSITIVE_BULLISH_THRESHOLD - 50.0,
            50.0 - TREND_SENSITIVE_BEARISH_THRESHOLD,
            places=6,
        )

    def test_custom_bearish_threshold(self):
        tm = TrendManager(bullish_threshold=70.0, bearish_threshold=20.0)

        self.assertEqual(tm.bearish_threshold, 20.0)

    def test_bias_bands(self):
        tm = TrendManager(bullish_threshold=70.0, bearish_threshold=30.0)

        self.assertEqual(tm._bias(70.0), "Bullish")
        self.assertEqual(tm._bias(69.9), "Neutral")
        self.assertEqual(tm._bias(30.1), "Neutral")
        self.assertEqual(tm._bias(30.0), "Bearish")


class TestRealModelConviction(unittest.TestCase):
    """用模型实测的概率分布锁定"没有确信度就不发声"这一契约。

    BTCUSDT 1h 模型实测 |p_bull - p_bear| 均值约 0.021（训练窗口 600 样本）、
    0.018（样本外 553 样本）—— 多空概率差只有 2 个百分点左右。
    默认阈值(70/30)下这类分布必须判为中性；旧实现会把它们几乎全部说成看涨或看跌。
    """

    REAL_SAMPLES = [
        [0.3693, 0.2051, 0.4257],   # 线上真实取到过的分布
        [0.269, 0.465, 0.266],
        [0.35, 0.40, 0.25],
        [0.32, 0.30, 0.38],
    ]

    def test_real_model_distributions_are_neutral(self):
        for probs in self.REAL_SAMPLES:
            with self.subTest(probs=probs):
                res = TrendManager().update(PRICE, probs)
                self.assertEqual(
                    res["final_trend"], "Neutral",
                    f"{probs} 的多空分离度不足以支撑方向信号",
                )

    def test_old_rule_would_have_spoken_on_the_same_inputs(self):
        """对照：旧规则（>=70 看涨 / <50 看跌 / 其余中性）在这些真实分布上会给出方向。

        旧实现的不对称是关键：看跌只要 <50，而看涨要 >=70。
        方向分均值约 50、标准差约 3，于是大量样本落进 [0,50) 被标成看跌 ——
        样本外评估里 71% 的预测是看跌，就是这么来的。
        """
        old_spoke = []
        for p_bear, _p_neutral, p_bull in self.REAL_SAMPLES:
            old_score = 50 + 50 * (p_bull - p_bear) / (p_bull + p_bear)
            old_amplified = max(0.0, min(100.0, 50 + (old_score - 50) * 1.5))
            old_bias = (
                "Bullish" if old_amplified >= 70.0
                else ("Bearish" if old_amplified < 50.0 else "Neutral")
            )
            new_state = TrendManager().update(PRICE, [p_bear, _p_neutral, p_bull])["final_trend"]
            old_spoke.append(old_bias != "Neutral" and new_state == "Neutral")

        self.assertTrue(
            any(old_spoke),
            "旧规则至少会在其中一个真实分布上给出方向（这才说明本次修复有意义）",
        )

    def test_confident_distribution_does_speak(self):
        """有确信度时必须发声，否则只是把中性做成了永远沉默"""
        res = TrendManager().update(PRICE, [0.05, 0.05, 0.90])

        self.assertEqual(res["final_trend"], "Bullish")
        self.assertGreaterEqual(res["directional_score"], 90.0)


class TestMomentum(unittest.TestCase):
    def test_no_bias_before_enough_history(self):
        tm = TrendManager()
        for _ in range(8):
            tm.update(PRICE, DEAD_EVEN)

        # 本次调用会把样本数补到 9，仍不足 10 个 -> 不修正
        self.assertEqual(tm._calculate_momentum_bias(PRICE * 1.05), 0.0)

    def test_upward_momentum_raises_score(self):
        tm = TrendManager()
        for i in range(12):
            tm.update(PRICE * (1 + 0.002 * i), DEAD_EVEN)

        bias = tm._calculate_momentum_bias(PRICE * 1.03)

        self.assertGreater(bias, 0)

    def test_momentum_bias_is_clipped(self):
        tm = TrendManager()
        for _ in range(12):
            tm.update(PRICE, DEAD_EVEN)

        self.assertEqual(tm._calculate_momentum_bias(PRICE * 2), 10.0)
        self.assertEqual(tm._calculate_momentum_bias(PRICE * 0.5), -10.0)

    def test_price_history_is_bounded(self):
        tm = TrendManager()
        for i in range(100):
            tm.update(PRICE + i, DEAD_EVEN)

        self.assertLessEqual(len(tm.price_history), 24)


class TestCodeMapping(unittest.TestCase):
    def test_state_to_code_matches_three_class_encoding(self):
        self.assertEqual(STATE_TO_TREND_CODE, {"Bearish": 0, "Neutral": 1, "Bullish": 2})

    def test_code_to_state_is_inverse(self):
        for state, code in STATE_TO_TREND_CODE.items():
            self.assertEqual(TREND_CODE_TO_STATE[code], state)

    def test_labels_match_mapping(self):
        for state, code in STATE_TO_TREND_CODE.items():
            self.assertIn(code, TREND_LABELS)
        self.assertEqual(TREND_LABELS[1], "中性")

    def test_all_three_codes_are_reachable(self):
        tm = TrendManager()
        seen = {
            tm.update(PRICE, STRONG_BULL)["trend_code"],
            TrendManager().update(PRICE, DEAD_EVEN)["trend_code"],
            tm.update(PRICE, STRONG_BEAR)["trend_code"],
        }

        self.assertEqual(seen, {0, 1, 2})


if __name__ == "__main__":
    unittest.main(verbosity=2)
