"""
信号状态机（services/signal_state_machine.py）单元测试

重点覆盖三件事：
  1. 迟滞 + 连续确认真的能阻止"每根翻转"（这是设计稿的核心诉求）；
  2. 失效线的语义正确，且失效后**不会当根立刻重入**；
  3. **因果性**：篡改未来 bar 不得改变历史状态。

第 3 点不是形式主义 —— 本仓库在 `experiments/state_machine_pnl.py` 的第一版实现里
就犯过"用第 i 根收盘价决定第 i 根仓位"的未来函数错误，跑出 +950%/Sharpe 4.75 的假结果。
状态机同样容易犯（pivot 检测天然想用未来 bar 确认），所以这里钉死。
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.signal_state_machine import (  # noqa: E402
    BEAR_STATES,
    BULL_STATES,
    STATE_POSITION,
    STATE_TO_TREND_CODE,
    SignalStateMachine,
    StateMachineParams,
    calibrate_thresholds,
    directional_score,
)


def run(scores, closes, params=None, highs=None, lows=None):
    """把分数/价格序列喂进状态机，返回每根之后的状态快照列表"""
    sm = SignalStateMachine(params or StateMachineParams())
    out = []
    for i, (sc, c) in enumerate(zip(scores, closes)):
        h = highs[i] if highs else c
        l = lows[i] if lows else c
        out.append(sm.step(score=sc, close=c, high=h, low=l))
    return out


class TestHysteresis(unittest.TestCase):
    """① 迟滞：分数在中间带时不得改方向"""

    def test_single_bar_spike_does_not_flip(self):
        """一根 K 线的模型翻转不得立即改变状态"""
        p = StateMachineParams(confirm_bars=3, stop_pct=None, enable_structure=False)
        closes = [100.0] * 10
        scores = [60.0] * 5 + [40.0] + [60.0] * 4      # 中间插一根反向
        snaps = run(scores, closes, p)

        states = [s.state for s in snaps]
        self.assertIn("BULLISH", states[:6], "前 5 根应进入看涨")
        self.assertNotIn("BEARISH", states, "单根反向不应产生看跌状态")

    def test_hold_band_keeps_state(self):
        """持有带（bear_enter=45 < score < bull_enter=55）：原状态不变"""
        p = StateMachineParams(confirm_bars=2, stop_pct=None, enable_structure=False)
        scores = [60.0, 60.0, 51.0, 51.0, 51.0]     # 51 在 45~55 之间，且 > bull_exit=50
        snaps = run(scores, [100.0] * 5, p)

        self.assertEqual(snaps[1].state, "BULLISH")
        self.assertEqual(snaps[-1].state, "BULLISH", "持有带内不得降级也不得翻向")
        self.assertEqual(snaps[-1].position, 1.0)

    def test_weak_state_below_exit_but_above_bear_enter(self):
        """跌破 bull_exit=50 但未到 bear_enter=45：降级为 WEAK 预警，仓位不变"""
        p = StateMachineParams(confirm_bars=2, stop_pct=None, enable_structure=False)
        scores = [60.0, 60.0, 48.0, 48.0, 48.0]     # 45 < 48 < 50
        snaps = run(scores, [100.0] * 5, p)

        self.assertEqual(snaps[1].state, "BULLISH")
        self.assertEqual(snaps[-1].state, "WEAK_BULLISH", "应降级为弱看涨而非翻空")
        self.assertEqual(snaps[-1].position, 1.0, "WEAK_BULLISH 仓位仍为多")

    def test_confirm_bars_required(self):
        """连续 confirm_bars 根同向才切换；少一根就不切"""
        p3 = StateMachineParams(confirm_bars=3, enable_structure=False, stop_pct=None)
        p5 = StateMachineParams(confirm_bars=5, enable_structure=False, stop_pct=None)
        scores = [60.0] * 4 + [40.0] * 4

        s3 = run(scores, [100.0] * len(scores), p3)
        s5 = run(scores, [100.0] * len(scores), p5)
        self.assertIn("BEARISH", [x.state for x in s3], "3 根确认应切换")
        self.assertNotIn("BEARISH", [x.state for x in s5], "5 根确认时 4 根反向不足以切换")

    def test_whipsaw_is_suppressed(self):
        """每根都翻转的高频噪声信号：状态切换应被确认窗口压到 0

        注意构造方式：若把信号造成"每 3 根一变"，正好等于 confirm_bars=3 的确认窗口，
        切换会原样通过 —— 那样等于没测到抖动抑制（第一版测试就是这么写的）。
        真正的抖动是**逐根翻转**。
        """
        p = StateMachineParams(confirm_bars=3, enable_structure=False, stop_pct=None)
        scores = [60.0 if i % 2 == 0 else 40.0 for i in range(60)]   # 每根翻转
        sm = SignalStateMachine(p)
        for sc in scores:
            sm.step(score=sc, close=100.0, high=100.0, low=100.0)
        self.assertEqual(sm.transitions, 0,
                         "逐根翻转的噪声不应产生任何状态切换")

    def test_confirmed_trend_still_switches(self):
        """反向对照：真正持续的行情必须能切换，别把状态机测成永远不动"""
        p = StateMachineParams(confirm_bars=3, enable_structure=False, stop_pct=None)
        scores = [60.0] * 10 + [40.0] * 10                # 各 10 根，远超确认窗口
        sm = SignalStateMachine(p)
        for sc in scores:
            sm.step(score=sc, close=100.0, high=100.0, low=100.0)
        self.assertGreaterEqual(sm.transitions, 2, "应发生 看涨 -> 看跌 的切换")
        self.assertEqual(sm.snap.state, "BEARISH")


class TestInvalidation(unittest.TestCase):
    """② 失效线：跌破 signal_price*(1-stop_pct) 则方向失效"""

    def _params(self, **kw):
        base = dict(confirm_bars=1, stop_pct=0.03, enable_structure=False)
        base.update(kw)
        return StateMachineParams(**base)

    def test_signal_price_recorded_on_entry(self):
        sm = SignalStateMachine(self._params())
        sm.step(score=60.0, close=70000.0, high=70100.0, low=69900.0)
        s = sm.snap
        self.assertEqual(s.state, "BULLISH")
        self.assertEqual(s.signal_price, 70000.0, "应记录入场时的 signal_price")
        self.assertAlmostEqual(s.invalidation_price, 70000.0 * 0.97, places=6)

    def test_invalidation_triggers_neutral(self):
        """设计稿例子：70000 入场，跌破 69300 则看涨失效"""
        sm = SignalStateMachine(self._params())
        sm.step(score=60.0, close=70000.0, high=70100.0, low=69900.0)
        # 失效线 = 70000 * (1-0.03) = 67900
        s = sm.step(score=60.0, close=67800.0, high=70000.0, low=67700.0)

        self.assertEqual(s.state, "NEUTRAL", "跌破失效线应转中性")
        self.assertEqual(s.position, 0.0)
        self.assertEqual(s.invalidation_reason, "close_below_invalidation")
        self.assertEqual(sm.invalidations, 1)

    def test_no_invalidation_within_buffer(self):
        """噪声缓冲：69980 这种小幅跌破（未到 69300）不应触发"""
        sm = SignalStateMachine(self._params())
        sm.step(score=60.0, close=70000.0, high=70100.0, low=69900.0)
        s = sm.step(score=60.0, close=69980.0, high=70100.0, low=69900.0)

        self.assertEqual(s.state, "BULLISH", "缓冲区内的波动不应触发失效")

    def test_latch_prevents_immediate_reentry(self):
        """失效后锁住：即使信号仍然是看涨，也不得当根重入"""
        sm = SignalStateMachine(self._params())
        sm.step(score=60.0, close=70000.0, high=70100.0, low=69900.0)
        s1 = sm.step(score=60.0, close=67000.0, high=70000.0, low=66900.0)   # < 67900
        s2 = sm.step(score=60.0, close=67000.0, high=67000.0, low=66900.0)

        self.assertEqual(s1.state, "NEUTRAL")
        self.assertTrue(s1.latched)
        self.assertEqual(s2.state, "NEUTRAL", "锁定期间不得重入")
        self.assertEqual(s2.position, 0.0)

    def test_latch_releases_on_new_signal(self):
        """出现与"被否掉方向"不同的新信号后解锁"""
        sm = SignalStateMachine(self._params())
        sm.step(score=60.0, close=70000.0, high=70100.0, low=69900.0)
        sm.step(score=60.0, close=67000.0, high=70000.0, low=66900.0)
        self.assertTrue(sm._latched)
        s = sm.step(score=40.0, close=68000.0, high=69000.0, low=67900.0)
        self.assertFalse(s.latched, "反向新信号应解锁")
        self.assertEqual(s.state, "BEARISH")

    def test_trailing_disabled_by_default(self):
        """移动失效线实测有害，默认必须关闭"""
        self.assertEqual(StateMachineParams().trail_atr_mult, 0.0)

    def test_trailing_line_follows_high_water(self):
        """开启时：失效线随持有期最高价抬高"""
        p = self._params(trail_atr_mult=2.0, stop_pct=None)
        sm = SignalStateMachine(p)
        sm.step(score=60.0, close=70000.0, high=70000.0, low=69000.0, atr=1000.0)
        first = sm.snap.invalidation_price
        sm.step(score=60.0, close=80000.0, high=80000.0, low=79000.0, atr=1000.0)
        s = sm.snap
        self.assertEqual(s.high_water, 80000.0)
        self.assertIsNotNone(s.invalidation_price)
        # 多头失效线取"更高"的那条 -> 应随最高价上移
        self.assertGreater(s.invalidation_price, first)


class TestCausality(unittest.TestCase):
    """③ 因果性：未来 bar 不得影响历史状态"""

    def _replay(self, closes, scores, params=None):
        return run(scores, closes, params or StateMachineParams(
            confirm_bars=2, enable_structure=True, stop_pct=0.03))

    def test_future_bars_do_not_change_history(self):
        n = 60
        closes = [100.0 + i * 0.5 for i in range(n)]
        scores = [55.0 + (i % 7) for i in range(n)]
        highs = [c + 1.0 for c in closes]
        lows = [c - 1.0 for c in closes]

        base = run(scores, closes, highs=highs, lows=lows)

        # 篡改后半段的未来数据（价格腰斩、分数反向）
        c2 = closes[:30] + [50.0] * (n - 30)
        s2 = scores[:30] + [20.0] * (n - 30)
        h2 = highs[:30] + [51.0] * (n - 30)
        l2 = lows[:30] + [49.0] * (n - 30)
        alt = run(s2, c2, highs=h2, lows=l2)

        for i in range(30):
            with self.subTest(bar=i):
                self.assertEqual(base[i].state, alt[i].state,
                                 f"第 {i} 根状态被未来数据改变了")
                self.assertEqual(base[i].position, alt[i].position)
                self.assertEqual(base[i].trend_code, alt[i].trend_code)

    def test_pivot_detection_uses_only_confirmed_pivots(self):
        """pivot 需要 k 根之后才能确认；确认前不得被用作结构判断"""
        p = StateMachineParams(enable_structure=True, pivot_halfwidth=3,
                              confirm_bars=1, stop_pct=None)
        sm = SignalStateMachine(p)
        # 造一个尖峰：如果不做滞后确认，尖峰当根就会被当成 pivot high
        # 尖峰在第 4 根（下标 3），k=3 -> 需要第 7 根（下标 6）才能确认
        for c in [100.0, 100.0, 100.0, 200.0, 100.0, 100.0]:
            sm.step(score=50.0, close=c, high=c, low=c)
        ph, _ = sm._confirmed_pivots(3)
        self.assertEqual(ph, [], "只喂 6 根时尖峰尚未被确认，不得进入 pivot 列表")
        sm.step(score=50.0, close=100.0, high=100.0, low=100.0)
        ph2, _ = sm._confirmed_pivots(3)
        self.assertEqual(ph2, [200.0], "满 k 根后应被确认为 pivot high")


class TestStateEncoding(unittest.TestCase):
    def test_state_tables_are_consistent(self):
        for st in STATE_POSITION:
            self.assertIn(st, STATE_TO_TREND_CODE, f"{st} 缺少 trend_code 映射")

    def test_weak_states_keep_direction(self):
        """WEAK_* 是预警不是翻向：仓位必须与原方向一致"""
        self.assertEqual(STATE_POSITION["WEAK_BULLISH"], STATE_POSITION["BULLISH"])
        self.assertEqual(STATE_POSITION["WEAK_BEARISH"], STATE_POSITION["BEARISH"])
        self.assertEqual(STATE_TO_TREND_CODE["WEAK_BULLISH"],
                         STATE_TO_TREND_CODE["BULLISH"])
        self.assertEqual(STATE_TO_TREND_CODE["WEAK_BEARISH"],
                         STATE_TO_TREND_CODE["BEARISH"])

    def test_neutral_is_flat(self):
        self.assertEqual(STATE_POSITION["NEUTRAL"], 0.0)
        self.assertEqual(STATE_TO_TREND_CODE["NEUTRAL"], 1)


class TestDirectionalScore(unittest.TestCase):
    def test_matches_codebase_formula(self):
        """50 + 50*(p_bull - p_bear)/100；三概率都参与"""
        probs = {"看跌": 35.40, "中性": 31.19, "看涨": 33.41}
        self.assertAlmostEqual(directional_score(probs), 49.005, places=3)
        self.assertAlmostEqual(
            directional_score([35.40, 31.19, 33.41]), 49.005, places=3)

    def test_neutral_heavy_gives_50(self):
        """模型强烈说"说不清"时，分数必须回到 50（旧公式会误判成方向）"""
        self.assertAlmostEqual(
            directional_score({"看跌": 5.0, "中性": 90.0, "看涨": 5.0}), 50.0, places=6)


class TestCalibration(unittest.TestCase):
    def test_calibrates_to_actual_distribution(self):
        """本模型分数只在 43.6~58.0 之间：标定后阈值必须落在这个区间内"""
        scores = [43.6 + (58.0 - 43.6) * i / 999 for i in range(1000)]
        th = calibrate_thresholds(scores, percentile=80.0)
        self.assertGreaterEqual(th["bull_enter"], 43.6)
        self.assertLessEqual(th["bull_enter"], 58.0)
        self.assertGreater(th["bull_enter"], th["bear_enter"])
        # 写死的 70/30 在这个分布下永远够不到 —— 这正是旧状态机失效的原因
        self.assertLess(th["bull_enter"], 70.0)

    def test_empty_scores_fall_back(self):
        th = calibrate_thresholds([])
        self.assertIn("bull_enter", th)
        self.assertGreater(th["bull_enter"], th["bear_enter"])


class TestPersistence(unittest.TestCase):
    def test_state_dict_roundtrip(self):
        sm = SignalStateMachine(StateMachineParams(confirm_bars=2, enable_structure=False))
        for _ in range(5):
            sm.step(score=60.0, close=100.0, high=101.0, low=99.0)
        d = sm.state_dict()
        self.assertEqual(d["snapshot"]["state"], "BULLISH")
        self.assertEqual(d["bar_index"], 4)
        self.assertIn("params", d)
        self.assertEqual(d["params"]["confirm_bars"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
