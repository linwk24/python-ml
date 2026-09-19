"""
标签定义的单元测试

重点钉住两件事：

1. **因果性**：第 j 根的标签只允许用到第 j+1 根的收益；阈值只能用到第 j 根及更早的信息。
   把"未来"的数据改掉，不能改变过去的标签 —— 否则就是标签泄漏。
2. **默认口径不变**：``labels_fixed`` 必须与生产历史的固定 ±0.1% 口径逐行一致。

需要 pandas（不需要 TensorFlow）。
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import numpy as np
    import pandas as pd

    from config import LABEL_THRESHOLD
    from services.labeling import (
        LABEL_MODES,
        get_label_fn,
        label_distribution,
        labels_fixed,
        labels_quantile,
        labels_volatility,
    )

    HAS_PANDAS = True
    SKIP_REASON = ""
except Exception as e:  # pragma: no cover
    HAS_PANDAS = False
    SKIP_REASON = f"pandas 不可用: {e}"


def frame(closes, highs=None, lows=None):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "open_time": np.arange(len(closes)) * 3600000,
        "open": closes,
        "high": np.asarray(highs, dtype=float) if highs is not None else closes * 1.001,
        "low": np.asarray(lows, dtype=float) if lows is not None else closes * 0.999,
        "close": closes,
        "volume": np.ones(len(closes)),
    })


def random_walk(n=1500, sigma=0.004, seed=0):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, sigma, n)
    return 100 * np.exp(np.cumsum(ret))


@unittest.skipUnless(HAS_PANDAS, SKIP_REASON)
class TestCausality(unittest.TestCase):
    """把所有标签模式的因果性都验一遍"""

    def make(self, mode, df):
        if mode == "A":
            return labels_fixed(df)
        if mode == "B_sigma":
            return labels_volatility(df, k=0.8, scale="sigma")
        if mode == "B_atr":
            return labels_volatility(df, k=0.8, scale="atr")
        return labels_quantile(df)

    def test_future_prices_do_not_affect_past_labels(self):
        closes = random_walk(1200)
        base = frame(closes)

        for mode in ("A", "B_sigma", "B_atr", "C"):
            with self.subTest(mode=mode):
                before = self.make(mode, base)
                # 改动最后 50 根价格（属于"未来"）：更早的标签必须一字不变
                perturbed = closes.copy()
                perturbed[-50:] *= 1.15
                after = self.make(mode, frame(perturbed))
                cut = len(closes) - 100
                np.testing.assert_array_equal(
                    before[:cut], after[:cut],
                    err_msg=f"{mode}: 过去的标签被未来的数据改变了（标签泄漏）",
                )

    def test_threshold_uses_only_past(self):
        """阈值本身也必须因果：改动未来不能改变过去的标签"""
        closes = random_walk(1200, sigma=0.005, seed=3)
        perturbed = closes.copy()
        perturbed[-80:] *= 1.2

        a = labels_volatility(frame(closes), k=0.8)
        b = labels_volatility(frame(perturbed), k=0.8)

        np.testing.assert_array_equal(a[: len(a) - 100], b[: len(b) - 100])

    def test_last_row_has_no_future(self):
        """最后一根没有"下一根"，必须是中性（不制造虚假样本）"""
        df = frame(random_walk(300))
        for mode in ("A", "B_sigma", "B_atr", "C"):
            with self.subTest(mode=mode):
                self.assertEqual(int(self.make(mode, df)[-1]), 1)


@unittest.skipUnless(HAS_PANDAS, SKIP_REASON)
class TestFixedMatchesProduction(unittest.TestCase):
    def test_matches_historical_formula(self):
        """默认口径必须与历史的内联实现逐行一致"""
        closes = random_walk(600, sigma=0.003, seed=7)
        df = frame(closes)

        got = labels_fixed(df)

        expected = np.full(len(closes), 1, dtype=int)
        for j in range(len(closes) - 1):
            ch = (closes[j + 1] - closes[j]) / closes[j]
            expected[j] = 2 if ch > LABEL_THRESHOLD else (0 if ch < -LABEL_THRESHOLD else 1)

        np.testing.assert_array_equal(got, expected)


@unittest.skipUnless(HAS_PANDAS, SKIP_REASON)
class TestVolatilityNormalization(unittest.TestCase):
    def test_fixed_threshold_makes_different_tasks(self):
        """同一个 ±0.1% 在不同波动下定义了难度不同的任务（这就是要修的问题）"""
        low_vol = frame(random_walk(400, sigma=0.0005, seed=1))
        high_vol = frame(random_walk(400, sigma=0.006, seed=1))

        d_low = label_distribution(labels_fixed(low_vol))
        d_high = label_distribution(labels_fixed(high_vol))

        self.assertGreater(d_low["neutral"], 0.5)
        self.assertLess(d_high["neutral"], 0.2)

    def test_volatility_normalization_aligns_class_mix(self):
        """波动归一化后，两个不同波动样本集的中性占比应当接近"""
        low_vol = frame(random_walk(1200, sigma=0.0005, seed=1))
        high_vol = frame(random_walk(1200, sigma=0.006, seed=1))

        v_low = label_distribution(labels_volatility(low_vol, k=0.8))
        v_high = label_distribution(labels_volatility(high_vol, k=0.8))

        self.assertLess(abs(v_low["neutral"] - v_high["neutral"]), 0.15,
                        "波动率归一化后类别分布应当对齐")

    def test_k_controls_neutral_share(self):
        df = frame(random_walk(1200, sigma=0.004, seed=2))

        # 阈值 = k*σ：k 越大，判定带越宽 -> "中性"越多
        shares = [label_distribution(labels_volatility(df, k=k))["neutral"]
                  for k in (0.5, 0.8, 1.0)]

        self.assertLess(shares[0], shares[1])
        self.assertLess(shares[1], shares[2])

    def test_atr_scale_runs(self):
        df = frame(random_walk(800, sigma=0.004, seed=4))

        labels = labels_volatility(df, k=0.5, scale="atr")

        self.assertEqual(len(labels), len(df))
        self.assertGreater(label_distribution(labels)["neutral"], 0.0)

    def test_unknown_scale_rejected(self):
        with self.assertRaises(ValueError):
            labels_volatility(frame(random_walk(200)), scale="nope")


@unittest.skipUnless(HAS_PANDAS, SKIP_REASON)
class TestQuantileLabels(unittest.TestCase):
    def test_classes_are_balanced(self):
        df = frame(random_walk(3000, sigma=0.004, seed=5))

        dist = label_distribution(labels_quantile(df, window=500))

        for name, share in dist.items():
            self.assertGreater(share, 0.2, f"{name} 占比过低: {share}")
            self.assertLess(share, 0.45, f"{name} 占比过高: {share}")

    def test_early_rows_are_neutral_when_window_unfilled(self):
        df = frame(random_walk(300, sigma=0.004, seed=6))

        labels = labels_quantile(df, window=500)

        self.assertTrue(np.all(labels[:100] == 1), "历史窗口不足时不应给出方向标签")


@unittest.skipUnless(HAS_PANDAS, SKIP_REASON)
class TestRegistry(unittest.TestCase):
    def test_all_modes_produce_full_length_labels(self):
        df = frame(random_walk(900, sigma=0.004, seed=8))

        for mode in LABEL_MODES:
            with self.subTest(mode=mode):
                labels = get_label_fn(mode)(df)
                self.assertEqual(len(labels), len(df))
                self.assertTrue(set(np.unique(labels)).issubset({0, 1, 2}))

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            get_label_fn("Z_nope")


if __name__ == "__main__":
    unittest.main(verbosity=2)
