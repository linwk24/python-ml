"""
特征口径对齐（train/serve skew）的单元测试

覆盖两个真实缺陷：
1. ``get_weighted_features`` 原来用**当前这批数据自己的** mean/std 做 z-score。
   训练喂几千根、推理只喂最近百余根，同一根 K 线会被两套统计量标准化成不同数值。
2. 特征列顺序取决于 dict 顺序，权重文件缺失/错配时报错信息难以定位。

只需要 numpy/pandas（不需要 TensorFlow）。
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

    from services.weighted_analyzer import FEATURE_NAMES, IndicatorWeightAnalyzer

    HAS_NP_PD = True
    SKIP_REASON = ""
except Exception as e:  # pragma: no cover - 取决于环境
    HAS_NP_PD = False
    SKIP_REASON = f"numpy/pandas 不可用: {e}"


@unittest.skipUnless(HAS_NP_PD, SKIP_REASON)  # 子类继承该跳过标记
class FeatureAlignmentTestBase(unittest.TestCase):
    def build_features(self, rows=600, seed=3):
        rng = np.random.default_rng(seed)
        data = {name: rng.normal(loc=i, scale=1 + i * 0.1, size=rows) for i, name in enumerate(FEATURE_NAMES)}
        return pd.DataFrame(data, columns=FEATURE_NAMES)

    def fitted_analyzer(self, features):
        """模拟训练：算权重 + 固化特征顺序与标准化统计量"""
        analyzer = IndicatorWeightAnalyzer()
        future_returns = np.random.default_rng(11).normal(0, 0.01, size=len(features))
        analyzer.update_weights(features, future_returns, FEATURE_NAMES)
        analyzer.fit_normalization(features, FEATURE_NAMES)
        return analyzer


class TestNormalizationReuse(FeatureAlignmentTestBase):
    def test_fitted_stats_are_reused_regardless_of_window_length(self):
        """核心回归：同一根 K 线，无论喂全量还是只喂尾部，加权结果必须一致"""
        features = self.build_features()
        analyzer = self.fitted_analyzer(features)

        weighted_full = analyzer.get_weighted_features(features)[-60:]
        weighted_tail = analyzer.get_weighted_features(features.tail(60))

        np.testing.assert_allclose(weighted_full, weighted_tail, rtol=1e-12, atol=1e-12)

    def test_without_fitted_stats_results_differ(self):
        """反证：不固化统计量时（旧行为），两次结果不一致 —— 这正是被修掉的 skew"""
        features = self.build_features()
        analyzer = IndicatorWeightAnalyzer()
        analyzer.update_weights(
            features, np.random.default_rng(11).normal(0, 0.01, size=len(features)), FEATURE_NAMES
        )
        analyzer._warned_missing_stats = True  # 静音预期中的警告

        weighted_full = analyzer.get_weighted_features(features)[-60:]
        weighted_tail = analyzer.get_weighted_features(features.tail(60))

        self.assertFalse(
            np.allclose(weighted_full, weighted_tail),
            "旧行为下两次结果应当不同（说明原始实现确实存在口径不一致）",
        )

    def test_stats_survive_state_roundtrip(self):
        """持久化后必须完全复现，否则线上加载模型会退回错误口径"""
        features = self.build_features()
        analyzer = self.fitted_analyzer(features)
        expected = analyzer.get_weighted_features(features.tail(60))

        restored = IndicatorWeightAnalyzer()
        restored.set_state(analyzer.get_state())
        actual = restored.get_weighted_features(features.tail(60))

        np.testing.assert_allclose(expected, actual, rtol=1e-12, atol=1e-12)
        self.assertEqual(restored.feature_order, FEATURE_NAMES)
        self.assertEqual(len(restored.feature_stats), len(FEATURE_NAMES))

    def test_weights_survive_state_roundtrip(self):
        features = self.build_features()
        analyzer = self.fitted_analyzer(features)

        restored = IndicatorWeightAnalyzer()
        restored.set_state(analyzer.get_state())

        self.assertEqual(restored.get_feature_importance(), analyzer.get_feature_importance())
        self.assertTrue(restored.initialized)


class TestWeightingPlacement(FeatureAlignmentTestBase):
    """权重必须作用在标准化**之后**才生效。

    历史实现把权重乘在标准化之前，而下游 StandardScaler 逐列减均值除标准差，
    会把这个按列常数乘子精确抵消 —— 实测极端不均匀权重与均匀权重的模型输入差异
    仅 3e-15。这里把"抵消"与"生效"两个事实都钉住，防止有人再把权重加回错误的位置。
    """

    def test_pre_scaler_weights_are_cancelled(self):
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(1)
        raw = rng.normal(size=(300, 17)) * np.arange(1, 18)
        z = (raw - raw.mean(axis=0)) / (raw.std(axis=0) + 1e-10)

        uniform = StandardScaler().fit_transform(z * np.full(17, 1 / 17))
        wild = StandardScaler().fit_transform(z * np.array([0.01 * (i + 1) for i in range(17)]))

        np.testing.assert_allclose(uniform, wild, rtol=0, atol=1e-9)

    def test_post_scaler_weights_do_change_the_matrix(self):
        """MI 权重加在标准化之后 -> 必须真正改变矩阵（否则修复无意义）"""
        features = self.build_features(rows=400)
        analyzer = self.fitted_analyzer(features)
        labels = np.random.default_rng(5).integers(0, 3, size=len(features))

        analyzer.fit_mutual_information(features[FEATURE_NAMES].to_numpy(), labels, FEATURE_NAMES)
        matrix = analyzer.get_normalized_features(features)
        weighted = analyzer.apply_feature_weights(matrix)

        self.assertFalse(np.allclose(matrix, weighted))
        # 权重归一化到均值 1：整体量级不应被放大或缩小
        self.assertAlmostEqual(float(np.mean(list(analyzer.feature_weights.values()))), 1.0, places=8)

    def test_mi_weights_survive_state_roundtrip(self):
        features = self.build_features(rows=400)
        analyzer = self.fitted_analyzer(features)
        labels = np.random.default_rng(6).integers(0, 3, size=len(features))
        analyzer.fit_mutual_information(features[FEATURE_NAMES].to_numpy(), labels, FEATURE_NAMES)
        expected = analyzer.apply_feature_weights(analyzer.get_normalized_features(features.tail(30)))

        restored = IndicatorWeightAnalyzer()
        restored.set_state(analyzer.get_state())
        actual = restored.apply_feature_weights(restored.get_normalized_features(features.tail(30)))

        np.testing.assert_allclose(expected, actual, rtol=1e-12, atol=1e-12)
        self.assertEqual(restored.feature_mi, analyzer.feature_mi)

    def test_mi_rejects_single_class(self):
        features = self.build_features(rows=200)

        with self.assertRaises(ValueError):
            IndicatorWeightAnalyzer().fit_mutual_information(
                features[FEATURE_NAMES].to_numpy(), np.zeros(200, dtype=int), FEATURE_NAMES
            )

    def test_apply_weights_requires_fit(self):
        with self.assertRaises(ValueError):
            IndicatorWeightAnalyzer().apply_feature_weights(np.zeros((3, 17)))

    def test_input_pipeline_version_defaults_to_normalized_for_new_analyzer(self):
        self.assertEqual(IndicatorWeightAnalyzer().input_pipeline, "normalized")

    def test_legacy_state_without_pipeline_field_is_treated_as_weighted(self):
        """旧模型状态里没有该字段 -> 必须判定为历史管线（否则 scaler 尺度会用错）"""
        restored = IndicatorWeightAnalyzer()
        restored.set_state({"weights": {"rsi": 1.0}, "initialized": True})

        self.assertEqual(restored.input_pipeline, "weighted")


class TestFeatureOrderAndValidation(FeatureAlignmentTestBase):
    def test_feature_order_is_fixed_by_training(self):
        features = self.build_features()
        analyzer = self.fitted_analyzer(features)

        self.assertEqual(analyzer.feature_order, FEATURE_NAMES)
        self.assertEqual(analyzer.get_weighted_features(features).shape[1], len(FEATURE_NAMES))

    def test_column_mapping_follows_feature_order(self):
        """改单一特征列，输出矩阵里只有它对应的那一列发生变化（列序不错位）"""
        features = self.build_features(rows=120)
        analyzer = self.fitted_analyzer(features)

        target = "cci"
        idx = analyzer.feature_order.index(target)
        baseline = features.tail(5).copy()
        perturbed = baseline.copy()
        perturbed[target] = perturbed[target] + 5.0  # 只动目标列

        delta = analyzer.get_weighted_features(perturbed) - analyzer.get_weighted_features(baseline)

        changed_cols = set(np.nonzero(np.abs(delta).sum(axis=0) > 1e-12)[0])
        self.assertEqual(changed_cols, {idx})

    def test_missing_feature_column_raises_clear_error(self):
        features = self.build_features()
        analyzer = self.fitted_analyzer(features)
        broken = features.drop(columns=["rsi"])

        with self.assertRaises(ValueError) as ctx:
            analyzer.get_weighted_features(broken)

        self.assertIn("rsi", str(ctx.exception))

    def test_uninitialized_weights_raise_clear_error(self):
        features = self.build_features(rows=120)

        with self.assertRaises(ValueError) as ctx:
            IndicatorWeightAnalyzer().get_weighted_features(features)

        self.assertIn("权重", str(ctx.exception))

    def test_fit_normalization_rejects_empty_frame(self):
        empty = pd.DataFrame({name: [] for name in FEATURE_NAMES})

        with self.assertRaises(ValueError):
            IndicatorWeightAnalyzer().fit_normalization(empty, FEATURE_NAMES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
