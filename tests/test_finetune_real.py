"""
真实 TensorFlow 微调路径的回归测试

这组测试需要 TensorFlow（仅装了 requirements.txt 的环境会跑它）。未安装时整体跳过，
因此默认的 stdlib 测试集仍然可跑。

背景（为什么需要它）:
    Keras 3 下 ``load_model('*.h5')`` 还原出的优化器，其内部变量表与模型不匹配，
    直接 ``fit()`` 会抛::

        ValueError: Unknown variable: <Variable path=sequential/lstm/lstm_cell/kernel ...>.
        This optimizer can only be called for the variables it was originally built with.

    凡是"线上加载模型 → 自我学习触发微调"的路径都会因此失败。
    修复方式是微调前用 ``compile()`` 重建优化器（保留权重、丢弃旧优化器状态）。
    本文件把这个行为钉住。

运行::

    .venv/bin/python -m unittest tests.test_finetune_real -v
"""

import json
import os
import random
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import numpy as np
    import tensorflow  # noqa: F401

    HAS_TF = True
    TF_SKIP_REASON = ""
except Exception as e:  # pragma: no cover - 取决于环境
    np = None
    HAS_TF = False
    TF_SKIP_REASON = f"TensorFlow 不可用: {e}"

SYMBOL = "ZZTEST"  # 专用符号，避免覆盖真实模型
BAR_MS = 3600 * 1000


def synthetic_klines(count=260, seed=7):
    """确定性随机游走，构造够长的 K 线（>= SEQUENCE_LENGTH + 10 + SEQUENCE_LENGTH）"""
    rng = random.Random(seed)
    price = 100.0
    now = 1_700_000_000_000
    bars = []
    for i in range(count):
        price *= 1 + rng.uniform(-0.01, 0.01)
        high = price * (1 + rng.uniform(0, 0.005))
        low = price * (1 - rng.uniform(0, 0.005))
        bars.append([now + i * BAR_MS, price, high, low, price, rng.uniform(1, 10)])
    return bars


@unittest.skipUnless(HAS_TF, TF_SKIP_REASON)
class TestFineTuneOnLoadedModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from models.enhanced_lstm import EnhancedLSTMPredictor

        cls.EnhancedLSTMPredictor = EnhancedLSTMPredictor
        cls.model_dir = os.path.join(PROJECT_ROOT, "models", "models")
        cls.klines = synthetic_klines()

    def setUp(self):
        # 每次测试前清掉该符号的历史产物
        self._cleanup()

    def tearDown(self):
        self._cleanup()

    def _cleanup(self):
        for name in os.listdir(self.model_dir) if os.path.isdir(self.model_dir) else []:
            if SYMBOL in name:
                try:
                    os.remove(os.path.join(self.model_dir, name))
                except OSError:
                    pass

    def test_fine_tune_after_load_model_succeeds(self):
        """回归：加载已保存模型后微调必须成功（此前 Keras 3 会抛 Unknown variable）"""
        from models.enhanced_lstm import FINE_TUNE_LEARNING_RATE

        # 1. 全量训练一次并落盘
        train_predictor = self.EnhancedLSTMPredictor(symbol=SYMBOL)
        train_predictor.train(self.klines, epochs=1, batch_size=32, is_fine_tune=False)
        self.assertTrue(os.path.exists(train_predictor.model_path), "模型未落盘")

        # 2. 模拟线上：新实例 + 从磁盘加载（self.model 不为 None）
        predictor = self.EnhancedLSTMPredictor(symbol=SYMBOL)
        self.assertTrue(predictor.load_model(), "模型加载失败")
        self.assertIsNotNone(predictor.model)

        weights_before = [w.copy() for w in predictor.model.get_weights()]

        # 3. 微调：这一步在修复前必定抛 ValueError
        history = predictor.train(self.klines, epochs=1, batch_size=32, is_fine_tune=True)
        self.assertIn("loss", history.history)

        # 4. 微调确实生效：学习率是微调值，且权重发生了变化（不是空跑）
        self.assertAlmostEqual(
            float(predictor.model.optimizer.learning_rate),
            FINE_TUNE_LEARNING_RATE,
            places=8,
        )
        weights_after = predictor.model.get_weights()
        self.assertEqual(len(weights_before), len(weights_after))
        changed = any(
            (a != b).any() for a, b in zip(weights_before, weights_after)
        )
        self.assertTrue(changed, "微调后权重未发生变化")

        # 5. 微调后仍可正常推理
        result = predictor.predict(self.klines[-120:])
        self.assertNotIn("error", result)
        self.assertIn(result["prediction"]["trend_code"], (0, 1, 2))

    def test_fine_tune_reloads_from_disk_after_training(self):
        """微调结果必须落盘，能被重新加载（闭环要靠它持久化）"""
        predictor = self.EnhancedLSTMPredictor(symbol=SYMBOL)
        predictor.train(self.klines, epochs=1, batch_size=32, is_fine_tune=False)
        predictor.train(self.klines, epochs=1, batch_size=32, is_fine_tune=True)

        reloaded = self.EnhancedLSTMPredictor(symbol=SYMBOL)
        self.assertTrue(reloaded.load_model(), "微调后的模型无法重新加载")
        self.assertNotIn("error", reloaded.predict(self.klines[-120:]))

    def test_full_training_still_works(self):
        """全量训练路径不能被微调修复影响"""
        predictor = self.EnhancedLSTMPredictor(symbol=SYMBOL)
        history = predictor.train(self.klines, epochs=1, batch_size=32, is_fine_tune=False)
        self.assertEqual(len(history.history["loss"]), 1)
        self.assertTrue(os.path.exists(predictor.scaler_path), "scaler 未落盘")
        with open(predictor.weights_path) as f:
            weights = json.load(f)
        self.assertEqual(len(weights), len(predictor.weight_analyzer.weights))

    def test_training_persists_normalization_stats(self):
        """训练必须固化特征顺序与标准化统计量，否则推理会退回错误口径"""
        predictor = self.EnhancedLSTMPredictor(symbol=SYMBOL)
        predictor.train(self.klines, epochs=1, batch_size=32, is_fine_tune=False)

        self.assertTrue(os.path.exists(predictor.analyzer_path), "权重分析器状态未落盘")

        reloaded = self.EnhancedLSTMPredictor(symbol=SYMBOL)
        self.assertTrue(reloaded.load_model())
        reloaded_analyzer = reloaded.weight_analyzer
        original_analyzer = predictor.weight_analyzer

        self.assertEqual(reloaded_analyzer.feature_order, original_analyzer.feature_order)
        self.assertEqual(
            set(reloaded_analyzer.feature_stats), set(original_analyzer.feature_stats)
        )
        for name, stats in original_analyzer.feature_stats.items():
            self.assertAlmostEqual(reloaded_analyzer.feature_stats[name]["mean"], stats["mean"])
            self.assertAlmostEqual(reloaded_analyzer.feature_stats[name]["std"], stats["std"])

    def test_default_decision_is_model_argmax(self):
        """默认不走状态机：方向必须等于 softmax 的 argmax，且标明 decision_source"""
        import numpy as np

        from config import TREND_MANAGER_ENABLED

        self.assertFalse(TREND_MANAGER_ENABLED, "默认应关闭状态机（实测 argmax 更好）")

        predictor = self.EnhancedLSTMPredictor(symbol=SYMBOL)
        predictor.train(self.klines, epochs=1, batch_size=32, is_fine_tune=False)
        result = predictor.predict(self.klines[-120:])

        payload = result["prediction"]
        self.assertEqual(payload["decision_source"], "model_argmax")
        probs = [payload["probabilities"][k] for k in ("看跌", "中性", "看涨")]
        self.assertEqual(payload["trend_code"], int(np.argmax(probs)))
        # 分离度诊断字段仍应给出，便于解释
        self.assertIsNotNone(payload["directional_score"])
        self.assertIsNone(payload["entry_price"], "无状态机时没有锚点价")

    def test_meta_records_fit_window_not_input_range(self):
        """回归：元数据里的 train_* 必须是**真正参与拟合**的窗口，而不是输入数据范围。

        早期版本记录的是全部输入 K 线的范围，会把真实的留出段误判成 in-sample，
        导致评估守卫对着样本外区间误报"重叠"。
        """
        from services.model_meta import load_train_meta

        bars = synthetic_klines(count=260, seed=9)
        predictor = self.EnhancedLSTMPredictor(symbol=SYMBOL)
        predictor.train(bars, epochs=1, batch_size=32, is_fine_tune=False)

        meta = load_train_meta(SYMBOL)
        self.assertIsNotNone(meta)
        fit_rows = meta["fit_rows"]
        self.assertIsNotNone(fit_rows)
        self.assertLess(fit_rows, len(bars), "拟合段必须短于全部输入")

        self.assertEqual(meta["train_start_ms"], bars[0][0])
        self.assertEqual(meta["train_end_ms"], bars[fit_rows - 1][0],
                         "train_end 应是最后一个拟合行，而非最后一行输入")
        self.assertEqual(meta["data_end_ms"], bars[-1][0])
        self.assertLess(meta["train_end_ms"], meta["data_end_ms"])

    def test_training_persists_score_distribution_and_calibration(self):
        """训练须固化方向分分布与标定阈值，供状态机开启时使用"""
        from services.model_meta import load_train_meta

        predictor = self.EnhancedLSTMPredictor(symbol=SYMBOL)
        predictor.train(self.klines, epochs=1, batch_size=32, is_fine_tune=False)

        meta = load_train_meta(SYMBOL)
        self.assertIsNotNone(meta)
        dist = meta["directional_score"]
        for key in ("min", "p10", "p50", "p90", "max", "mean"):
            self.assertIn(key, dist)
        self.assertLessEqual(dist["min"], dist["p50"])
        self.assertLessEqual(dist["p50"], dist["max"])

        cal = meta["calibrated_thresholds"]
        self.assertLess(cal["bearish"], cal["bullish"], "看跌阈值必须低于看涨阈值")
        self.assertGreaterEqual(cal["bearish"], 0.0)
        self.assertLessEqual(cal["bullish"], 100.0)
        self.assertEqual(meta["trend_decision"]["manager_enabled"], False)

    def test_new_training_marks_normalized_pipeline(self):
        """新训练的模型必须把输入管线标成 normalized；旧模型缺该字段则按 weighted 处理"""
        import pickle

        predictor = self.EnhancedLSTMPredictor(symbol=SYMBOL)
        predictor.train(self.klines, epochs=1, batch_size=32, is_fine_tune=False)

        with open(predictor.analyzer_path, "rb") as f:
            state = pickle.load(f)
        self.assertEqual(state.get("input_pipeline"), "normalized")

        reloaded = self.EnhancedLSTMPredictor(symbol=SYMBOL)
        self.assertTrue(reloaded.load_model())
        self.assertEqual(reloaded.weight_analyzer.input_pipeline, "normalized")

    def test_input_matrix_is_invariant_to_diagnostic_weights_when_normalized(self):
        """normalized 管线下，诊断权重不参与输入构造（历史那一步是无效的）"""
        import numpy as np

        predictor = self.EnhancedLSTMPredictor(symbol=SYMBOL)
        predictor.train(self.klines, epochs=1, batch_size=32, is_fine_tune=False)
        feats = predictor._compute_features(self._as_df(self.klines))

        baseline = predictor.build_input_matrix(feats)
        # 把诊断权重换成极端不均匀的值，输入矩阵不应改变
        predictor.weight_analyzer.weights = {
            f: 0.01 * (i + 1) for i, f in enumerate(predictor.weight_analyzer.feature_order)
        }
        perturbed = predictor.build_input_matrix(feats)

        np.testing.assert_allclose(baseline, perturbed, rtol=0, atol=1e-9)

    def test_prediction_rejects_too_few_klines(self):
        """K 线不足时必须给出明确的错误（含需要的根数），而不是静默用 0 填出来的假特征"""
        from models.enhanced_lstm import MIN_KLINES_FOR_PREDICT, SEQUENCE_LENGTH

        predictor = self.EnhancedLSTMPredictor(symbol=SYMBOL)
        predictor.train(self.klines, epochs=1, batch_size=32, is_fine_tune=False)

        # 刚够 60 步序列，但不足以预热 60 根滚动窗口
        result = predictor.predict(self.klines[: SEQUENCE_LENGTH + 5])

        self.assertIn("error", result)
        self.assertEqual(result["required"], MIN_KLINES_FOR_PREDICT)
        self.assertIn(str(MIN_KLINES_FOR_PREDICT), result["error"])

    def test_warmup_bars_prevent_zero_filled_features(self):
        """MIN_KLINES_FOR_PREDICT 必须足够让最后 60 行不再出现被 fillna(0) 填平的滚动特征"""
        from models.enhanced_lstm import FEATURE_WARMUP_BARS, MIN_KLINES_FOR_PREDICT

        predictor = self.EnhancedLSTMPredictor(symbol=SYMBOL)
        long_bars = synthetic_klines(count=MIN_KLINES_FOR_PREDICT + 200)
        short_bars = long_bars[-MIN_KLINES_FOR_PREDICT:]

        features_long = predictor._compute_features(self._as_df(long_bars)).tail(FEATURE_WARMUP_BARS)
        features_short = predictor._compute_features(self._as_df(short_bars)).tail(FEATURE_WARMUP_BARS)

        # 纯滚动窗口类特征：窗口长度 <= FEATURE_WARMUP_BARS，取值只依赖局部历史，
        # 因此长短两种输入的最后 60 行必须完全一致（一致即说明没有出现 NaN->0 的假特征）
        for column in ("ma_ratio_10_60", "ma_ratio_5_20", "bb_width", "cci", "ichimoku_kijun_sen"):
            with self.subTest(column=column):
                long_values = features_long[column].to_numpy()
                short_values = features_short[column].to_numpy()
                self.assertFalse(
                    (long_values == 0).all(), f"{column} 全为 0，说明预热不足"
                )
                np.testing.assert_allclose(long_values, short_values, rtol=1e-9, atol=1e-9)

    @staticmethod
    def _as_df(bars):
        import pandas as pd

        return pd.DataFrame(
            bars, columns=["open_time", "open", "high", "low", "close", "volume"]
        )


@unittest.skipUnless(HAS_TF, TF_SKIP_REASON)
class TestConfidenceSemantics(unittest.TestCase):
    """confidence 必须等于"所报方向"的概率，而不是 softmax 最大值"""

    def payload(self, probs, trend_code):
        from models.enhanced_lstm import build_prediction_payload

        return build_prediction_payload(probs, trend_code)

    def test_confidence_is_probability_of_reported_direction(self):
        # 最大概率落在"中性"(0.40)，但所报方向是看跌(0)
        payload = self.payload([0.35, 0.40, 0.25], trend_code=0)

        self.assertEqual(payload["confidence"], 35.0, "应是看跌类概率，而不是最大值 40")
        self.assertEqual(payload["max_probability"], 40.0)
        self.assertEqual(payload["max_probability_class"], "中性")

    def test_confidence_equals_max_when_direction_is_argmax(self):
        payload = self.payload([0.10, 0.20, 0.70], trend_code=2)

        self.assertEqual(payload["confidence"], 70.0)
        self.assertEqual(payload["max_probability"], 70.0)
        self.assertEqual(payload["max_probability_class"], "看涨")

    def test_the_original_bug_cannot_recur(self):
        """回归：旧实现直接返回 max(probabilities)，此时会报 46.5 而方向是看跌"""
        probs = [0.269, 0.465, 0.266]  # 实测出现过的分布
        payload = self.payload(probs, trend_code=0)

        self.assertEqual(payload["confidence"], 26.9)
        self.assertNotEqual(payload["confidence"], payload["max_probability"])

    def test_probabilities_are_numeric_and_sum_to_100(self):
        payload = self.payload([0.2, 0.3, 0.5], trend_code=2)

        self.assertEqual(sum(payload["probabilities"].values()), 100.0)
        for value in payload["probabilities"].values():
            self.assertIsInstance(value, float)
        self.assertEqual(payload["probabilities_pct"]["看涨"], "50.0%")

    def test_confidence_definition_is_declared(self):
        payload = self.payload([0.2, 0.3, 0.5], trend_code=2)

        self.assertEqual(payload["confidence_definition"], "probability_of_reported_direction")

    def test_unnormalized_probabilities_are_tolerated(self):
        payload = self.payload([1.0, 2.0, 1.0], trend_code=2)  # 和为 4

        self.assertEqual(payload["confidence"], 25.0)

    def test_softmax_max_may_be_checkable_against_direction(self):
        payload = self.payload([0.1, 0.6, 0.3], trend_code=0)

        self.assertLess(payload["confidence"], payload["max_probability"])
        self.assertEqual(payload["max_probability_class"], "中性")

    def test_invalid_inputs_are_rejected(self):
        from models.enhanced_lstm import build_prediction_payload

        with self.assertRaises(ValueError):
            build_prediction_payload([0.5, 0.5], 0)
        with self.assertRaises(ValueError):
            build_prediction_payload([0.5, 0.3, 0.2], 5)
        with self.assertRaises(ValueError):
            build_prediction_payload([0.0, 0.0, 0.0], 2)


@unittest.skipUnless(HAS_TF, TF_SKIP_REASON)
class TestNoDataLeakage(unittest.TestCase):
    """标准化器/权重/统计量只能拟合在训练段上"""

    def test_fit_scaler_uses_prefix_only(self):
        from sklearn.preprocessing import StandardScaler

        from models.enhanced_lstm import EnhancedLSTMPredictor

        rng = np.random.default_rng(0)
        weighted = np.vstack(
            [rng.normal(0, 1, (400, 17)), rng.normal(10, 3, (100, 17))]  # 后段分布明显不同
        )
        split = 400

        scaler = EnhancedLSTMPredictor.fit_scaler_on(weighted, split)

        np.testing.assert_allclose(scaler.mean_, weighted[:split].mean(axis=0), rtol=1e-12)
        # 关键：不能等于全量均值（那就是泄漏）
        self.assertFalse(np.allclose(scaler.mean_, weighted.mean(axis=0)))
        self.assertFalse(np.allclose(scaler.mean_, StandardScaler().fit(weighted).mean_))

    def test_fit_scaler_rejects_invalid_split(self):
        from models.enhanced_lstm import EnhancedLSTMPredictor

        weighted = np.zeros((10, 3))
        for bad in (0, -1, 10, 11):
            with self.subTest(split=bad):
                with self.assertRaises(ValueError):
                    EnhancedLSTMPredictor.fit_scaler_on(weighted, bad)

    def test_prepare_data_fits_stats_on_training_segment_only(self):
        from models.enhanced_lstm import SEQUENCE_LENGTH, EnhancedLSTMPredictor

        # 制造明显的分布切换：后 20% 价格水平与波动都不同
        bars = synthetic_klines(count=600, seed=5)
        for i in range(480, len(bars)):
            for col in (1, 2, 3, 4):
                bars[i][col] *= 3.0
            bars[i][5] *= 5.0

        predictor = EnhancedLSTMPredictor(symbol=SYMBOL)
        _, _, features_df, split_idx = predictor.prepare_data(bars, fit_ratio=0.8)
        fit_rows = split_idx + SEQUENCE_LENGTH

        for column in ("rsi", "bb_position", "ma_ratio_5_20"):
            with self.subTest(column=column):
                stats = predictor.weight_analyzer.feature_stats[column]
                prefix_mean = float(features_df[column].iloc[:fit_rows].mean())
                full_mean = float(features_df[column].mean())
                self.assertAlmostEqual(stats["mean"], prefix_mean, places=6)
                self.assertGreater(
                    abs(stats["mean"] - full_mean), 1e-3,
                    "统计量若与全量一致，说明用到了验证段（泄漏）",
                )

    def test_label_aligns_to_the_window_last_bar(self):
        """回归：标签必须是"窗口最后一根 -> 下一根"，不能跳过紧邻窗口的那一根。

        构造每根交替 ±2% 的序列，两种候选口径方向必然相反，可判定标签落在哪一根。
        """
        from models.enhanced_lstm import SEQUENCE_LENGTH, EnhancedLSTMPredictor

        n = 400
        closes = [100.0]
        for i in range(1, n):
            closes.append(closes[-1] * (1.02 if i % 2 == 0 else 0.98))
        bars = [
            [1_700_000_000_000 + i * 3600_000, closes[i], closes[i], closes[i], closes[i], 1.0]
            for i in range(n)
        ]

        predictor = EnhancedLSTMPredictor(symbol=SYMBOL)
        _, y, _, _ = predictor.prepare_data(bars, fit_ratio=0.8)

        def label(a, b):
            ch = (b - a) / a
            return 2 if ch > 0.001 else (0 if ch < -0.001 else 1)

        i = SEQUENCE_LENGTH  # 第一条序列对应的下标
        # 正确口径: 行 i-1 -> 行 i
        self.assertEqual(y[0], label(closes[i - 1], closes[i]))
        # 旧口径(行 i -> 行 i+1)方向相反，必须不成立
        self.assertNotEqual(y[0], label(closes[i], closes[i + 1]))

        ok_new = sum(1 for k in range(20) if y[k] == label(closes[i + k - 1], closes[i + k]))
        ok_old = sum(1 for k in range(20) if y[k] == label(closes[i + k], closes[i + k + 1]))
        self.assertEqual(ok_new, 20, "前 20 条序列都应吻合新口径")
        self.assertEqual(ok_old, 0, "旧口径应完全不吻合")

    def test_prepare_data_split_alignment(self):
        from models.enhanced_lstm import SEQUENCE_LENGTH, EnhancedLSTMPredictor

        bars = synthetic_klines(count=600, seed=6)
        predictor = EnhancedLSTMPredictor(symbol=SYMBOL)

        X, y, _, split_idx = predictor.prepare_data(bars, fit_ratio=0.8)

        self.assertEqual(len(X), len(y))
        self.assertGreater(split_idx, 0)
        self.assertLess(split_idx, len(X))
        # 训练/验证序列数之和等于总序列数（无重叠、无遗漏）
        self.assertEqual(split_idx + (len(X) - split_idx), len(X))
        self.assertEqual(split_idx, int(len(bars) * 0.8) - SEQUENCE_LENGTH)

    def test_train_writes_meta_with_window(self):
        from services.model_meta import load_train_meta

        bars = synthetic_klines(count=260, seed=7)
        predictor = self._train_quick(bars)
        self.addCleanup(self._cleanup_symbol_files)

        meta = load_train_meta(SYMBOL)

        self.assertIsNotNone(meta, "训练后必须写入训练窗口元数据")
        self.assertAlmostEqual(meta["train_start_ms"], bars[0][0], places=0)
        # train_end 是**拟合窗口**末端（不是最后一行输入），详见 fit_window 测试
        self.assertLessEqual(meta["train_end_ms"], bars[-1][0])
        self.assertAlmostEqual(meta["data_end_ms"], bars[-1][0], places=0)
        self.assertEqual(meta["rows"], len(bars))
        self.assertIsNotNone(meta["val_accuracy"])
        self.assertGreater(meta["train_sequences"], 0)
        self.assertGreater(meta["test_sequences"], 0)

    def _train_quick(self, bars):
        from models.enhanced_lstm import EnhancedLSTMPredictor

        predictor = EnhancedLSTMPredictor(symbol=SYMBOL)
        predictor.train(bars, epochs=1, batch_size=32, is_fine_tune=False)
        return predictor

    def _cleanup_symbol_files(self):
        model_dir = os.path.join(PROJECT_ROOT, "models", "models")
        for name in os.listdir(model_dir):
            if SYMBOL in name:
                try:
                    os.remove(os.path.join(model_dir, name))
                except OSError:
                    pass

if __name__ == "__main__":
    unittest.main(verbosity=2)
