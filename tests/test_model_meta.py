"""
训练元数据与 in-sample 守卫的单元测试

覆盖的回归：离线评估/回测可以对着模型训练过的同一段行情跑，打印出漂亮但无意义的
准确率；现在这类重叠会被自动发现并中止（除非显式 --allow-in-sample）。

只用标准库。
"""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services import model_meta  # noqa: E402
from services.model_meta import (  # noqa: E402
    check_window_overlap,
    describe_meta,
    guard_in_sample_window,
    load_train_meta,
    save_train_meta,
    window_overlap,
)

DAY_MS = 86400000.0


def ms(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d").timestamp() * 1000


class MetaTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._orig_subdir = model_meta.MODEL_SUBDIR
        model_meta.MODEL_SUBDIR = self._tmp.name
        self.addCleanup(self._restore_subdir)
        self.symbol = "TESTUSDT"

    def _restore_subdir(self):
        model_meta.MODEL_SUBDIR = self._orig_subdir

    def write_meta(self, train_start="2026-01-01", train_end="2026-06-30", **extra):
        meta = {
            "rows": 4000,
            "train_start_ms": ms(train_start),
            "train_end_ms": ms(train_end),
            "val_loss": 1.1,
            "val_accuracy": 0.36,
            "epochs_run": 10,
        }
        meta.update(extra)
        return save_train_meta(self.symbol, meta)


class TestWindowOverlap(unittest.TestCase):
    def test_no_overlap(self):
        self.assertEqual(window_overlap(0, 100, 200, 300), 0.0)

    def test_partial_overlap(self):
        self.assertEqual(window_overlap(50, 200, 100, 300), 100.0)

    def test_contained(self):
        self.assertEqual(window_overlap(100, 200, 0, 1000), 100.0)

    def test_touching_is_not_overlap(self):
        self.assertEqual(window_overlap(0, 100, 100, 200), 0.0)


class TestTrainMetaStorage(MetaTestBase):
    def test_roundtrip(self):
        self.write_meta()

        meta = load_train_meta(self.symbol)

        self.assertIsNotNone(meta)
        self.assertEqual(meta["symbol"], self.symbol)
        self.assertEqual(meta["rows"], 4000)
        self.assertIn("created_at", meta)

    def test_missing_returns_none(self):
        self.assertIsNone(load_train_meta("NOSUCHSYMBOL"))

    def test_corrupt_returns_none(self):
        save_train_meta(self.symbol, {"rows": 1})
        with open(model_meta.train_meta_path(self.symbol), "w") as f:
            f.write("{not json")

        self.assertIsNone(load_train_meta(self.symbol))

    def test_describe_meta_mentions_window(self):
        self.write_meta()

        text = describe_meta(self.symbol)

        self.assertIn("2026-01-01", text)
        self.assertIn("2026-06-30", text)

    def test_describe_meta_without_file(self):
        self.assertIn("无训练元数据", describe_meta("NOSUCHSYMBOL"))


class TestCheckWindowOverlap(MetaTestBase):
    def test_unknown_when_no_meta(self):
        info = check_window_overlap(self.symbol, ms("2026-01-01"), ms("2026-02-01"))

        self.assertFalse(info["known"])
        self.assertFalse(info["overlaps"])
        self.assertIn("缺少训练窗口元数据", info["message"])

    def test_detects_overlap(self):
        self.write_meta(train_start="2026-01-01", train_end="2026-06-30")

        info = check_window_overlap(self.symbol, ms("2026-05-01"), ms("2026-08-01"))

        self.assertTrue(info["known"])
        self.assertTrue(info["overlaps"])
        self.assertAlmostEqual(info["overlap_days"], 60.0, places=1)
        self.assertIn("in-sample", info["message"])

    def test_clean_out_of_sample_window(self):
        self.write_meta(train_start="2026-01-01", train_end="2026-06-30")

        info = check_window_overlap(self.symbol, ms("2026-07-01"), ms("2026-09-01"))

        self.assertFalse(info["overlaps"])
        self.assertIn("out-of-sample", info["message"])

    def test_reports_training_metrics(self):
        self.write_meta()

        info = check_window_overlap(self.symbol, ms("2026-01-01"), ms("2026-02-01"))

        self.assertEqual(info["train_metrics"]["val_accuracy"], 0.36)


class TestGuardInSampleWindow(MetaTestBase):
    def test_blocks_overlapping_window_without_flag(self):
        self.write_meta()

        allowed = guard_in_sample_window(self.symbol, "2026-02-01", "2026-08-01")

        self.assertFalse(allowed, "重叠且未放行时必须中止")

    def test_allows_overlapping_window_with_flag(self):
        self.write_meta()

        allowed = guard_in_sample_window(self.symbol, "2026-02-01", "2026-08-01", allow_in_sample=True)

        self.assertTrue(allowed)

    def test_allows_out_of_sample_window(self):
        self.write_meta()

        allowed = guard_in_sample_window(self.symbol, "2026-07-01", "2026-09-01")

        self.assertTrue(allowed)

    def test_allows_when_meta_missing_but_warns(self):
        allowed = guard_in_sample_window(self.symbol, "2020-01-01", "2021-01-01")

        self.assertTrue(allowed, "无元数据时只能警告，不能阻断")

    def test_bad_date_format_does_not_block(self):
        self.write_meta()

        self.assertTrue(guard_in_sample_window(self.symbol, "not-a-date", "2026-01-01"))


class TestWriteFormat(MetaTestBase):
    def test_meta_is_json_readable(self):
        self.write_meta()

        with open(model_meta.train_meta_path(self.symbol)) as f:
            data = json.load(f)

        self.assertEqual(data["symbol"], self.symbol)
        self.assertIsInstance(data["train_start_ms"], float)


if __name__ == "__main__":
    unittest.main(verbosity=2)
