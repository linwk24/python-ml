"""
自我学习闭环

一轮巡检 = 核对 → 评估 → 微调：

1. **核对**：用「预测时点 + 视界」之后的真实 K 线收盘价给历史预测打标签
   （详见 ``services.prediction_tracker``）。回看窗口已覆盖不到的旧预测标记
   ``expired``，不参与准确率统计 —— 绝不再拿"当前价"去审判几个月前的预测。
2. **评估**：取最近 ``SELF_LEARNING_WINDOW`` 条已核对样本计算准确率；
   已核对样本少于 ``SELF_LEARNING_MIN_SAMPLES`` 时不作任何判断，
   避免"1 错 1"就把模型拖去微调。
3. **微调**：仅当「本轮确实有新核对的样本」且「准确率低于阈值」时触发一次增量微调，
   防止每次巡检都在同一批旧样本上反复训练。

所有阈值来自 ``config``，本模块不再自带魔法数字。
"""

import logging
import os
from typing import Any, Dict, Optional

from config import (
    LOG_DIR,
    SELF_LEARNING_ACCURACY_THRESHOLD,
    SELF_LEARNING_MIN_SAMPLES,
    SELF_LEARNING_WINDOW,
)
from services.prediction_tracker import PredictionTracker, infer_horizon_minutes

# 日志目录必须先建好：此前 logging.basicConfig(filename='logs/...') 在目录不存在时
# 会在 import 阶段直接抛 FileNotFoundError，且依赖启动时的工作目录。
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("self_learning")
if not logger.handlers:
    _handler = logging.FileHandler(
        os.path.join(LOG_DIR, "self_learning.log"), encoding="utf-8"
    )
    _handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class SelfLearningManager:
    """单个交易对的自我学习闭环。"""

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        accuracy_threshold: float = SELF_LEARNING_ACCURACY_THRESHOLD,
        window: int = SELF_LEARNING_WINDOW,
        min_samples: int = SELF_LEARNING_MIN_SAMPLES,
        predictor: Optional[Any] = None,
        tracker: Optional[PredictionTracker] = None,
    ):
        self.symbol = symbol
        self.accuracy_threshold = accuracy_threshold
        self.window = window
        self.min_samples = min_samples
        self.tracker = tracker if tracker is not None else PredictionTracker(symbol=symbol)
        # predictor 可注入（测试用假模型，避免依赖 TensorFlow）
        self.predictor = predictor if predictor is not None else self._build_predictor()

    def _build_predictor(self):
        """延迟导入：让本模块在未安装 TensorFlow 的环境下也能被导入与测试。"""
        from models.enhanced_lstm import EnhancedLSTMPredictor

        return EnhancedLSTMPredictor(symbol=self.symbol)

    # ------------------------------------------------------------------ 闭环

    def run_self_learning_cycle(
        self, current_klines: list, current_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """执行一次自我学习周期：核对 → 评估 → 微调。"""
        logger.info(f"开始 {self.symbol} 自我学习巡检...")

        if current_price is None and current_klines:
            try:
                current_price = float(current_klines[-1][4])
            except (TypeError, ValueError, IndexError):
                current_price = None

        # 1. 核对（以真实 K 线为准，current_price 仅作兜底）
        verify_result = self.tracker.verify_predictions(
            current_price=current_price, klines=current_klines
        )
        logger.info(
            f"核对完成: 新核对 {verify_result['verified']} 条, "
            f"其中预测错误 {verify_result['failed']} 条, "
            f"标记过期 {verify_result['expired']} 条"
        )

        stats = self.tracker.stats()
        stats_payload = self._stats_payload(stats, verify_result)

        # 2. 评估：样本不足不做判断
        if stats["verified"] < self.min_samples:
            logger.info(
                f"已核对样本不足 ({stats['verified']}/{self.min_samples})，本轮不做判断。"
            )
            return {
                "status": "skipped",
                "message": f"Insufficient verified samples ({stats['verified']}/{self.min_samples})",
                "accuracy": None,
                "stats": stats_payload,
            }

        # 2b. 本轮没有新证据则跳过，避免在旧样本上反复触发微调
        if verify_result["verified"] == 0:
            logger.info("本轮没有新核对的样本，跳过评估。")
            return {
                "status": "skipped",
                "message": "No newly verified samples in this cycle",
                "accuracy": None,
                "stats": stats_payload,
            }

        accuracy = self.tracker.recent_accuracy(self.window)
        if accuracy is None:
            logger.info("近期准确率: 无可用的已核对样本")
        else:
            logger.info(f"近期准确率 (最近 {self.window} 条已核对样本): {accuracy:.2%}")

        # 3. 低于阈值才微调
        if accuracy is not None and accuracy < self.accuracy_threshold:
            logger.warning(
                f"准确率 {accuracy:.2%} 低于阈值 {self.accuracy_threshold:.2%}，触发自我微调..."
            )
            try:
                self.predictor.train(current_klines, is_fine_tune=True)
                logger.info("自我微调训练完成，模型权重已更新。")
                return {
                    "status": "learned",
                    "message": "Model fine-tuned successfully",
                    "accuracy": accuracy,
                    "stats": stats_payload,
                }
            except Exception as e:
                logger.error(f"自我学习训练失败: {e}")
                return {
                    "status": "error",
                    "message": str(e),
                    "accuracy": accuracy,
                    "stats": stats_payload,
                }

        if accuracy is None:
            logger.info("近期无可用的已核对样本，本轮不做判断。")
            return {
                "status": "skipped",
                "message": "No verified samples available for evaluation",
                "accuracy": None,
                "stats": stats_payload,
            }

        logger.info(f"准确率 {accuracy:.2%} 达标，无需微调。")
        return {
            "status": "stable",
            "message": "Accuracy above threshold",
            "accuracy": accuracy,
            "stats": stats_payload,
        }

    @staticmethod
    def _stats_payload(stats: Dict[str, Any], verify_result: Dict[str, Any]) -> Dict[str, Any]:
        """保持历史字段名的同时补充核对明细字段。"""
        return {
            # 历史字段（保持兼容）
            "total_predictions": stats["total"],
            "verified_count": stats["verified"],
            "correct": stats["correct"],
            "incorrect": stats["incorrect"],
            "accuracy": stats["accuracy_pct"],
            # 新增字段
            "pending": stats["pending"],
            "expired": stats["expired"],
            "newly_verified": verify_result["verified"],
            "newly_failed": verify_result["failed"],
            "newly_expired": verify_result["expired"],
            "avg_verify_lag_minutes": stats["avg_verify_lag_minutes"],
        }

    # ------------------------------------------------------------------ 预测

    def wrap_predict(self, recent_klines: list, interval: Optional[str] = None) -> Dict[str, Any]:
        """包装预测：预测的同时记录结果，供后续核对。

        ``interval`` 仅用于留档（如 "1h"）；核对所用的视界由 K 线时间戳自动推断，
        因此即使调用方不传 interval，闭环也能正确核对。
        """
        result = self.predictor.predict(recent_klines)
        if "error" not in result:
            self.tracker.record_prediction(
                symbol=self.symbol,
                current_price=result["current_price"],
                trend_code=result["prediction"]["trend_code"],
                confidence=result["prediction"]["confidence"],
                interval=interval,
                horizon_minutes=infer_horizon_minutes(recent_klines),
            )
        return result
