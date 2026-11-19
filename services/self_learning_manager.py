import os
import logging
from services.prediction_tracker import PredictionTracker
from models.enhanced_lstm import EnhancedLSTMPredictor

# 配置日志
logging.basicConfig(
    filename='logs/self_self_learning.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SelfLearningManager:
    def __init__(self, symbol="BTCUSDT", accuracy_threshold=0.6):
        self.symbol = symbol
        self.accuracy_threshold = accuracy_threshold
        self.tracker = PredictionTracker(symbol=symbol)
        self.predictor = EnhancedLSTMPredictor(symbol=symbol)
        
        # 确保日志目录存在
        os.makedirs('logs', exist_ok=True)

    def run_self_learning_cycle(self, current_klines: list, current_price: float):
        """执行一次自我学习周期: 核对 -> 评估 -> 微调"""
        logging.info(f"开始 {self.symbol} 自我学习巡检...")
        
        # 1. 核对预测结果
        failed_samples = self.tracker.verify_predictions(self.symbol, current_price)
        
        # 获取核对统计信息
        with open(self.tracker.file_path, 'r') as f:
            all_data = json.load(f)
            total = len(all_data)
            verified = [d for d in all_data if d["actual"] is not None]
            verified_count = len(verified)
            correct = sum(1 for d in verified if d["trend_code"] == d["actual"])
            incorrect = verified_count - correct
            accuracy = correct / verified_count if verified_count > 0 else 0
        
        if not failed_samples:
            logging.info(f"核对完成: {verified_count}/{total} 条, 正确: {correct}, 错误: {incorrect}, 准确率: {accuracy:.2%}")
            return {
                "status": "skipped", 
                "message": "No errors found",
                "stats": {
                    "total_predictions": total,
                    "verified_count": verified_count,
                    "correct": correct,
                    "incorrect": incorrect,
                    "accuracy": f"{accuracy:.2%}"
                }
            }

        # 2. 计算近期准确率 (简单估算)
        # 获取最近 10 次预测的准确率
        with open(self.tracker.file_path, 'r') as f:
            all_data = json.load(f)
            recent_verified = [d for d in all_data if d["actual"] is not None][-10:]
            if not recent_verified:
                return {"status": "skipped", "message": "Insufficient verified data"}
            
            correct = sum(1 for d in recent_verified if d["trend_code"] == d["actual"])
            accuracy = correct / len(recent_verified)
            logging.info(f"近期准确率: {accuracy:.2%}")

        # 3. 判断是否触发学习
        if accuracy < self.accuracy_threshold:
            logging.warning(f"准确率 {accuracy:.2%} 低于阈值 {self.accuracy_threshold:.2%}，触发自我微调...")
            try:
                # 调用我们刚刚修改的 train 方法，设置 is_fine_tune=True
                self.predictor.train(current_klines, is_fine_tune=True)
                logging.info("自我微调训练完成，模型权重已更新。")
                return {"status": "learned", "accuracy": accuracy, "message": "Model fine-tuned successfully"}
            except Exception as e:
                logging.error(f"自我学习训练失败: {str(e)}")
                return {"status": "error", "message": str(e)}
        else:
            logging.info(f"准确率 {accuracy:.2%} 达标，无需微调。")
            return {"status": "stable", "accuracy": accuracy}

    def wrap_predict(self, recent_klines: list) -> dict:
        """包装预测方法，在预测的同时记录结果"""
        result = self.predictor.predict(recent_klines)
        if "error" not in result:
            self.tracker.record_prediction(
                symbol=self.symbol,
                current_price=result["current_price"],
                trend_code=result["prediction"]["trend_code"],
                confidence=result["prediction"]["confidence"]
            )
        return result
