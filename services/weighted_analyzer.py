"""
指标权重分析器
基于过去6个月内每个指标对价格趋势的预测能力进行动态加权
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta


class IndicatorWeightAnalyzer:
    """指标权重分析器"""
    
    def __init__(self, lookback_days: int = 180):
        self.lookback_days = lookback_days  # 6个月
        self.weights: Dict[str, float] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self.initialized = False
    
    def evaluate_indicator_accuracy(
        self,
        indicator_values: np.ndarray,
        future_returns: np.ndarray,
        threshold: float = 0.02
    ) -> float:
        """评估单个指标的预测准确率"""
        if len(indicator_values) < 2 or len(future_returns) < 2:
            return 0.5
        
        n = min(len(indicator_values), len(future_returns))
        indicator_values = indicator_values[-n:]
        future_returns = future_returns[-n:]
        
        correct = 0
        total = 0
        
        # 生成信号
        normalized = (indicator_values - np.mean(indicator_values)) / np.std(indicator_values) if np.std(indicator_values) > 0 else indicator_values
        
        for i in range(1, len(normalized)):
            signal = 0
            if normalized[i] > threshold:
                signal = 1  # 看涨信号
            elif normalized[i] < -threshold:
                signal = -1  # 看跌信号
            
            if signal != 0:
                actual = 1 if future_returns[i] > 0 else -1
                if signal == actual:
                    correct += 1
                total += 1
        
        if total == 0:
            return 0.5
        
        return correct / total
    
    def evaluate_correlation(
        self,
        indicator_values: np.ndarray,
        future_returns: np.ndarray
    ) -> float:
        """评估指标与未来收益的相关性"""
        n = min(len(indicator_values), len(future_returns))
        if n < 10:
            return 0.0
        
        ind_vals = indicator_values[-n:]
        fut_ret = future_returns[-n:]
        
        # 归一化
        ind_norm = (ind_vals - np.mean(ind_vals)) / (np.std(ind_vals) + 1e-10)
        ret_norm = (fut_ret - np.mean(fut_ret)) / (np.std(fut_ret) + 1e-10)
        
        correlation = np.corrcoef(ind_norm, ret_norm)[0, 1]
        
        if np.isnan(correlation):
            return 0.0
        
        return abs(correlation)
    
    def evaluate_sharpe_ratio(
        self,
        indicator_values: np.ndarray,
        future_returns: np.ndarray,
        threshold: float = 0.02
    ) -> float:
        """基于指标信号的夏普比率"""
        n = min(len(indicator_values), len(future_returns))
        if n < 10:
            return 0.0
        
        ind_vals = indicator_values[-n:]
        fut_ret = future_returns[-n:]
        
        normalized = (ind_vals - np.mean(ind_vals)) / (np.std(ind_vals) + 1e-10)
        
        strategy_returns = []
        for i in range(1, len(normalized)):
            if normalized[i] > threshold:
                strategy_returns.append(fut_ret[i])
            elif normalized[i] < -threshold:
                strategy_returns.append(-fut_ret[i])
        
        if len(strategy_returns) < 5:
            return 0.0
        
        mean_ret = np.mean(strategy_returns)
        std_ret = np.std(strategy_returns) + 1e-10
        
        return mean_ret / std_ret * np.sqrt(365)  # 年化
    
    def calculate_weights(
        self,
        features_df: pd.DataFrame,
        future_returns: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """计算所有指标的权重"""
        weights = {}
        
        for feature in feature_names:
            if feature not in features_df.columns:
                continue
            
            values = features_df[feature].values
            
            # 综合评分
            accuracy = self.evaluate_indicator_accuracy(values, future_returns)
            correlation = self.evaluate_correlation(values, future_returns)
            sharpe = self.evaluate_sharpe_ratio(values, future_returns)
            
            # 归一化夏普比率到0-1范围
            normalized_sharpe = 1 / (1 + np.exp(-sharpe))
            
            # 加权综合得分
            score = (
                0.4 * accuracy +
                0.3 * correlation +
                0.3 * normalized_sharpe
            )
            
            weights[feature] = score
        
        return weights
    
    def update_weights(
        self,
        features_df: pd.DataFrame,
        future_returns: np.ndarray,
        feature_names: List[str]
    ):
        """更新权重并记录历史"""
        new_weights = self.calculate_weights(features_df, future_returns, feature_names)
        
        for feature, weight in new_weights.items():
            if feature not in self.performance_history:
                self.performance_history[feature] = []
            self.performance_history[feature].append(weight)
        
        self.weights = new_weights
        self.initialized = True
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要度排名（百分比权重相加=1）"""
        if not self.weights:
            return {}
        
        total = sum(self.weights.values())
        if total == 0:
            return {k: 1.0 / len(self.weights) for k in self.weights}
        
        return {k: v / total for k, v in self.weights.items()}
    
    def get_weighted_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """返回加权后的特征矩阵"""
        feature_importance = self.get_feature_importance()
        
        weighted_features = np.zeros((len(features_df), len(feature_importance)))
        
        for i, (feature, weight) in enumerate(feature_importance.items()):
            if feature in features_df.columns:
                values = features_df[feature].values
                # 归一化
                normalized = (values - np.mean(values)) / (np.std(values) + 1e-10)
                weighted_features[:, i] = normalized * weight
            else:
                weighted_features[:, i] = 0
        
        return weighted_features
    
    def get_top_indicators(self, n: int = 5) -> List[Tuple[str, float]]:
        """获取权重最高的 N 个指标"""
        sorted_weights = sorted(
            self.weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_weights[:n]


# 特征名称列表（全部11项技术指标的衍生特征）
FEATURE_NAMES = [
    'rsi',
    'macd_histogram',
    'bb_width',
    'bb_position',
    'ma_ratio_5_20',
    'ma_ratio_10_60',
    'adx',
    'cci',
    'williams_r',
    'stoch_k',
    'stoch_d',
    'mfi',
    'ichimoku_tenkan_sen',
    'ichimoku_kijun_sen',
    'price_change_pct',
    'high_low_ratio',
    'close_open_ratio'
]