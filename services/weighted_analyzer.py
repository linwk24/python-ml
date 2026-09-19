"""
指标权重分析器

两类职责要严格区分：

1. **标准化**（`get_normalized_features`）：逐列 z-score，统计量在训练段固化，
   推理端原样复用。这是模型输入真正依赖的部分。

2. **指标权重**（`get_feature_importance` / `apply_feature_weights`）：
   早期实现把权重乘在标准化**之前**（`(x-μ)/σ * w`），而下游紧接着是
   `StandardScaler` —— 后者逐列减均值、除标准差，会把任何非零的按列常数乘子
   **精确抵消**。实测：把权重从"近似均匀"换成"极端不均匀"，模型输入的最大差异只有
   3e-15（浮点噪声）。也就是说那一版加权对模型**完全没有影响**，纯属无效复杂度。

   现在权重只在**标准化之后**施加（`apply_feature_weights`）才真正生效；
   权重的来源也换成了训练段上的互信息（`fit_mutual_information`），
   而不是原来那套"准确率/相关性/夏普"打分 —— 后者输出几乎恒定，
   与真正有预测力的特征并不一致。
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

# 输入管线版本：
#   "normalized" = scaler 在"逐列标准化后的矩阵"上拟合（新，权重不再乘在标准化前）
#   "weighted"   = scaler 在"标准化*权重"的矩阵上拟合（历史版本）
# 两者数学上等价，但**必须先知道 scaler 是在哪种矩阵上拟合的**，否则会拿错尺度去变换。
INPUT_PIPELINE_NORMALIZED = "normalized"
INPUT_PIPELINE_WEIGHTED = "weighted"


class IndicatorWeightAnalyzer:
    """指标权重分析器"""
    
    def __init__(self, lookback_days: int = 180):
        self.lookback_days = lookback_days  # 6个月
        self.weights: Dict[str, float] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self.initialized = False
        # 训练期固化的特征顺序与标准化统计量（推理必须复用，见 fit_normalization）
        self.feature_order: List[str] = []
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        # 互信息（诊断用）与互信息权重（标准化之后施加，真正生效）
        self.feature_mi: Dict[str, float] = {}
        self.feature_weights: Dict[str, float] = {}
        # 输入管线版本；新训练一律写 normalized，旧模型缺该字段时按 weighted 处理
        self.input_pipeline: str = INPUT_PIPELINE_NORMALIZED
        self._warned_missing_stats = False
    
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

    # ------------------------------------------------------------------ 口径固化

    def fit_normalization(self, features_df: pd.DataFrame, feature_names: Optional[List[str]] = None) -> None:
        """在训练集上固化「特征顺序 + 每个特征的 mean/std」，供推理原样复用。

        为什么必须固化：``get_weighted_features`` 原来用**当前这批数据自己的** mean/std 做
        z-score。训练时喂的是几千根 K 线、推理时只喂最近 120 根，同一个特征值在两处会被
        标准化成完全不同的数值 —— 典型的 train/serve skew，模型看到的输入分布与训练时不一致。
        """
        names = feature_names if feature_names is not None else list(self.weights.keys())
        if features_df is None or len(features_df) == 0:
            raise ValueError("fit_normalization: 特征矩阵为空，无法固化标准化统计量")
        order = [f for f in names if f in features_df.columns]
        if not order:
            raise ValueError("fit_normalization: 没有任何可用的特征列")

        self.feature_order = order
        self.feature_stats = {}
        for feature in order:
            values = np.asarray(features_df[feature].values, dtype=float)
            self.feature_stats[feature] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

    def get_state(self) -> Dict[str, object]:
        """导出可持久化的状态"""
        return {
            "input_pipeline": self.input_pipeline,
            "weights": dict(self.weights),
            "feature_order": list(self.feature_order),
            "feature_stats": {k: dict(v) for k, v in self.feature_stats.items()},
            "feature_mi": dict(self.feature_mi),
            "feature_weights": dict(self.feature_weights),
            "initialized": self.initialized,
        }

    def set_state(self, state: Dict[str, object]) -> None:
        """恢复持久化状态"""
        if not isinstance(state, dict):
            raise ValueError("权重分析器状态格式不正确")
        self.weights = dict(state.get("weights") or {})
        self.feature_order = list(state.get("feature_order") or [])
        self.feature_stats = {
            k: dict(v) for k, v in (state.get("feature_stats") or {}).items()
        }
        self.feature_mi = dict(state.get("feature_mi") or {})
        self.feature_weights = dict(state.get("feature_weights") or {})
        # 旧状态没有该字段 -> 历史管线（scaler 拟合在"标准化*权重"上）
        self.input_pipeline = state.get("input_pipeline") or INPUT_PIPELINE_WEIGHTED
        self.initialized = bool(state.get("initialized", bool(self.weights)))

    def get_normalized_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """逐列标准化后的特征矩阵（列顺序固定为 ``feature_order``）。

        **不再乘指标权重** —— 那个乘法会被下游 StandardScaler 精确抵消（见模块说明）。
        没有持久化统计量的旧模型会退回"用当前窗口统计量"并打印一次警告，
        那正是需要重新训练的信号。
        """
        order = self.feature_order
        if not order:
            importance = self.get_feature_importance()
            if not importance:
                raise ValueError(
                    "特征权重未初始化（缺少权重文件或模型尚未训练），无法构造特征矩阵"
                )
            order = list(importance.keys())

        missing = [f for f in order if f not in features_df.columns]
        if missing:
            raise ValueError(f"特征缺少这些列: {missing}")

        matrix = np.zeros((len(features_df), len(order)))
        used_fallback = False

        for i, feature in enumerate(order):
            values = np.asarray(features_df[feature].values, dtype=float)
            stats = self.feature_stats.get(feature)
            if stats and float(stats.get("std", 0.0)) > 0:
                matrix[:, i] = (values - float(stats["mean"])) / float(stats["std"])
            else:
                used_fallback = True
                matrix[:, i] = (values - np.mean(values)) / (np.std(values) + 1e-10)

        if used_fallback and not self._warned_missing_stats:
            self._warned_missing_stats = True
            print(
                "警告: 权重分析器缺少训练期标准化统计量，已退回用当前窗口 mean/std —— "
                "训练/推理口径不一致，建议重新训练模型以固化统计量。"
            )

        return matrix

    def get_weighted_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """兼容别名，等价于 get_normalized_features"""
        return self.get_normalized_features(features_df)

    # ------------------------------------------------------------------ 真正生效的加权

    def fit_mutual_information(self, feature_values, labels, feature_names=None) -> None:
        """在**训练段**上计算每个特征与标签的互信息，并归一化到均值 1。

        归一化到均值 1 是为了在缩放特征的同时保持整体量级（否则整体尺度变化会与
        下游 scaler 的尺度纠缠在一起，难以解释）。
        """
        from sklearn.feature_selection import mutual_info_classif

        values = np.asarray(feature_values, dtype=float)
        labels = np.asarray(labels)
        if values.ndim != 2 or len(values) != len(labels):
            raise ValueError("fit_mutual_information: 特征与标签长度不一致")
        if len(np.unique(labels)) < 2:
            raise ValueError("fit_mutual_information: 标签只有一个类别，无法评估互信息")

        names = list(feature_names) if feature_names is not None else list(self.feature_order)
        mi = mutual_info_classif(values, labels, random_state=0)
        mi = np.asarray(mi, dtype=float)
        mean_mi = float(mi.mean())
        weights = mi / mean_mi if mean_mi > 0 else np.ones_like(mi)

        self.feature_mi = {name: float(v) for name, v in zip(names, mi)}
        self.feature_weights = {name: float(w) for name, w in zip(names, weights)}

    def apply_feature_weights(self, matrix: np.ndarray) -> np.ndarray:
        """把互信息权重作用到**标准化之后**的矩阵上。

        这是权重唯一能真正生效的位置：下游没有再对列做减均值/除标准差的操作。
        """
        if not self.feature_weights:
            raise ValueError("尚未计算互信息权重（请先调用 fit_mutual_information）")
        order = self.feature_order or list(self.feature_weights.keys())
        missing = [f for f in order if f not in self.feature_weights]
        if missing:
            raise ValueError(f"互信息权重缺少这些特征: {missing}")

        w = np.array([self.feature_weights[f] for f in order], dtype=float)
        if matrix.shape[1] != len(w):
            raise ValueError(
                f"特征维度不一致: 矩阵 {matrix.shape[1]} 列, 权重 {len(w)} 项"
            )
        return matrix * w
    
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