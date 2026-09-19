"""
增强版 LSTM 模型
结合11项技术指标 + 加权机制 + 6个月预测能力评估
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.indicators import TechnicalIndicators
from services.weighted_analyzer import IndicatorWeightAnalyzer, FEATURE_NAMES
from config import (
    FEATURE_WEIGHTING,
    LABEL_THRESHOLD,
    TREND_CALIBRATION_PERCENTILE,
    TREND_MANAGER_ENABLED,
    TREND_ABSOLUTE_THRESHOLD,
    TREND_BEARISH_THRESHOLD,
    TREND_BULLISH_THRESHOLD,
    TREND_PRICE_BREAK_PCT,
    TREND_SENSITIVE_ABSOLUTE_THRESHOLD,
    TREND_SENSITIVE_BEARISH_THRESHOLD,
    TREND_SENSITIVE_BULLISH_THRESHOLD,
    TREND_SENSITIVE_PRICE_BREAK_PCT,
)

# 模型参数
SEQUENCE_LENGTH = 60
FEATURE_DIM = len(FEATURE_NAMES)  # 17维特征
PREDICTION_HORIZON = 1
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# 最长滚动窗口（ma_ratio_10_60 需要 60 根；bb/cci 20 根；ichimoku 26 根）。
# 窗口未成熟时这些特征是 NaN，会被 _compute_features 里的 fillna(0) 填成 0 ——
# 因此喂进来的 K 线不够长时，模型输入序列中会混入一批恒为 0 的"假特征"。
FEATURE_WARMUP_BARS = 60
# 推理所需的最小 K 线数：60 步序列 + 60 根预热
MIN_KLINES_FOR_PREDICT = SEQUENCE_LENGTH + FEATURE_WARMUP_BARS

# 学习率：全量训练用 1e-3；微调用更小的 1e-4，避免破坏已学到的权重
TRAIN_LEARNING_RATE = 0.001
FINE_TUNE_LEARNING_RATE = 0.0001

TREND_LABELS = {0: "看跌", 1: "中性", 2: "看涨"}


def build_prediction_payload(
    probabilities,
    trend_code: int,
    trend: Optional[str] = None,
    trend_score: Optional[float] = None,
    entry_price: Optional[float] = None,
    model_bias: Optional[str] = None,
    directional_score: Optional[float] = None,
    required_score: Optional[float] = None,
    decision_source: Optional[str] = None,
) -> dict:
    """由 softmax 概率与最终方向组装对外返回的 prediction 块。

    **confidence 的语义**：最终所报方向对应的那类概率（probability of the reported
    direction），不再是 softmax 的最大值。原实现取 ``max(probabilities)``，而最大值
    可能落在"中性"类上 —— 实测出现过 ``trend=看跌, confidence=46.5%``，而 46.5% 恰好
    是中性类的概率，前端无从解读。

    最大值仍然保留，但改名 ``max_probability`` / ``max_probability_class``，
    让"模型最倾向哪一类"与"所报方向有多可信"两件事各自有名有姓。

    **directional_score / required_score**：状态机判定"中性"时，这两个字段说明原因 ——
    当前方向分离度是多少、需要多少才够给出方向。没有它们，"一直中性"很容易被误读成故障。
    """
    probs = [float(p) for p in probabilities]
    if len(probs) != 3:
        raise ValueError(f"probabilities 必须是 3 类概率，收到 {len(probs)} 个")
    if trend_code not in TREND_LABELS:
        raise ValueError(f"trend_code 必须是 0/1/2，收到 {trend_code}")

    total = sum(probs)
    if total <= 0:
        raise ValueError("probabilities 之和必须为正")
    normalized = [p / total for p in probs]  # 容忍未归一化的输入

    direction_probability = normalized[trend_code]
    max_index = int(np.argmax(normalized))
    bias = {"Bullish": "看涨", "Bearish": "看跌", "Neutral": "中性"}.get(model_bias, model_bias)
    by_label = {
        TREND_LABELS[i]: normalized[i] * 100 for i in range(3)
    }

    return {
        "trend": trend or TREND_LABELS[trend_code],
        "trend_code": int(trend_code),
        # 所报方向对应的概率（0-100）
        "confidence": round(direction_probability * 100, 2),
        "confidence_definition": "probability_of_reported_direction",
        # 原 confidence 语义（softmax 最大值），保留以便对比与排查
        "max_probability": round(normalized[max_index] * 100, 2),
        "max_probability_class": TREND_LABELS[max_index],
        "trend_score": round(float(trend_score), 2) if trend_score is not None else None,
        # 方向分离度(50=无方向)与"要说话需要多少分"：解释为什么现在是中性
        "directional_score": (
            round(float(directional_score), 2) if directional_score is not None else None
        ),
        "required_score": required_score,
        # 方向是谁定的："model_argmax"（默认）或 "trend_state_machine"
        "decision_source": decision_source,
        "model_bias": bias if bias is not None else TREND_LABELS[max_index],
        "entry_price": entry_price,
        # 概率统一为数值（与 models/lstm_model.py、README 一致），展示用百分比另附
        "probabilities": {label: round(v, 2) for label, v in by_label.items()},
        "probabilities_pct": {label: f"{v:.1f}%" for label, v in by_label.items()},
    }

class EnhancedLSTMPredictor:
    def __init__(self, symbol: str = "BTCUSDT", lookback_days: int = 180, sensitive: bool = False):
        self.symbol = symbol
        self.model = None
        self.scaler = None
        self.feature_importances: Dict[str, float] = {}
        self.weight_analyzer = IndicatorWeightAnalyzer(lookback_days=lookback_days)
        self.sensitive = sensitive  # 敏感模式：更及时的信号切换
        
        # 文件路径
        model_subdir = os.path.join(MODEL_DIR, "models")
        os.makedirs(model_subdir, exist_ok=True)
        
        self.model_path = os.path.join(model_subdir, f"{symbol}_enhanced_lstm.h5")
        self.scaler_path = os.path.join(model_subdir, f"{symbol}_enhanced_scaler.pkl")
        self.weights_path = os.path.join(model_subdir, f"{symbol}_feature_weights.json")
        # 权重分析器状态（权重 + 特征顺序 + 训练期标准化统计量）
        self.analyzer_path = os.path.join(model_subdir, f"{symbol}_weight_analyzer.pkl")
        
        # 加载已有权重
        self._load_weights()
    
    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算全部11项技术指标"""
        ti = TechnicalIndicators(df)
        
        features = pd.DataFrame(index=df.index)
        
        # 基本信息
        features['close_open_ratio'] = df['close'] / df['open']
        features['high_low_ratio'] = df['high'] / df['low']
        features['price_change_pct'] = df['close'].pct_change()
        
        # 1. RSI
        features['rsi'] = ti.rsi()
        
        # 2. MACD
        macd_line, signal_line, macd_hist = ti.macd()
        macd_hist = ti.safe_series(macd_hist, 'macd_hist')
        features['macd_histogram'] = macd_hist
        
        # 3. 布林带
        bb_upper, bb_middle, bb_lower = ti.bollinger_bands()
        bb_upper_s = ti.safe_series(bb_upper, 'bb_upper')
        bb_lower_s = ti.safe_series(bb_lower, 'bb_lower')
        bb_middle_s = ti.safe_series(bb_middle, 'bb_middle')
        
        bb_width = (bb_upper_s - bb_lower_s) / (bb_middle_s + 1e-10)
        bb_position = (df['close'] - bb_lower_s) / (bb_upper_s - bb_lower_s + 1e-10)
        features['bb_width'] = bb_width
        features['bb_position'] = bb_position

        # --- [方案 B: 引入波动率与动量特征] ---
        # 4. ATR (平均真实波幅) - 衡量波动强度
        # 简化实现：使用 (High-Low) 的滚动平均
        tr = df['high'] - df['low']
        features['atr'] = tr.rolling(window=14).mean()
        
        # 5. 波动率 (Rolling Std)
        features['volatility'] = df['close'].pct_change().rolling(window=20).std()
        
        # 6. 价格与 MA 的偏离度 (Distance from MA)
        ma20 = df['close'].rolling(window=20).mean()
        features['ma20_dist'] = (df['close'] - ma20) / (ma20 + 1e-10)
        # ---------------------------------------
        
        # 4. 移动平均线 (原有逻辑)
        ma5 = df['close'].rolling(window=5).mean()
        ma10 = df['close'].rolling(window=10).mean()
        ma20_val = df['close'].rolling(window=20).mean()
        ma60 = df['close'].rolling(window=60).mean()
        features['ma_ratio_5_20'] = ma5 / (ma20_val + 1e-10)
        features['ma_ratio_10_60'] = ma10 / (ma60 + 1e-10)

        
        # 5. ADX
        features['adx'] = ti.adx()
        
        # 6. CCI
        features['cci'] = ti.cci()
        
        # 7. Williams %R
        features['williams_r'] = ti.williams_r()
        
        # 8. 随机指标 (KDJ)
        stoch_k, stoch_d = ti.stochastic()
        features['stoch_k'] = ti.safe_series(stoch_k, 'stoch_k')
        features['stoch_d'] = ti.safe_series(stoch_d, 'stoch_d')
        
        # 9. MFI
        features['mfi'] = ti.mfi()
        
        # 10. 一目均衡图
        tenkan_sen, kijun_sen = ti.ichimoku()
        features['ichimoku_tenkan_sen'] = ti.safe_series(tenkan_sen, 'tenkan')
        features['ichimoku_kijun_sen'] = ti.safe_series(kijun_sen, 'kijun')
        
        # 填充缺失值
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        return features
    
    def compute_future_returns(self, df: pd.DataFrame, horizon: int = PREDICTION_HORIZON) -> np.ndarray:
        """计算未来 horizon 周期的收益"""
        future_prices = df['close'].shift(-horizon)
        current_prices = df['close']
        returns = (future_prices - current_prices) / (current_prices + 1e-10)
        return returns.fillna(0).values
    
    def build_input_matrix(self, features_df: pd.DataFrame) -> np.ndarray:
        """构造 LSTM 输入矩阵（训练与推理共用同一路径）。

        步骤：逐列标准化 -> （旧模型才需要）乘诊断权重 -> scaler -> （可选）互信息加权。

        为什么旧模型还要乘那一步：历史版本的 scaler 是在"标准化*权重"的矩阵上拟合的，
        虽然那一步在数学上会被 scaler 抵消，但**不能擅自去掉** —— 去掉就等于拿旧尺度去
        变换新矩阵，输入会被整体放大约 1/w ≈ 17 倍。新训练写 normalized 版本后不再需要它。
        """
        normalized = self.weight_analyzer.get_normalized_features(features_df)

        if self.weight_analyzer.input_pipeline != "normalized":
            importance = self.weight_analyzer.get_feature_importance()
            order = self.weight_analyzer.feature_order or list(importance.keys())
            w = np.array([float(importance.get(f, 0.0)) for f in order], dtype=float)
            if len(w) == normalized.shape[1]:
                normalized = normalized * w

        matrix = self.scaler.transform(normalized)

        if FEATURE_WEIGHTING == "mutual_info" and self.weight_analyzer.feature_weights:
            matrix = self.weight_analyzer.apply_feature_weights(matrix)

        return matrix

    @staticmethod
    def fit_scaler_on(weighted_features: np.ndarray, split_idx: int) -> StandardScaler:
        """只在训练段上拟合标准化器。

        原实现是 ``scaler.fit_transform(weighted_features)`` —— 用**含测试段的全量数据**
        拟合，测试集的均值/方差因此泄漏进训练管线，报告的测试指标偏乐观。
        split_idx 由 ``prepare_data`` 按同一时间切分给出，保证"拟合边界"与"训练/测试边界"
        完全对齐。
        """
        scaler = StandardScaler()
        if split_idx <= 0 or split_idx >= len(weighted_features):
            raise ValueError(f"无效的切分点: {split_idx}（样本数 {len(weighted_features)}）")
        scaler.fit(weighted_features[:split_idx])
        return scaler

    def prepare_data(
        self, klines: list, fit_ratio: float = 0.8, label_fn=None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[pd.DataFrame], int]:
        """准备训练数据。

        返回 (X, y, features_df, split_idx)，其中 ``split_idx`` 是训练序列数 —— 训练/测试
        边界与标准化器、指标权重、特征统计量的拟合边界都由它统一决定（避免泄漏）。

        时序切分：权重评估、标准化统计量、scaler 都**只用 split_idx 之前的数据**拟合，
        之后的数据只用于验证。

        ``label_fn``：可选的标签函数 ``f(df) -> np.ndarray``，返回**逐行**标签
        （index j = "行 j -> 行 j+1" 的涨跌类别）。默认 ``None`` 时使用生产口径
        （固定 ±0.1% 阈值，见 services/labeling.labels_fixed），行为与历史版本完全一致。
        传入自定义标签即可对比不同"预测目标"（如波动率归一化标签、分位数标签）。
        """
        # 支持 6 列或 7 列数据（7 列时忽略 close_time）
        if len(klines[0]) == 7:
            df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time'])
            df = df.drop(columns=['close_time'])
        else:
            df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 计算成交量相关指标（MFI需要）
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['money_flow'] = df['typical_price'] * df['volume']
        
        df = df.dropna()
        
        # 计算技术指标
        print(f"正在计算 {len(FEATURE_NAMES)} 项技术指标...")
        features_df = self._compute_features(df)
        
        # 计算未来收益（用于权重评估）
        future_returns = self.compute_future_returns(df)

        # 逐行标签：默认走生产口径；传入 label_fn 时用于对比不同目标
        if label_fn is None:
            from services.labeling import labels_fixed
            row_labels = labels_fixed(df)
            self.label_mode = "A_fixed_0.1pct"
        else:
            row_labels = np.asarray(label_fn(df))
            if len(row_labels) != len(df):
                raise ValueError(
                    f"label_fn 必须返回长度 {len(df)} 的逐行标签，实际 {len(row_labels)}"
                )
            self.label_mode = getattr(label_fn, "__name__", "custom")

        # ---- 时序切分：拟合只用前半段，后半段留作验证 ----
        total_rows = len(features_df)
        fit_rows = int(total_rows * fit_ratio)
        fit_rows = max(fit_rows, MIN_KLINES_FOR_PREDICT)
        fit_rows = min(fit_rows, total_rows - 1)
        self._last_fit_rows = fit_rows           # 供元数据记录"真正用于拟合的窗口"
        split_idx = fit_rows - SEQUENCE_LENGTH  # 序列下标与行下标的对齐
        if split_idx <= 0:
            raise ValueError(
                f"数据不足：需要至少 {MIN_KLINES_FOR_PREDICT + SEQUENCE_LENGTH} 条 K 线，"
                f"当前 {total_rows} 条"
            )

        train_features = features_df.iloc[:fit_rows]
        train_returns = future_returns[:fit_rows]
        print(
            f"时序切分: 拟合段 {fit_rows} 行(前 {fit_ratio:.0%}), 验证段 {total_rows - fit_rows} 行; "
            f"训练序列 {split_idx} 条, 验证序列 {len(features_df) - SEQUENCE_LENGTH - split_idx} 条"
        )

        # 权重评估与统计量固化都只用拟合段
        print("正在评估指标权重（仅基于拟合段，避免用到验证段信息）...")
        self.weight_analyzer.update_weights(train_features, train_returns, FEATURE_NAMES)
        self.feature_importances = self.weight_analyzer.get_feature_importance()

        # 固化特征顺序与标准化统计量：推理端必须原样复用，否则同一根 K 线会被
        # 训练期/推理期两套 mean/std 标准化成不同数值（train/serve skew）
        self.weight_analyzer.fit_normalization(train_features, FEATURE_NAMES)

        # 保存权重
        self._save_weights()
        self._save_analyzer()
        
        # 逐列标准化（统计量来自拟合段；不再乘指标权重，见 weighted_analyzer 模块说明）
        self.weight_analyzer.input_pipeline = "normalized"
        normalized_features = self.weight_analyzer.get_normalized_features(features_df)

        # 互信息权重：只在拟合段上评估，且只作用于标准化之后的矩阵才真正生效
        if FEATURE_WEIGHTING == "mutual_info":
            mi_rows = np.arange(SEQUENCE_LENGTH - 1, fit_rows - 1)
            if len(mi_rows) > 50:
                self.weight_analyzer.fit_mutual_information(
                    features_df.iloc[mi_rows][FEATURE_NAMES].to_numpy(),
                    row_labels[mi_rows],
                    FEATURE_NAMES,
                )
                print("已按拟合段互信息计算特征权重（标准化之后施加）")

        # 标准化：只在拟合段上 fit，避免测试集统计量泄漏进训练管线
        if self.scaler is None:
            self.scaler = self.fit_scaler_on(normalized_features, fit_rows)
        scaled_features = self.scaler.transform(normalized_features)
        if FEATURE_WEIGHTING == "mutual_info" and self.weight_analyzer.feature_weights:
            scaled_features = self.weight_analyzer.apply_feature_weights(scaled_features)
        
        # 创建序列
        X, y = [], []
        for i in range(SEQUENCE_LENGTH, len(scaled_features)):
            X.append(scaled_features[i-SEQUENCE_LENGTH:i])

            # 标签 = 窗口最后一根(i-1) -> 下一根(i)。
            #
            # 原实现用"行 i -> 行 i+1"，等于**跳过了紧邻窗口的那一根**：模型被训练去预测
            # "窗口之后第二根起"的涨跌，而线上接口与自我学习闭环都把同一个窗口的预测当作
            # "紧邻的下一根"来解读（记录价 = 窗口最后一根收盘，核对价 = 它的下一根）。
            # 训练目标与推理目标因此差了一根 K 线（实测确认：前 20 条序列 20/20 吻合旧口径）。
            #
            # 顺带消除一个边界泄漏：旧口径下最后一条训练序列的标签会用到 fit_rows 那一行，
            # 而它属于验证段；改为 i-1 -> i 之后，训练标签严格落在拟合段内。
            # 标签直接取自逐行标签数组（index i-1 = "行 i-1 -> 行 i"）
            y.append(int(row_labels[i - 1]))
        
        # 打印权重信息
        top_indicators = self.weight_analyzer.get_top_indicators(5)
        print(f"权重最高的5个指标:")
        for name, weight in top_indicators:
            print(f"  {name}: {weight:.4f}")
        
        return np.array(X), np.array(y), features_df, split_idx
    
    def compile_model(self, model, learning_rate: float = TRAIN_LEARNING_RATE):
        """按统一配置编译模型（全量训练与微调共用）"""
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def build_model(self, input_dim: int = FEATURE_DIM):
        """构建增强版 LSTM 模型"""
        model = Sequential([
            Input(shape=(SEQUENCE_LENGTH, input_dim)),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')
        ])

        return self.compile_model(model, TRAIN_LEARNING_RATE)
    
    def train(self, klines: list, epochs: int = 100, batch_size: int = 32, is_fine_tune: bool = False,
              fit_ratio: float = 0.8, label_fn=None):
        """训练增强版模型 (支持全量训练和微调)

        ``fit_ratio``：时序切分比例 —— 前 fit_ratio 的数据用于拟合（指标权重、统计量、
        标准化器），其余只用于验证。测试集信息不会进入训练管线。
        """
        print(f"\n{'='*50}")
        mode_str = "微调 (Fine-tuning)" if is_fine_tune else "全量训练 (Full Training)"
        print(f"{mode_str} {self.symbol} 增强版 LSTM 模型")
        print(f"{'='*50}")
        
        X, y, features_df, split_idx = self.prepare_data(klines, fit_ratio=fit_ratio, label_fn=label_fn)
        
        if len(X) < SEQUENCE_LENGTH + 10:
            raise ValueError(
                f"数据不足：至少需要 {MIN_KLINES_FOR_PREDICT + SEQUENCE_LENGTH} 条 K 线"
                f"（含 {FEATURE_WARMUP_BARS} 根特征预热与 {SEQUENCE_LENGTH} 步序列），"
                f"当前只构造出 {len(X)} 条训练序列"
            )
        if split_idx <= 0 or split_idx >= len(X):
            raise ValueError(
                f"时序切分失败: 训练序列 {split_idx} 条 / 合计 {len(X)} 条，"
                f"请提供更长的历史数据（当前 {len(klines)} 条）"
            )

        # 训练/测试边界直接采用 prepare_data 的切分点，保证与 scaler、指标权重、
        # 统计量的拟合边界完全一致（否则测试段信息会从这些环节漏进训练）
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"\n训练样本: {len(X_train)}, 测试样本: {len(X_test)} (时序切分，无重叠)")
        print(f"特征维度: {X.shape[2]}")
        
        # 构建模型或加载已有模型
        if not is_fine_tune or self.model is None:
            self.model = self.build_model(input_dim=X.shape[2])
        else:
            print("使用已有权重进行增量微调...")
            # 微调必须重建优化器：Keras 3 下 load_model() 还原出的优化器其变量表与模型
            # 不匹配，直接 fit 会抛
            #   "Unknown variable ... This optimizer can only be called for the variables
            #    it was originally built with."
            # 重新 compile 保留已加载的权重、丢弃旧优化器状态，并以更低学习率继续训练。
            self.compile_model(self.model, FINE_TUNE_LEARNING_RATE)
        
        # 回调
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15 if not is_fine_tune else 5,
            restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        # 训练 (微调时大幅减少 epoch)
        current_epochs = epochs if not is_fine_tune else min(epochs, 10)
        print(f"\n开始训练... (Epochs: {current_epochs})")
        
        # --- [方案 A: 实现代价敏感学习 (Class Weights)] ---
        # 计算每个类别的样本数量。注意 np.bincount 只覆盖 0..max(label)：
        # 某个类别在训练段缺失时 counts[i] 会是 0，直接做除法会得到 inf/nan 权重并污染训练。
        counts = np.bincount(y_train, minlength=3)
        total = len(y_train)
        class_weights = {}
        for label in (0, 1, 2):
            if counts[label] > 0:
                class_weights[label] = total / (len(counts) * int(counts[label]))
            else:
                print(f"警告: 训练段缺少类别 {label}，跳过其类别权重")
        # 额外增强：将涨跌的权重再提升 20%，强迫模型关注趋势
        for label in (0, 2):
            if label in class_weights:
                class_weights[label] *= 1.2
        print(f"类别分布: {dict(zip((0, 1, 2), counts.tolist()))}")
        print(f"计算类别权重: {class_weights}")
        
        history = self.model.fit(
            X_train, y_train,
            epochs=current_epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stop, reduce_lr],
            class_weight=class_weights,
            verbose=1
        )
        
        # 保存
        self.save_model()
        self._save_weights()

        # 记录训练窗口与切分信息：离线评估/回测据此判断是否为 in-sample 结果；
        # 同时统计方向分分布并标定状态机阈值（供 TREND_MANAGER_ENABLED=True 时使用）
        self._save_train_meta(klines, history, split_idx, len(X), fit_ratio,
                              score_info=self._score_distribution(X_train))

        return history
    
    def predict(self, recent_klines: list) -> dict:
        """预测趋势"""
        if self.model is None:
            if not self.load_model():
                return {
                    "error": "模型未训练，请先调用 /train 接口训练模型",
                    "symbol": self.symbol
                }
        
        # 准备数据 - 支持 6 列或 7 列数据（7 列时忽略 close_time）
        if len(recent_klines[0]) == 7:
            df = pd.DataFrame(recent_klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time'])
            df = df.drop(columns=['close_time'])
        else:
            df = pd.DataFrame(recent_klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['money_flow'] = df['typical_price'] * df['volume']
        
        df = df.dropna()
        
        if len(df) < MIN_KLINES_FOR_PREDICT:
            return {
                "error": (
                    f"数据不足：需要至少 {MIN_KLINES_FOR_PREDICT} 条 K 线"
                    f"（{SEQUENCE_LENGTH} 步序列 + {FEATURE_WARMUP_BARS} 根特征预热），"
                    f"当前仅 {len(df)} 条"
                ),
                "symbol": self.symbol,
                "current": len(df),
                "required": MIN_KLINES_FOR_PREDICT,
            }

        # 在**全部**可用 K 线上计算特征再取尾部：
        # 1) 滚动指标需要 60 根预热，只取 SEQUENCE_LENGTH*2 会让序列前段出现被 fillna(0)
        #    填平的假特征；
        # 2) RSI/MACD/ADX 用 ewm（无限记忆），上下文越长越接近训练时的取值。
        features_df = self._compute_features(df).tail(SEQUENCE_LENGTH)
        
        # 输入矩阵（与训练同一路径；旧模型兼容逻辑在 build_input_matrix 内）
        weighted = self.build_input_matrix(features_df)

        if weighted.shape[1] != FEATURE_DIM:
            return {
                "error": (
                    f"特征维度不一致：加权后 {weighted.shape[1]} 维，模型期望 {FEATURE_DIM} 维。"
                    f"通常是权重文件与模型不匹配（请重新训练），"
                    f"当前已固化特征顺序: {len(self.weight_analyzer.feature_order)} 项"
                ),
                "symbol": self.symbol,
            }

        scaled = weighted

        # 创建序列
        X = scaled.reshape(1, SEQUENCE_LENGTH, FEATURE_DIM)
        
        # 预测
        prediction = self.model.predict(X, verbose=0)[0]
        
        current_price = float(df['close'].iloc[-1])

        # 方向判定：默认直接用模型 argmax（实测优于任何状态机阈值配置）；
        # 状态机是可选平滑层，需显式打开 TREND_MANAGER_ENABLED。
        predicted_class = int(np.argmax(prediction))
        probs = prediction.tolist()
        # 方向分离度（50 = 完全无方向）：无论走不走状态机都给出，便于解释与排查
        directional_score = round(
            TrendManager.directional_score_static(probs) if TREND_MANAGER_ENABLED
            else 50.0 + 50.0 * (float(probs[2]) - float(probs[0])),
            2,
        )

        if not TREND_MANAGER_ENABLED:
            final_trend_code = predicted_class
            final_trend_str = TREND_LABELS[final_trend_code]
            trend_score = None
            entry_price = None
            model_bias = final_trend_str
            required_score = None
            decision_source = "model_argmax"
        else:
            try:
                from services.trend_manager import TrendManager
                # 为了保持状态，trend_manager 必须作为类成员变量
                if not hasattr(self, '_trend_manager'):
                    bull_thr, bear_thr = self._resolve_machine_thresholds()
                    if self.sensitive:
                        self._trend_manager = TrendManager(
                            bullish_threshold=TREND_SENSITIVE_BULLISH_THRESHOLD,
                            bearish_threshold=TREND_SENSITIVE_BEARISH_THRESHOLD,
                            price_break_pct=TREND_SENSITIVE_PRICE_BREAK_PCT,
                            absolute_threshold=TREND_SENSITIVE_ABSOLUTE_THRESHOLD,
                        )
                    else:
                        self._trend_manager = TrendManager(
                            bullish_threshold=bull_thr,
                            bearish_threshold=bear_thr,
                            price_break_pct=TREND_PRICE_BREAK_PCT,
                            absolute_threshold=TREND_ABSOLUTE_THRESHOLD,
                        )

                trend_res = self._trend_manager.update(current_price, probs)
                # 状态机三态 -> trend_code(0/1/2)：中性档必须能真正出现在输出里
                final_trend_code = trend_res["trend_code"]
                final_trend_str = TREND_LABELS[final_trend_code]
                trend_score = trend_res["trend_score"]
                entry_price = trend_res["entry_price"]
                model_bias = trend_res.get("model_bias")
                directional_score = trend_res.get("directional_score", directional_score)
                required_score = {
                    "bullish": round(self._trend_manager.bullish_threshold, 2),
                    "bearish": round(self._trend_manager.bearish_threshold, 2),
                }
                decision_source = "trend_state_machine"
            except Exception as e:
                print(f"TrendManager Error: {e}")
                final_trend_str = TREND_LABELS[predicted_class]
                final_trend_code = predicted_class
                trend_score = float(np.max(prediction) * 100)
                entry_price = None
                model_bias = TREND_LABELS[predicted_class]
            directional_score = None
            required_score = None
        # ----------------------------------------
        
        # 解析
        top_indicators = self.weight_analyzer.get_top_indicators(5)
        indicator_weights = [
            {"name": name, "weight": round(w, 4)}
            for name, w in top_indicators
        ]

        payload = build_prediction_payload(
            probabilities=prediction,
            trend_code=final_trend_code,
            trend=final_trend_str,
            trend_score=trend_score,
            entry_price=entry_price,
            model_bias=model_bias,
            directional_score=directional_score,
            required_score=required_score,
            decision_source=decision_source,
        )

        return {
            "symbol": self.symbol,
            "current_price": current_price,
            "model_type": "enhanced_lstm",
            "feature_dim": FEATURE_DIM,
            "prediction": payload,
            "feature_weights": indicator_weights,
            "model_status": "ready",
            "timestamp": datetime.now().isoformat()
        }
    
    def save_model(self):
        """保存模型、scaler、权重与权重分析器状态"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        self._save_weights()
        self._save_analyzer()
        print(f"模型已保存: {self.model_path}")
    
    def load_model(self):
        """加载模型"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = load_model(self.model_path)
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self._load_weights()
            self._load_analyzer()
            print(f"模型已加载: {self.model_path}")
            return True
        return False
    
    def _save_weights(self):
        """保存特征权重"""
        try:
            with open(self.weights_path, 'w') as f:
                json.dump(self.feature_importances, f)
        except Exception as e:
            print(f"保存权重失败: {e}")
    
    def _load_weights(self):
        """加载特征权重"""
        try:
            if os.path.exists(self.weights_path):
                with open(self.weights_path, 'r') as f:
                    self.feature_importances = json.load(f)
                if self.feature_importances:
                    self.weight_analyzer.weights = self.feature_importances
                    self.weight_analyzer.initialized = True
        except Exception as e:
            print(f"加载权重失败: {e}")

    def _save_analyzer(self):
        """保存权重分析器状态（含训练期固化的特征顺序与标准化统计量）"""
        try:
            with open(self.analyzer_path, 'wb') as f:
                pickle.dump(self.weight_analyzer.get_state(), f)
        except Exception as e:
            print(f"保存权重分析器状态失败: {e}")

    def _load_analyzer(self):
        """加载权重分析器状态；旧模型没有该文件时退回仅有权重的模式（会打印口径警告）"""
        try:
            if os.path.exists(self.analyzer_path):
                with open(self.analyzer_path, 'rb') as f:
                    self.weight_analyzer.set_state(pickle.load(f))
                self.feature_importances = self.weight_analyzer.get_feature_importance()
        except Exception as e:
            print(f"加载权重分析器状态失败: {e}")

    def _resolve_machine_thresholds(self):
        """状态机阈值：优先用训练时按模型自身分数分布标定出的分位数。

        写死的 70/30 是沿用旧评分尺度的产物，与现在 directional_score 的实际分布
        （实测 43.6~58.0）完全不匹配，会让状态机永远不发声。
        """
        try:
            from services.model_meta import load_train_meta

            meta = load_train_meta(self.symbol) or {}
            cal = meta.get("calibrated_thresholds") or {}
            bull = cal.get("bullish")
            bear = cal.get("bearish")
            if bull is not None and bear is not None and bear < bull:
                print(f"状态机阈值使用训练期标定值: bullish={bull:.2f}, bearish={bear:.2f}")
                return float(bull), float(bear)
        except Exception as e:  # pragma: no cover
            print(f"读取标定阈值失败: {e}")
        return TREND_BULLISH_THRESHOLD, TREND_BEARISH_THRESHOLD

    def _score_distribution(self, X_train):
        """在训练序列上统计方向分分布，并按分位数标定状态机阈值。

        这样"状态机开口的频率"由模型自身的分数分布决定，而不是一个与尺度脱节的常数。
        """
        try:
            probs = self.model.predict(X_train, verbose=0)
            scores = probs[:, 2] - probs[:, 0]
            scores = 50.0 + 50.0 * scores
            lo = float(np.percentile(scores, 100.0 - TREND_CALIBRATION_PERCENTILE))
            hi = float(np.percentile(scores, TREND_CALIBRATION_PERCENTILE))
            return {
                "directional_score": {
                    "min": float(np.min(scores)),
                    "p10": float(np.percentile(scores, 10)),
                    "p50": float(np.percentile(scores, 50)),
                    "p90": float(np.percentile(scores, 90)),
                    "max": float(np.max(scores)),
                    "mean": float(np.mean(scores)),
                },
                "calibrated_thresholds": {"bullish": hi, "bearish": lo,
                                          "percentile": TREND_CALIBRATION_PERCENTILE},
            }
        except Exception as e:  # pragma: no cover
            print(f"分数分布标定失败: {e}")
            return {}

    def _save_train_meta(self, klines, history, split_idx, total_sequences, fit_ratio, score_info=None):
        """写入训练窗口/切分/指标/分数标定，供评估与回测判断 in-sample。

        注意区分两个窗口：
          - ``fit_*``：**真正参与拟合**的行范围（前 fit_ratio），决定"哪些区间算 in-sample"；
          - ``data_*``：传入的全部 K 线范围（含留出段）。
        早期版本把 data 范围当成训练范围，会把真实留出段误判为 in-sample。
        """
        try:
            from services.model_meta import save_train_meta

            stamps = []
            for row in klines:
                try:
                    stamps.append(float(row[0]))
                except (TypeError, ValueError, IndexError):
                    continue
            hist = getattr(history, "history", {}) or {}
            fit_rows = getattr(self, "_last_fit_rows", None)
            fit_stamps = stamps[:fit_rows] if fit_rows else stamps
            meta = {
                "rows": len(klines),
                # train_* 语义 = 拟合窗口（评估守卫据此判断 in-sample）
                "train_start_ms": min(fit_stamps) if fit_stamps else None,
                "train_end_ms": max(fit_stamps) if fit_stamps else None,
                "fit_rows": fit_rows,
                "data_start_ms": min(stamps) if stamps else None,
                "data_end_ms": max(stamps) if stamps else None,
                "fit_ratio": fit_ratio,
                "train_sequences": int(split_idx),
                "test_sequences": int(total_sequences - split_idx),
                "epochs_run": len(hist.get("loss", [])),
                "epochs_requested": None,
                "final_loss": float(hist["loss"][-1]) if hist.get("loss") else None,
                "final_accuracy": float(hist["accuracy"][-1]) if hist.get("accuracy") else None,
                "val_loss": float(hist["val_loss"][-1]) if hist.get("val_loss") else None,
                "val_accuracy": float(hist["val_accuracy"][-1]) if hist.get("val_accuracy") else None,
                "label_threshold": LABEL_THRESHOLD,
                "sequence_length": SEQUENCE_LENGTH,
                "trend_decision": {
                    "manager_enabled": TREND_MANAGER_ENABLED,
                    "note": "状态机默认关闭，方向由模型 argmax 决定；开启时使用 calibrated_thresholds",
                },
                **(score_info or {}),
            }
            path = save_train_meta(self.symbol, meta)
            print(f"训练元数据已保存: {path}")
        except Exception as e:  # pragma: no cover - 不影响训练主流程
            print(f"保存训练元数据失败: {e}")
    
    def get_feature_importance_report(self) -> dict:
        """获取特征权重的详细报告"""
        # 将 numpy 类型转换为原生 Python 类型
        importance = {}
        for k, v in self.feature_importances.items():
            if hasattr(v, 'item'):
                importance[k] = float(v)
            else:
                importance[k] = v

        analyzer = self.weight_analyzer
        report = {
            "symbol": self.symbol,
            "feature_count": len(FEATURE_NAMES),
            # 注意：这些分数是**诊断用**的。历史版本把它们乘进模型输入，但那个位置会被
            # 下游 StandardScaler 精确抵消（见 services/weighted_analyzer 模块说明），
            # 因此它们并不影响预测结果。
            "feature_importance_role": "diagnostic_only",
            "feature_importance": importance,
            "top_features": [
                {"name": name, "weight": round(float(w) if hasattr(w, 'item') else w, 4)}
                for name, w in analyzer.get_top_indicators(5)
            ],
            "analyzer_ready": analyzer.initialized,
            "feature_weighting": FEATURE_WEIGHTING,
            "input_pipeline": analyzer.input_pipeline,
        }
        if analyzer.feature_mi:
            report["mutual_information"] = {
                k: round(float(v), 6) for k, v in
                sorted(analyzer.feature_mi.items(), key=lambda kv: -kv[1])
            }
            report["top_mutual_information"] = [
                {"name": k, "mi": round(float(v), 6)}
                for k, v in sorted(analyzer.feature_mi.items(), key=lambda kv: -kv[1])[:5]
            ]
            if FEATURE_WEIGHTING == "mutual_info":
                report["applied_feature_weights"] = {
                    k: round(float(v), 4) for k, v in analyzer.feature_weights.items()
                }
        return report