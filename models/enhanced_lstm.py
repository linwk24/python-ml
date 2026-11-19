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

# 模型参数
SEQUENCE_LENGTH = 60
FEATURE_DIM = len(FEATURE_NAMES)  # 17维特征
PREDICTION_HORIZON = 1
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

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
    
    def prepare_data(self, klines: list) -> Tuple[np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
        """准备训练数据，返回 X, y, features_df"""
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
        
        # 更新权重（基于过去6个月的预测能力）
        print("正在评估指标权重（基于过去6个月预测能力）...")
        self.weight_analyzer.update_weights(features_df, future_returns, FEATURE_NAMES)
        self.feature_importances = self.weight_analyzer.get_feature_importance()
        
        # 保存权重
        self._save_weights()
        
        # 获取加权特征矩阵
        weighted_features = self.weight_analyzer.get_weighted_features(features_df)
        
        # 标准化
        if self.scaler is None:
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(weighted_features)
        else:
            scaled_features = self.scaler.transform(weighted_features)
        
        # 创建序列
        X, y = [], []
        for i in range(SEQUENCE_LENGTH, len(scaled_features)):
            X.append(scaled_features[i-SEQUENCE_LENGTH:i])
            
            # 标签：未来价格走势
            current_close = float(df['close'].iloc[i])
            future_close = float(df['close'].iloc[min(i + PREDICTION_HORIZON, len(df)-1)])
            
            change_pct = (future_close - current_close) / current_close
            
            if change_pct > 0.001:  # 将阈值从 0.002 降低到 0.001 (0.1%)
                y.append(2)  # 看涨
            elif change_pct < -0.001:  # 将阈值从 -0.002 降低到 -0.001 (-0.1%)
                y.append(0)  # 看跌
            else:
                y.append(1)  # 中性
        
        # 打印权重信息
        top_indicators = self.weight_analyzer.get_top_indicators(5)
        print(f"权重最高的5个指标:")
        for name, weight in top_indicators:
            print(f"  {name}: {weight:.4f}")
        
        return np.array(X), np.array(y), features_df
    
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
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, klines: list, epochs: int = 100, batch_size: int = 32, is_fine_tune: bool = False):
        """训练增强版模型 (支持全量训练和微调)"""
        print(f"\\n{'='*50}")
        mode_str = "微调 (Fine-tuning)" if is_fine_tune else "全量训练 (Full Training)"
        print(f"{mode_str} {self.symbol} 增强版 LSTM 模型")
        print(f"{'='*50}")
        
        X, y, features_df = self.prepare_data(klines)
        
        if len(X) < SEQUENCE_LENGTH + 10:
            raise ValueError(f"数据不足，需要至少 {SEQUENCE_LENGTH + 10} 条 K 线数据")
        
        # 划分训练集和测试集
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        print(f"\\n训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
        print(f"特征维度: {X.shape[2]}")
        
        # 构建模型或加载已有模型
        if not is_fine_tune or self.model is None:
            self.model = self.build_model(input_dim=X.shape[2])
        else:
            print("使用已有权重进行增量微调...")
            # 适当降低微调时的学习率以防止破坏已有权重
            self.model.optimizer.learning_rate = 0.0001 
        
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
        print(f"\\n开始训练... (Epochs: {current_epochs})")
        
        # --- [方案 A: 实现代价敏感学习 (Class Weights)] ---
        # 计算每个类别的样本数量
        counts = np.bincount(y_train)
        total = len(y_train)
        # 计算权重: 样本越少，权重越高 (Inverse Class Frequency)
        # 给看涨(2)和看跌(0)更高的权重，降低中性(1)的权重
        class_weights = {i: total / (len(counts) * counts[i]) for i in range(len(counts))}
        # 额外增强：将涨跌的权重再提升 20%，强迫模型关注趋势
        class_weights[0] *= 1.2
        class_weights[2] *= 1.2
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
        
        
        # 保存模型和特征权重
        self.save_model()
        self._save_weights()

        # 自动同步到 Cloudflare R2
        try:
            from services.model_storage import upload_models

            result = upload_models()

            if result["status"] == "success":
                print(
                    f"Cloudflare R2 同步成功："
                    f"{result['uploaded']} 个模型文件"
                )
            else:
                print("Cloudflare R2 未配置，保留本地模型")

        except Exception as exc:
            print(f"Cloudflare R2 上传失败：{exc}")

            # 生产环境要求持久化时，不允许静默忽略
            if os.getenv("R2_REQUIRED", "false").lower() == "true":
                raise

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
        
        if len(df) < SEQUENCE_LENGTH:
            return {
                "error": f"数据不足，需要至少 {SEQUENCE_LENGTH} 条 K 线数据",
                "symbol": self.symbol,
                "current": len(df),
                "required": SEQUENCE_LENGTH
            }
        
        # 计算特征
        features_df = self._compute_features(df.tail(SEQUENCE_LENGTH * 2))
        
        # 取最后 SEQUENCE_LENGTH 条
        features_df = features_df.tail(SEQUENCE_LENGTH)
        
        # 加权
        weighted = self.weight_analyzer.get_weighted_features(features_df)
        
        # 标准化
        scaled = self.scaler.transform(weighted)
        
        # 创建序列
        X = scaled.reshape(1, SEQUENCE_LENGTH, FEATURE_DIM)
        
        # 预测
        prediction = self.model.predict(X, verbose=0)[0]
        
        current_price = float(df['close'].iloc[-1])

        # --- [新引入：趋势状态机过滤信号抖动] ---
        try:
            from services.trend_manager import TrendManager
            # 为了保持状态，trend_manager 必须作为类成员变量
            if not hasattr(self, '_trend_manager'):
                # 敏感模式：更及时的信号切换
                if self.sensitive:
                    self._trend_manager = TrendManager(
                        bullish_threshold=60.0,      # 降低看涨阈值（70→60）
                        price_break_pct=0.005,       # 降低价格突破阈值（1%→0.5%）
                        absolute_threshold=75.0      # 降低绝对阈值（85→75）
                    )
                else:
                    self._trend_manager = TrendManager(bullish_threshold=70.0)
            
            trend_res = self._trend_manager.update(current_price, prediction.tolist())
            final_trend_str = "看涨" if trend_res["final_trend"] == 'Bullish' else "看跌"
            final_trend_code = 2 if trend_res["final_trend"] == 'Bullish' else 0
            trend_score = trend_res["trend_score"]
            entry_price = trend_res["entry_price"]
        except Exception as e:
            print(f"TrendManager Error: {e}")
            # 降级到原逻辑
            predicted_class = int(np.argmax(prediction))
            final_trend_str = ['看跌', '中性', '看涨'][predicted_class]
            final_trend_code = predicted_class
            trend_score = float(np.max(prediction) * 100)
            entry_price = None
        # ----------------------------------------
        
        # 解析
        trend_labels = ['看跌', '中性', '看涨']
        
        # 注意：这里我们将原来基于 argmax 的结果替换为状态机给出的最终结果
        predicted_class = final_trend_code
        
        # 使用模型的原始最大概率作为置信度（更能反映模型真实信心）
        model_confidence = float(np.max(prediction) * 100)
        # 同时保留 trend_score 用于参考
        confidence = model_confidence / 100.0 # 转换为 0-1 概率以保持兼容性为 0-1 概率以保持兼容性
        
        # 返回指标权重信息
        top_indicators = self.weight_analyzer.get_top_indicators(5)
        indicator_weights = [
            {"name": name, "weight": round(w, 4)}
            for name, w in top_indicators
        ]
        
        return {
            "symbol": self.symbol,
            "current_price": current_price,
            "model_type": "enhanced_lstm",
            "feature_dim": FEATURE_DIM,
            "prediction": {
                "trend": final_trend_str,
                "trend_code": final_trend_code,
                "confidence": round(model_confidence, 2),  # 使用模型原始最大概率
                "trend_score": round(trend_score, 2),  # 保留 trend_score 用于参考
                "entry_price": entry_price,
                "probabilities": {
                    "看跌": f"{prediction[0]*100:.1f}%",
                    "中性": f"{prediction[1]*100:.1f}%",
                    "看涨": f"{prediction[2]*100:.1f}%"
                }
            },
            "feature_weights": indicator_weights,
            "model_status": "ready",
            "timestamp": datetime.now().isoformat()
        }
    
    def save_model(self):
        """保存模型和权重"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        self._save_weights()
        print(f"模型已保存: {self.model_path}")
    
    def load_model(self):
        """加载模型"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = load_model(self.model_path)
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self._load_weights()
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
    
    def get_feature_importance_report(self) -> dict:
        """获取特征权重的详细报告"""
        # 将 numpy 类型转换为原生 Python 类型
        importance = {}
        for k, v in self.feature_importances.items():
            if hasattr(v, 'item'):
                importance[k] = float(v)
            else:
                importance[k] = v

        return {
            "symbol": self.symbol,
            "feature_count": len(FEATURE_NAMES),
            "feature_importance": importance,
            "top_features": [
                {"name": name, "weight": round(float(w) if hasattr(w, 'item') else w, 4)}
                for name, w in self.weight_analyzer.get_top_indicators(5)
            ],
            "analyzer_ready": self.weight_analyzer.initialized
        }