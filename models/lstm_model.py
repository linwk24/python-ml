"""
LSTM 模型用于加密货币趋势预测
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import pickle
from datetime import datetime

from config import LABEL_THRESHOLD

# 模型参数
SEQUENCE_LENGTH = 60  # 使用过去60个时间步
FEATURE_DIM = 5       # 特征维度: open, high, low, close, volume
PREDICTION_HORIZON = 1  # 预测未来1个时间步
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

class LSTMPredictor:
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.model = None
        self.scaler = None
        self.model_path = os.path.join(MODEL_DIR, f"models/{symbol}_lstm_model.h5")
        self.scaler_path = os.path.join(MODEL_DIR, f"models/{symbol}_scaler.pkl")
        
    def prepare_data(self, klines: list) -> tuple:
        """准备训练数据"""
        # 转换为 DataFrame
        df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time'])
        
        # 确保数值类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 删除无效行
        df = df.dropna()
        
        # 特征列
        features = df[['open', 'high', 'low', 'close', 'volume']].values
        
        # 标准化
        if self.scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_features = self.scaler.fit_transform(features)
        else:
            scaled_features = self.scaler.transform(features)
        
        # 创建序列
        X, y = [], []
        for i in range(SEQUENCE_LENGTH, len(scaled_features)):
            X.append(scaled_features[i-SEQUENCE_LENGTH:i])
            # 预测下一个时间步的收盘价涨跌
            current_price = features[i, 3]  # close price
            next_price = features[i-1, 3] if i > 0 else current_price
            # 0: 下跌, 1: 持平, 2: 上涨（阈值与核对端统一取自 config.LABEL_THRESHOLD）
            if current_price > next_price * (1 + LABEL_THRESHOLD):  # 涨超过阈值
                y.append(2)
            elif current_price < next_price * (1 - LABEL_THRESHOLD):  # 跌超过阈值
                y.append(0)
            else:
                y.append(1)
        
        return np.array(X), np.array(y)
    
    def build_model(self):
        """构建 LSTM 模型"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(SEQUENCE_LENGTH, FEATURE_DIM)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(3, activation='softmax')  # 3个类别: 跌, 持平, 涨
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, klines: list, epochs: int = 50, batch_size: int = 32):
        """训练模型"""
        print(f"正在准备 {self.symbol} 的训练数据...")
        X, y = self.prepare_data(klines)
        
        if len(X) < SEQUENCE_LENGTH + 10:
            raise ValueError(f"数据不足，需要至少 {SEQUENCE_LENGTH + 10} 条 K 线数据")
        
        # 划分训练集和测试集
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        print(f"训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
        
        # 构建模型
        self.model = self.build_model()
        
        # 早停
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # 训练
        print("开始训练 LSTM 模型...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=1
        )
        
        # 保存模型和 scaler
        self.save_model()
        
        return history
    
    def save_model(self):
        """保存模型"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        
        self.model.save(self.model_path)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"模型已保存: {self.model_path}")
    
    def load_model(self):
        """加载模型"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = load_model(self.model_path)
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"模型已加载: {self.model_path}")
            return True
        return False
    
    def predict(self, recent_klines: list) -> dict:
        """预测趋势"""
        if self.model is None:
            if not self.load_model():
                return {
                    "error": "模型未训练，请先调用 /train 接口训练模型",
                    "symbol": self.symbol
                }
        
        # 准备预测数据
        df = pd.DataFrame(recent_klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time'])
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        if len(df) < SEQUENCE_LENGTH:
            return {
                "error": f"数据不足，需要至少 {SEQUENCE_LENGTH} 条 K 线数据",
                "symbol": self.symbol
            }
        
        # 只取最后 SEQUENCE_LENGTH 条
        df = df.tail(SEQUENCE_LENGTH)
        features = df[['open', 'high', 'low', 'close', 'volume']].values
        
        # 标准化
        scaled_features = self.scaler.transform(features)
        
        # 创建序列
        X = scaled_features.reshape(1, SEQUENCE_LENGTH, FEATURE_DIM)
        
        # 预测
        prediction = self.model.predict(X, verbose=0)[0]
        
        # 解析结果
        labels = ['下跌', '持平', '上涨']
        trend_labels = ['看跌', '中性', '看涨']
        
        predicted_class = np.argmax(prediction)
        confidence = float(prediction[predicted_class])
        
        # 当前价格
        current_price = float(df['close'].iloc[-1])
        
        return {
            "symbol": self.symbol,
            "prediction": {
                "trend": trend_labels[predicted_class],
                "trend_code": int(predicted_class),
                "confidence": round(confidence * 100, 2),
                "probabilities": {
                    "下跌": round(float(prediction[0]) * 100, 2),
                    "持平": round(float(prediction[1]) * 100, 2),
                    "上涨": round(float(prediction[2]) * 100, 2)
                }
            },
            "current_price": current_price,
            "timestamp": datetime.now().isoformat(),
            "model_status": "ready"
        }


def get_sample_klines(symbol: str, count: int = 200) -> list:
    """获取样本 K 线数据（用于演示）"""
    import random
    from datetime import datetime, timedelta
    
    # 生成模拟数据
    base_price = 50000 if symbol == "BTCUSDT" else 3000
    klines = []
    current_time = int(datetime.now().timestamp() * 1000) - count * 60000
    
    for i in range(count):
        change = random.uniform(-0.02, 0.02)
        base_price *= (1 + change)
        
        high = base_price * random.uniform(1.001, 1.02)
        low = base_price * random.uniform(0.98, 0.999)
        open_price = random.uniform(low, high)
        close_price = random.uniform(low, high)
        volume = random.uniform(100, 1000)
        
        klines.append([
            current_time,
            float(open_price),
            float(high),
            float(low),
            float(close_price),
            float(volume),
            current_time + 60000
        ])
        current_time += 60000
    
    return klines
