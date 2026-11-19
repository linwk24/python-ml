# LSTM 趋势预测服务

基于 LSTM 深度学习模型的加密货币趋势预测服务。

## 功能

- 趋势预测：看涨 / 中性 / 看跌
- 置信度分析
- 实时预测

## 快速开始

### 1. 安装依赖

```bash
cd python-ml
pip install -r requirements.txt
```

### 2. 启动服务

```bash
python app/main.py
# 或
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. 访问 API

- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

## API 接口

### 训练模型

```bash
POST /train
{
  "symbol": "BTCUSDT",
  "use_sample_data": true,
  "epochs": 50
}
```

### 预测趋势

```bash
POST /predict
{
  "symbol": "BTCUSDT",
  "use_sample_data": true
}
```

### 获取模型状态

```bash
GET /model/BTCUSDT
```

## 返回格式

```json
{
  "symbol": "BTCUSDT",
  "prediction": {
    "trend": "看涨",
    "trend_code": 2,
    "confidence": 75.5,
    "probabilities": {
      "下跌": 10.2,
      "持平": 14.3,
      "上涨": 75.5
    }
  },
  "current_price": 52000.00,
  "timestamp": "2024-01-15T10:30:00",
  "model_status": "ready"
}
```

## 与 Node.js 集成

在 Node.js 中调用 Python API：

```typescript
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ symbol: 'BTCUSDT', use_sample_data: true })
});
const result = await response.json();
```

## 注意事项

- 需要 Python 3.8+
- 首次使用需要训练模型
- 训练可能需要几分钟时间
