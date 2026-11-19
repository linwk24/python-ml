import numpy as np
from typing import Optional

class TrendManager:
    """
    极致优化版趋势状态机：价格优先 + 信号增强 (方案一 + 方案二)
    核心逻辑：
    1. 信号增强：通过【价格动量权重】和【非线性放大】解决模型在 40-60 分徘徊的问题。
    2. 进入价 (Entry Price)：状态锁，防止频繁抖动。
    3. 三级切换：
       - Level 1 [绝对重置]: 增强后分数 >= 85 或 <= 15 -> 强制重置。
       - Level 2 [动态起涨]: 距离锚点太远且分数回升 -> 认作新趋势起点。
       - Level 3 [价格驱动]: 价格破位 + 分数合理 -> 切换状态。
    """
    def __init__(self, bullish_threshold: float = 70.0, price_break_pct: float = 0.01, absolute_threshold: float = 85.0):
        self.bullish_threshold = bullish_threshold
        self.price_break_pct = price_break_pct
        self.absolute_threshold = absolute_threshold
        self.current_state = None
        self.entry_price = None
        self.price_history = [] # 用于计算动量

    def _calculate_momentum_bias(self, current_price: float) -> float:
        """
        方案二：价格动量权重
        分析近期价格走势，为趋势分提供方向性修正
        """
        self.price_history.append(current_price)
        if len(self.price_history) > 24: # 观察过去 24 小时
            self.price_history.pop(0)
        
        if len(self.price_history) < 10:
            return 0.0 # 数据不足，不修正
            
        # 计算简单斜率 (当前价 - 起始价) / 价格
        start_price = self.price_history[0]
        momentum = (current_price - start_price) / start_price
        
        # 将动量映射到分数修正值 (-10 到 +10)
        # 假设 2% 的 24h 涨幅即为强动量
        bias = np.clip(momentum / 0.02, -10, 10)
        return bias

    def _amplify_score(self, score: float) -> float:
        """
        方案一：非线性分数放大 (S-Curve 思想)
        将 40-60 之间的犹豫分数向两端推，打破中性陷阱
        """
        # 以 50 为中心进行对称放大
        # 公式: new_score = 50 + (score - 50) * 1.5
        # 限制在 0-100 之间
        diff = score - 50
        amplified = 50 + (diff * 1.5) 
        return np.clip(amplified, 0, 100)

    def update(self, price: float, raw_probabilities: list) -> dict:
        # 1. 基础趋势分计算
        p_bear = raw_probabilities[0]
        p_bull = raw_probabilities[2]
        
        if (p_bear + p_bull) > 0:
            base_score = (p_bull / (p_bull + p_bear)) * 100
        else:
            base_score = 50.0

        # 2. 应用【方案二：动量修正】
        momentum_bias = self._calculate_momentum_bias(price)
        biased_score = base_score + momentum_bias

        # 3. 应用【方案一：非线性放大】
        final_score = self._amplify_score(biased_score)

        # 4. 判定模型倾向 (使用增强后的分数)
        if final_score >= self.bullish_threshold:
            model_bias = 'Bullish'
        elif final_score < 50.0:
            model_bias = 'Bearish'
        else:
            model_bias = 'Neutral'

        # 5. 状态机逻辑
        if self.current_state is None:
            self.current_state = 'Bullish' if final_score >= 50 else 'Bearish'
            self.entry_price = price
            return self._build_res(price, final_score, model_bias)

        # Level 1: 绝对重置 (使用增强分数)
        if final_score >= self.absolute_threshold:
            if self.current_state != 'Bullish':
                self.current_state = 'Bullish'
                self.entry_price = price
        elif final_score <= (100 - self.absolute_threshold):
            if self.current_state != 'Bearish':
                self.current_state = 'Bearish'
                self.entry_price = price

        # Level 2 & 3: 价格驱动与维持
        if self.current_state == 'Bullish':
            price_drop = (self.entry_price - price) / self.entry_price
            if (price < self.entry_price and model_bias == 'Bearish') or \
               (price_drop > self.price_break_pct and final_score < self.bullish_threshold):
                self.current_state = 'Bearish'
                self.entry_price = price
        
        elif self.current_state == 'Bearish':
            price_rise = (price - self.entry_price) / self.entry_price
            dist_from_anchor = (self.entry_price - price) / self.entry_price
            
            if (price > self.entry_price and model_bias == 'Bullish') or \
               (price_rise > self.price_break_pct and final_score >= 50.0) or \
               (dist_from_anchor > 0.10 and final_score >= self.bullish_threshold):
                self.current_state = 'Bullish'
                self.entry_price = price

        return self._build_res(price, final_score, model_bias)

    def _build_res(self, price, trend_score, model_bias):
        return {
            "final_trend": self.current_state,
            "trend_score": round(trend_score, 2),
            "entry_price": self.entry_price,
            "model_bias": model_bias,
            "is_state_locked": (model_bias != self.current_state if model_bias != 'Neutral' else False)
        }
