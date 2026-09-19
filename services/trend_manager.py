"""
趋势状态机（去抖动 + 中性档）

## 为什么要有"中性"

模型是 3 分类（0 看跌 / 1 中性 / 2 看涨），但历史实现有两个问题把中性档彻底吃掉了：

1. **评分丢弃了中性概率**：``base_score = p_bull / (p_bull + p_bear)`` 把中性概率约掉再
   归一化。于是一个"强烈说不清方向"的预测 ``[0.05, 0.85, 0.10]`` 会算出 66.7 分，
   经 S 曲线放大后越过 70 分阈值，被判成 **看涨** —— 模型明明有 85% 概率说"中性"。
2. **状态机只有两态**：``current_state`` 非 Bullish 即 Bearish，导致 ``trend_code``
   永远只能是 0 或 2，接口、预测记录、回测、评估里从未出现过中性。
   样本外评估里"中性命中率 0.00%"、预测分布只有 0/2，正是这个原因。

现在：

- 方向分改为 ``50 + 50 * (p_bull - p_bear)``：三类概率都参与，模型犹豫时自然回到 50；
- 状态机扩为 **Bullish / Neutral / Bearish** 三态，中间区间是真正的"观望"档；
- 切换带**滞回**，避免在阈值附近抖动：
  进入方向需要越过各自的进入阈值(默认 70 / 30)，退出方向先回到 Neutral，
  而不是立刻反手做空/做多；只有绝对强度的信号(>=85 / <=15)才允许直接跨方向切换。

``entry_price`` 的含义：进入某个方向（或进入中性）时的价格锚点，用于价格破位判断。
"""

from typing import Any, Dict, List, Optional

from config import (
    TREND_ABSOLUTE_THRESHOLD,
    TREND_BEARISH_THRESHOLD,
    TREND_BULLISH_THRESHOLD,
    TREND_PRICE_BREAK_PCT,
)

# 状态 -> 对外 trend_code（与模型的三分类编码一致：0 看跌 / 1 中性 / 2 看涨）
STATE_TO_TREND_CODE = {"Bearish": 0, "Neutral": 1, "Bullish": 2}
TREND_CODE_TO_STATE = {code: state for state, code in STATE_TO_TREND_CODE.items()}
TREND_LABELS = {0: "看跌", 1: "中性", 2: "看涨"}


def clip(value: float, low: float, high: float) -> float:
    """把 value 限制在 [low, high]（替代 np.clip，去掉对 numpy 的依赖以便单测）"""
    return max(low, min(high, value))


class TrendManager:
    """三态趋势状态机：价格优先 + 信号增强 + 滞回

    核心逻辑：
    1. 方向分：``50 + 50*(p_bull - p_bear)``，再叠加价格动量修正与非线性放大；
    2. 模型倾向：>= bullish_threshold 看涨、<= bearish_threshold 看跌、中间中性；
    3. 状态切换：
       - Level 1 [绝对重置]：分数 >= absolute_threshold 或 <= 100-absolute_threshold，直接跨方向；
       - Level 2 [退出到中性]：方向支撑减弱（分数回到阈值内）→ 先回 Neutral；
       - Level 3 [方向确认]：分数越过进入阈值或价格破位 + 模型同向 → 进入该方向。
    """

    def __init__(
        self,
        bullish_threshold: Optional[float] = None,
        price_break_pct: Optional[float] = None,
        absolute_threshold: Optional[float] = None,
        bearish_threshold: Optional[float] = None,
    ):
        # 默认值统一来自 config（唯一事实来源），避免"两处各写一套阈值"
        self.bullish_threshold = (
            TREND_BULLISH_THRESHOLD if bullish_threshold is None else bullish_threshold
        )
        # 缺省与看涨阈值关于 50 对称
        self.bearish_threshold = (
            TREND_BEARISH_THRESHOLD
            if bearish_threshold is None
            else bearish_threshold
        )
        self.price_break_pct = (
            TREND_PRICE_BREAK_PCT if price_break_pct is None else price_break_pct
        )
        self.absolute_threshold = (
            TREND_ABSOLUTE_THRESHOLD if absolute_threshold is None else absolute_threshold
        )
        self.current_state: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.price_history: List[float] = []  # 用于计算动量

    # ------------------------------------------------------------------ 打分

    def _calculate_momentum_bias(self, current_price: float) -> float:
        """
        方案二：价格动量权重
        分析近期价格走势，为趋势分提供方向性修正
        """
        self.price_history.append(current_price)
        if len(self.price_history) > 24:  # 观察过去 24 个采样点
            self.price_history.pop(0)

        if len(self.price_history) < 10:
            return 0.0  # 数据不足，不修正

        start_price = self.price_history[0]
        momentum = (current_price - start_price) / start_price

        # 将动量映射到分数修正值 (-10 到 +10)：假设 2% 的涨幅即为强动量
        return clip(momentum / 0.02, -10, 10)

    def _amplify_score(self, score: float) -> float:
        """
        方案一：非线性分数放大 (S-Curve 思想)
        """
        diff = score - 50
        return clip(50 + (diff * 1.5), 0, 100)

    @staticmethod
    def directional_score_static(raw_probabilities: List[float]) -> float:
        """三类概率合成的方向分（0-100，50 表示模型无方向）。

        与旧的 ``p_bull/(p_bull+p_bear)`` 的关键差别：中性概率大时两者都小，
        分差随之变小，分数自然回到 50，而不会被归一化放大成方向信号。
        """
        p_bear = float(raw_probabilities[0])
        p_neutral = float(raw_probabilities[1])
        p_bull = float(raw_probabilities[2])

        total = p_bear + p_neutral + p_bull
        if total <= 0:
            return 50.0
        p_bear, p_bull = p_bear / total, p_bull / total

        return 50.0 + 50.0 * (p_bull - p_bear)

    def directional_score(self, raw_probabilities: List[float]) -> float:
        """实例方法，等价于 directional_score_static"""
        return self.directional_score_static(raw_probabilities)

    def _bias(self, score: float) -> str:
        if score >= self.bullish_threshold:
            return "Bullish"
        if score <= self.bearish_threshold:
            return "Bearish"
        return "Neutral"

    # ------------------------------------------------------------------ 状态机

    def _enter(self, state: str, price: float) -> None:
        self.current_state = state
        self.entry_price = price

    def update(self, price: float, raw_probabilities: List[float]) -> Dict[str, Any]:
        raw_score = self.directional_score(raw_probabilities)
        momentum_bias = self._calculate_momentum_bias(price)
        final_score = self._amplify_score(raw_score + momentum_bias)
        model_bias = self._bias(final_score)

        previous_state = self.current_state

        if self.current_state is None:
            # 首次：允许直接落地为中性（模型犹豫就是犹豫，不必强行选边）
            self._enter(model_bias, price)
            return self._build_res(final_score, raw_score, model_bias, previous_state)

        # Level 1: 绝对重置 —— 强度足够时允许直接跨方向
        if final_score >= self.absolute_threshold:
            if self.current_state != "Bullish":
                self._enter("Bullish", price)
        elif final_score <= (100.0 - self.absolute_threshold):
            if self.current_state != "Bearish":
                self._enter("Bearish", price)
        elif self.current_state == "Bullish":
            price_drop = (self.entry_price - price) / self.entry_price
            if model_bias == "Bearish" and price_drop > self.price_break_pct:
                # 模型已反手且价格实质破位 -> 直接反向
                self._enter("Bearish", price)
            elif final_score < self.bullish_threshold:
                # 看涨支撑减弱：先退到中性观望，不再立刻反手做空
                self._enter("Neutral", price)
        elif self.current_state == "Bearish":
            price_rise = (price - self.entry_price) / self.entry_price
            if model_bias == "Bullish" and price_rise > self.price_break_pct:
                self._enter("Bullish", price)
            elif final_score > self.bearish_threshold:
                # 看跌支撑减弱：先退到中性观望
                self._enter("Neutral", price)
        else:  # Neutral
            price_rise = (price - self.entry_price) / self.entry_price
            price_drop = (self.entry_price - price) / self.entry_price
            if model_bias == "Bullish" and (
                final_score >= self.bullish_threshold or price_rise > self.price_break_pct
            ):
                self._enter("Bullish", price)
            elif model_bias == "Bearish" and (
                final_score <= self.bearish_threshold or price_drop > self.price_break_pct
            ):
                self._enter("Bearish", price)
            else:
                # 仍无方向：锚点跟随最新价格，破位判断始终相对"最近一次观望"
                self.entry_price = price

        return self._build_res(final_score, raw_score, model_bias, previous_state)

    def _build_res(
        self,
        trend_score: float,
        directional_score: float,
        model_bias: str,
        previous_state: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "final_trend": self.current_state,
            "trend_code": STATE_TO_TREND_CODE[self.current_state],
            "trend_score": round(trend_score, 2),
            "directional_score": round(directional_score, 2),
            "entry_price": self.entry_price,
            "model_bias": model_bias,
            "state_changed": previous_state != self.current_state,
            "is_state_locked": (
                model_bias != self.current_state if model_bias != "Neutral" else False
            ),
        }
