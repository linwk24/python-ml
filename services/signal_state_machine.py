"""
信号状态机：模型信号 + 失效线 + 结构确认（设计稿落地）

## 设计目标

模型每根 K 线独立预测，方向翻转频繁 → 换手极高 → 手续费吃光毛收益
（实测 BTC 换手 2190 次，5bp 下毛 +138% 变净 −20%）。
本模块把"每根重新决策"改成"进入一个状态后维持，直到被**确认**打破"。

## 五态

    BULLISH ──分数衰减──> WEAK_BULLISH ──空头确认──> BEARISH
       │                       │                      │
       └──跌破失效线───────────┴──────────────────────┘
                        ↓
                     NEUTRAL  （空仓，等新信号）

    WEAK_BEARISH 为看跌方向的镜像。WEAK_* 表示"优势在减弱但尚未反转"，
    仓位仍维持原方向，只作为减仓/预警信号 —— 不立即翻向。

## 三个确认层（对应设计稿 ①②③）

1. **迟滞双阈值**：进入方向需要分数越过 `*_enter`，退出回到 `*_exit`，
   中间区间保持原状态（这才是"不要每根翻转"的核心）。
2. **连续确认**：方向切换需连续 `confirm_bars` 根满足条件，过滤单根假信号。
3. **价格失效线**：进入方向时记录 `signal_price`，跌破
   `signal_price * (1 - stop_pct)` 则该方向失效 → 转 NEUTRAL，并**锁住**
   直到模型给出新信号（防止当根立刻重入）。
   可选叠加移动失效线 `highest * (1 - trail_atr_mult * atr_ratio)`。

## 关键适配：概率阈值在本模型上不可用

设计稿写的是"P(up) >= 60% 进入看涨"。但本模型概率极平 —— 线上实测
`看跌 35.4 / 中性 31.19 / 看涨 33.41`，最大概率从未接近 60%。
旧 `trend_manager` 的 70/30 就是死在这里（方向输出恒为 0，准确率 32.23%）。

因此本模块以 **`directional_score`（50 + 50*(p_bull − p_bear)）** 为输入，
阈值可用 `calibrate_thresholds()` 按模型自身分数分布的分位数标定，
而不是写死一个与尺度脱节的常数。

## 因果性

`step()` 每根**已收盘** K 线调用一次，返回的状态用于**下一根**的仓位。
任何判断只用到当前及之前的 bar（含 pivot 的滞后确认，见 `_confirmed_pivots`）。
`tests/test_signal_state_machine.py::TestCausality` 用"篡改未来 bar、看历史状态是否变化"
钉住这一点 —— 本模块的第一版实现在回测里犯过用当根收盘价决定当根仓位的未来函数错误。
"""

from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, List, Optional, Tuple

# 状态 -> 建议仓位（WEAK_* 维持原方向，只预警不翻向）
STATE_POSITION = {
    "BEARISH": -1.0,
    "WEAK_BEARISH": -1.0,
    "NEUTRAL": 0.0,
    "WEAK_BULLISH": 1.0,
    "BULLISH": 1.0,
}
# 状态 -> 对外 trend_code（保持与模型三分类编码一致：0 看跌 / 1 中性 / 2 看涨）
STATE_TO_TREND_CODE = {
    "BEARISH": 0, "WEAK_BEARISH": 0, "NEUTRAL": 1,
    "WEAK_BULLISH": 2, "BULLISH": 2,
}
TREND_LABELS = {0: "看跌", 1: "中性", 2: "看涨"}
BULL_STATES = ("BULLISH", "WEAK_BULLISH")
BEAR_STATES = ("BEARISH", "WEAK_BEARISH")


@dataclass
class StateMachineParams:
    """全部阈值集中在此，便于回测扫描与持久化。

    默认值来自 `experiments/state_machine_pnl.py` 的口径；
    注意那里同时发现：**移动失效线（trail_atr_mult > 0）在这份数据上有害**
    （组合毛收益 70% → 33%），因此默认关闭。
    """
    # 迟滞双阈值（作用在 directional_score 上，50 = 完全无方向）
    bull_enter: float = 55.0
    bull_exit: float = 50.0
    bear_enter: float = 45.0
    bear_exit: float = 50.0
    # 连续确认根数
    confirm_bars: int = 3
    # 价格失效线：signal_price * (1 - stop_pct)；None/0 表示关闭
    stop_pct: Optional[float] = 0.03
    # 移动失效线：high_water * (1 - trail_atr_mult * atr_ratio)。0 = 关闭（实测有害）
    trail_atr_mult: float = 0.0
    # 结构确认
    structure_lookback: int = 20
    pivot_halfwidth: int = 3
    enable_structure: bool = True
    # 失效后是否必须等"新信号"才允许重入（防止当根立刻重入）
    latch_after_invalidation: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StateSnapshot:
    state: str = "NEUTRAL"
    previous_state: Optional[str] = None
    state_since_bar: int = 0
    bars_in_state: int = 0
    trend_code: int = 1
    position: float = 0.0
    score: float = 50.0
    signal_price: Optional[float] = None
    invalidation_price: Optional[float] = None
    high_water: Optional[float] = None
    low_water: Optional[float] = None
    invalidation_reason: Optional[str] = None
    structure: Optional[str] = None       # "HH_HL" / "LH_LL" / None
    confirm_progress: int = 0
    latched: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _is_pivot_high(highs: List[float], i: int, k: int) -> bool:
    lo, hi = i - k, i + k
    if lo < 0 or hi >= len(highs):
        return False
    return highs[i] == max(highs[lo:hi + 1])


def _is_pivot_low(lows: List[float], i: int, k: int) -> bool:
    lo, hi = i - k, i + k
    if lo < 0 or hi >= len(lows):
        return False
    return lows[i] == min(lows[lo:hi + 1])


class SignalStateMachine:
    """逐根推进的信号状态机。

    用法（严格按已收盘 K 线推进，返回的状态用于**下一根**）：

        sm = SignalStateMachine()
        for bar in closed_bars:
            snap = sm.step(score=..., close=..., high=..., low=..., atr=...)
        # snap.position 就是下一根应持有的仓位
    """

    def __init__(self, params: Optional[StateMachineParams] = None):
        self.p = params or StateMachineParams()
        self.bar_index = -1
        self._highs: List[float] = []
        self._lows: List[float] = []
        self.snap = StateSnapshot(state="NEUTRAL", state_since_bar=0)
        self._confirm_target: Optional[str] = None
        self._confirm_count = 0
        self._latched = False
        # 被失效否掉的那个方向；解锁要求出现**不同**的新信号（见 step 的锁存分支）
        self._latch_dir: Optional[str] = None
        self.invalidations = 0
        self.transitions = 0

    # ---------------------------------------------------------------- 结构

    def _confirmed_pivots(self, k: int) -> Tuple[List[float], List[float]]:
        """因果的 pivot 检测：第 i 根的 pivot 需要 i+k 根才能确认。

        只看「已确认」的 pivot（下标 <= len-1-k），不使用未来数据。
        """
        n = len(self._highs)
        ph, pl = [], []
        for i in range(n):
            if i + k > n - 1:
                break                       # 尚未确认，不能用
            if _is_pivot_high(self._highs, i, k):
                ph.append(self._highs[i])
            if _is_pivot_low(self._lows, i, k):
                pl.append(self._lows[i])
        return ph, pl

    def _structure(self) -> Optional[str]:
        """Higher High + Higher Low -> 'HH_HL'；Lower High + Lower Low -> 'LH_LL'"""
        if not self.p.enable_structure:
            return None
        ph, pl = self._confirmed_pivots(self.p.pivot_halfwidth)
        if len(ph) < 2 or len(pl) < 2:
            return None
        if ph[-1] > ph[-2] and pl[-1] > pl[-2]:
            return "HH_HL"
        if ph[-1] < ph[-2] and pl[-1] < pl[-2]:
            return "LH_LL"
        return None

    # ---------------------------------------------------------------- 进入/切换

    def _target_direction(self, score: float) -> Optional[str]:
        """迟滞：返回分数"想要"的方向（None = 保持现状）"""
        p = self.p
        if score >= p.bull_enter:
            return "BULLISH"
        if score <= p.bear_enter:
            return "BEARISH"
        return None

    def _enter(self, state: str, close: float, atr: Optional[float] = None) -> None:
        prev = self.snap.state
        if state != prev:
            self.transitions += 1
        self.snap.previous_state = prev
        self.snap.state = state
        self.snap.state_since_bar = self.bar_index
        self.snap.bars_in_state = 1
        self.snap.trend_code = STATE_TO_TREND_CODE[state]
        self.snap.position = STATE_POSITION[state]
        # 记录信号价格锚点（设计稿的 signal_price）
        self.snap.signal_price = close
        self.snap.high_water = close
        self.snap.low_water = close
        self.snap.invalidation_reason = None
        self.snap.confirm_progress = 0
        self._confirm_target = None
        self._confirm_count = 0
        self._update_invalidation(atr)

    def _update_invalidation(self, atr: Optional[float] = None) -> None:
        """按当前状态重算失效价。

        两档可叠加：
          · 固定档 `signal_price * (1 - stop_pct)`（设计稿的 70000 -> 69300）
          · 移动档 `high_water - trail_atr_mult * ATR`（设计稿的 highest - ATR*N）

        移动档**必须用 ATR 这种"绝对价格单位"的波动度量**。早期实现用
        `(high_water - low_water)/high_water` 当波动率，结果是持有期盈利越多、
        测得的"波动"越大、防线离价格越远 —— 方向完全反了（测试
        `test_trailing_line_follows_high_water` 抓到了这一点）。
        """
        p = self.p
        s = self.snap
        s.invalidation_price = None
        if s.state not in BULL_STATES + BEAR_STATES:
            return
        candidates = []
        if p.stop_pct:
            if s.state in BULL_STATES and s.signal_price:
                candidates.append(s.signal_price * (1 - p.stop_pct))
            elif s.state in BEAR_STATES and s.signal_price:
                candidates.append(s.signal_price * (1 + p.stop_pct))
        if p.trail_atr_mult and atr and atr > 0:
            if s.state in BULL_STATES and s.high_water:
                candidates.append(s.high_water - p.trail_atr_mult * atr)
            elif s.state in BEAR_STATES and s.low_water:
                candidates.append(s.low_water + p.trail_atr_mult * atr)
        if not candidates:
            return
        # 多头失效线取"更高"的那条（更贴近价格、更保守地保护利润）；空头反之
        s.invalidation_price = max(candidates) if s.state in BULL_STATES else min(candidates)

    def _check_invalidation(self, close: float) -> Optional[str]:
        s = self.snap
        if s.invalidation_price is None:
            return None
        if s.state in BULL_STATES and close < s.invalidation_price:
            return "close_below_invalidation"
        if s.state in BEAR_STATES and close > s.invalidation_price:
            return "close_above_invalidation"
        return None

    # ---------------------------------------------------------------- 主循环

    def step(
        self,
        score: float,
        close: float,
        high: Optional[float] = None,
        low: Optional[float] = None,
        atr: Optional[float] = None,
    ) -> StateSnapshot:
        """推进一根**已收盘** K 线，返回用于下一根的状态快照。

        **返回的是快照的副本**，不是内部对象本身。早期实现直接 `return self.snap`，
        导致调用方收集到的所有快照都指向同一个被反复改写的对象 ——
        回测里会静默拿到"最后一根的状态"，而不是每根当时的状态
        （`TestCausality::test_future_bars_do_not_change_history` 在 bar 0 就抓到了）。
        """
        self.bar_index += 1
        self._highs.append(float(high if high is not None else close))
        self._lows.append(float(low if low is not None else close))
        s = self.snap
        s.score = float(score)
        s.structure = self._structure()
        s.latched = self._latched

        # 持有中的状态：更新移动档水位
        if s.state in BULL_STATES:
            s.high_water = max(s.high_water or close, close)
            s.low_water = min(s.low_water or close, close)
        elif s.state in BEAR_STATES:
            s.high_water = max(s.high_water or close, close)
            s.low_water = min(s.low_water or close, close)

        # ── 第 3 层：价格失效线（优先级最高）──
        inv = self._check_invalidation(close)
        if inv:
            self.invalidations += 1
            self._enter("NEUTRAL", close, atr)
            s.invalidation_reason = inv
            s.reason = f"失效线触发（{inv}）：{s.previous_state} -> NEUTRAL"
            if self.p.latch_after_invalidation:
                self._latched = True
                # 记住被否掉的方向。若这里只记"当前状态"，失效后状态已是 NEUTRAL，
                # 下游算出的 cur_dir 为 None，任何方向信号都会被判成"新信号"而立刻解锁
                # —— 锁形同虚设（测试 test_latch_prevents_immediate_reentry 抓到过）。
                self._latch_dir = ("BULLISH" if s.previous_state in BULL_STATES
                                   else "BEARISH" if s.previous_state in BEAR_STATES
                                   else None)
            self._update_invalidation(atr)
            s.latched = self._latched
            return replace(s)

        # ── 第 1+2 层：迟滞 + 连续确认 ──
        want = self._target_direction(score)
        cur_dir = ("BULLISH" if s.state in BULL_STATES
                   else "BEARISH" if s.state in BEAR_STATES else None)

        # 失效锁：必须等到与"被否掉的方向"不同的新信号才解锁
        if self._latched:
            if want is not None and want != self._latch_dir:
                self._latched = False
                self._latch_dir = None
                s.latched = False        # 快照字段必须同步，否则对外仍显示"锁定中"
            else:
                s.latched = True
                s.bars_in_state += 1
                s.reason = "失效后锁定中，等待新信号"
                self._update_invalidation(atr)
                return replace(s)

        if want is None:
            # 分数回到中间带：不切换，但方向状态降级为 WEAK 作为预警
            if s.state == "BULLISH" and score < self.p.bull_exit:
                self._enter("WEAK_BULLISH", close, atr)
                s.reason = f"看涨优势减弱（score {score:.1f} < {self.p.bull_exit}）"
            elif s.state == "BEARISH" and score > self.p.bear_exit:
                self._enter("WEAK_BEARISH", close, atr)
                s.reason = f"看跌优势减弱（score {score:.1f} > {self.p.bear_exit}）"
            else:
                s.bars_in_state += 1
                s.reason = "分数在中间带，维持原状态"
                self._confirm_target = None
                self._confirm_count = 0
            self._update_invalidation(atr)
            return replace(s)

        # 已在目标方向（含 WEAK -> 强）：升级回强状态
        if want == cur_dir:
            self._confirm_target = None
            self._confirm_count = 0
            if s.state == "WEAK_BULLISH" and score >= self.p.bull_enter:
                self._enter("BULLISH", close, atr)
                s.reason = "看涨优势恢复"
            elif s.state == "WEAK_BEARISH" and score <= self.p.bear_enter:
                self._enter("BEARISH", close, atr)
                s.reason = "看跌优势恢复"
            else:
                s.bars_in_state += 1
                s.reason = f"维持 {s.state}"
            self._update_invalidation(atr)
            return replace(s)

        # 结构确认（额外门槛，可选）：反向切换要求结构也配合
        structure_ok = True
        if self.p.enable_structure and s.structure is not None:
            if want == "BEARISH" and s.structure == "HH_HL":
                structure_ok = False
            elif want == "BULLISH" and s.structure == "LH_LL":
                structure_ok = False

        # 连续确认计数
        if structure_ok:
            if self._confirm_target == want:
                self._confirm_count += 1
            else:
                self._confirm_target, self._confirm_count = want, 1
        else:
            self._confirm_target, self._confirm_count = want, 0

        s.confirm_progress = self._confirm_count
        if self._confirm_count >= self.p.confirm_bars:
            self._enter(want, close, atr)
            s.reason = (f"连续 {self.p.confirm_bars} 根确认切换 -> {want}"
                        + ("" if structure_ok else "（结构未配合）"))
        else:
            s.bars_in_state += 1
            s.reason = (f"等待确认 {self._confirm_count}/{self.p.confirm_bars} -> {want}"
                        + ("" if structure_ok else "（结构未配合，计数归零）"))
        self._update_invalidation(atr)
        return replace(s)

    # ---------------------------------------------------------------- 工具

    def state_dict(self) -> Dict[str, Any]:
        """可持久化的完整状态（供服务重启后续接）"""
        return {
            "params": self.p.to_dict(),
            "bar_index": self.bar_index,
            "snapshot": self.snap.to_dict(),
            "confirm_target": self._confirm_target,
            "confirm_count": self._confirm_count,
            "latched": self._latched,
            "latch_dir": self._latch_dir,
            "invalidations": self.invalidations,
            "transitions": self.transitions,
        }


def directional_score(probabilities: Any) -> float:
    """50 + 50*(p_bull - p_bear)。接受 dict{"看跌","中性","看涨"} 或长度 3 的序列。

    与 models/enhanced_lstm 的 `directional_score_static` 同一口径 —— 三个类概率都参与，
    模型犹豫时自然回到 50（而不是像 p_bull/(p_bull+p_bear) 那样把中性约掉）。
    """
    if isinstance(probabilities, dict):
        bear = float(probabilities.get("看跌", 0.0))
        bull = float(probabilities.get("看涨", 0.0))
    else:
        seq = list(probabilities)
        bear, bull = float(seq[0]), float(seq[2])
    return 50.0 + 50.0 * (bull - bear) / 100.0


def calibrate_thresholds(scores: List[float], percentile: float = 70.0) -> Dict[str, float]:
    """按模型自身分数分布标定迟滞阈值。

    为什么必须标定：本模型 directional_score 实测只在 43.6~58.0 之间，
    设计稿里"P(up) >= 60%"对应的分数区间根本不存在；写死 70/30 时状态机
    方向输出恒为 0。用分位数标定可保证"进入方向"的频率由模型自身分布决定。
    """
    if not scores:
        return {"bull_enter": 55.0, "bull_exit": 50.0,
                "bear_enter": 45.0, "bear_exit": 50.0}
    s = sorted(float(x) for x in scores)
    n = len(s)

    def pct(q: float) -> float:
        idx = min(n - 1, max(0, int(round(q / 100.0 * (n - 1)))))
        return s[idx]

    return {
        "bull_enter": pct(percentile),
        "bull_exit": pct(100.0 - percentile),
        "bear_enter": pct(100.0 - percentile),
        "bear_exit": pct(percentile),
    }


def replay(
    bars: List[Dict[str, Any]],
    score_by_index,
    params: Optional[StateMachineParams] = None,
    snapshot: Optional[StateSnapshot] = None,
) -> SignalStateMachine:
    """用一段历史 bar 重建状态机（服务重启/冷启动时用）。

    为什么需要重建而不是持久化进程内状态：状态机是**有状态**的，而服务进程会重启。
    与其把内部状态落盘（还得处理与 K 线数据不一致），不如每次从最近 N 根已收盘 K 线
    确定性重放 —— 只要输入相同，结果就相同，不依赖任何持久化。

    bars: 每项需含 close/high/low，按时间升序，**只含已收盘 K 线**
    score_by_index: 可调用，接收 bar 下标，返回该根**决策时**的 directional_score
                    （即用截至该根收盘的窗口算出的分数）
    """
    sm = SignalStateMachine(params)
    if snapshot is not None:
        sm.snap = snapshot
    for i, b in enumerate(bars):
        sm.step(
            score=float(score_by_index(i)),
            close=float(b["close"]),
            high=float(b.get("high", b["close"])),
            low=float(b.get("low", b["close"])),
            atr=b.get("atr"),
        )
    return sm
