"""技术指标计算模块"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional


class TechnicalIndicators:
    """技术指标计算类"""
    
    def __init__(self, df: pd.DataFrame):
        """
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        """
        self.df = df.copy()
        # 确保数值类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        self._close = self.df['close'].values
        self._high = self.df['high'].values
        self._low = self.df['low'].values
        self._open = self.df['open'].values
        self._volume = self.df['volume'].values
    
    def safe_series(self, data, name='series', default=0.0):
        """安全处理序列，处理 NaN 和 None"""
        import numpy as np
        if data is None:
            return pd.Series([default] * len(self.df), name=name)
        series = pd.Series(data, name=name)
        return series.fillna(default).replace([np.inf, -np.inf], default)
    
    def rsi(self, period: int = 14) -> np.ndarray:
        """RSI 指标"""
        delta = np.diff(self._close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.concatenate([[np.nan], 
            pd.Series(gain).ewm(span=period, adjust=False).mean().values])
        avg_loss = np.concatenate([[np.nan], 
            pd.Series(loss).ewm(span=period, adjust=False).mean().values])
        rs = avg_gain / np.where(avg_loss == 0, 0.001, avg_loss)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD"""
        exp1 = pd.Series(self._close).ewm(span=fast, adjust=False).mean()
        exp2 = pd.Series(self._close).ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_hist = macd_line - signal_line
        return macd_line.values, signal_line.values, macd_hist.values
    
    def bollinger_bands(self, period: int = 20, std: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """布林带"""
        sma = pd.Series(self._close).rolling(window=period).mean()
        rolling_std = pd.Series(self._close).rolling(window=period).std()
        upper = sma + (rolling_std * std)
        lower = sma - (rolling_std * std)
        return upper.values, sma.values, lower.values
    
    def adx(self, period: int = 14) -> np.ndarray:
        """ADX"""
        high, low, close = self._high, self._low, self._close
        tr = np.maximum(high[1:] - low[1:], 
                        np.maximum(np.abs(high[1:] - close[:-1]), 
                                   np.abs(low[1:] - close[:-1])))
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values / np.where(atr == 0, 0.001, atr)
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values / np.where(atr == 0, 0.001, atr)
        dx = 100 * np.abs(plus_di - minus_di) / np.where(plus_di + minus_di == 0, 0.001, plus_di + minus_di)
        adx_values = pd.Series(dx).ewm(span=period, adjust=False).mean().values
        
        return np.concatenate([[np.nan], adx_values]) if len(adx_values) < len(close) else adx_values
    
    def cci(self, period: int = 20) -> np.ndarray:
        """CCI"""
        tp = (self._high + self._low + self._close) / 3
        sma = pd.Series(tp).rolling(window=period).mean()
        mad = pd.Series(tp).rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        cci_values = (tp - sma) / (0.015 * mad)
        return cci_values.values
    
    def williams_r(self, period: int = 14) -> np.ndarray:
        """Williams %R"""
        highest_high = pd.Series(self._high).rolling(window=period).max()
        lowest_low = pd.Series(self._low).rolling(window=period).min()
        wr = -100 * (highest_high - self._close) / (highest_high - lowest_low + 0.001)
        return wr.values
    
    def stochastic(self, k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic"""
        lowest_low = pd.Series(self._low).rolling(window=k_period).min()
        highest_high = pd.Series(self._high).rolling(window=k_period).max()
        k = 100 * (self._close - lowest_low) / (highest_high - lowest_low + 0.001)
        d = k.rolling(window=d_period).mean()
        return k.values, d.values
    
    def mfi(self, period: int = 14) -> np.ndarray:
        """MFI"""
        typical_price = (self._high + self._low + self._close) / 3
        money_flow = typical_price * self._volume
        positive_flow = pd.Series(index=self.df.index, dtype=float)
        negative_flow = pd.Series(index=self.df.index, dtype=float)
        
        for i in range(1, len(typical_price)):
            if typical_price[i] > typical_price[i-1]:
                positive_flow.iloc[i] = money_flow[i]
                negative_flow.iloc[i] = 0
            else:
                positive_flow.iloc[i] = 0
                negative_flow.iloc[i] = money_flow[i]
        
        mf_ratio = positive_flow.rolling(window=period).sum() / (negative_flow.rolling(window=period).sum() + 0.001)
        mfi_values = 100 - (100 / (1 + mf_ratio))
        return mfi_values.values
    
    def ichimoku(self) -> Tuple[np.ndarray, np.ndarray]:
        """Ichimoku Cloud"""
        high9 = pd.Series(self._high).rolling(window=9).max()
        low9 = pd.Series(self._low).rolling(window=9).min()
        tenkan_sen = (high9 + low9) / 2
        
        high26 = pd.Series(self._high).rolling(window=26).max()
        low26 = pd.Series(self._low).rolling(window=26).min()
        kijun_sen = (high26 + low26) / 2
        
        return tenkan_sen.values, kijun_sen.values
    
    def moving_average(self, period: int = 20) -> np.ndarray:
        """移动平均线"""
        return pd.Series(self._close).rolling(window=period).mean().values