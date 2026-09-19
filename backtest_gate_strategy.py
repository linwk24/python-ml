#!/usr/bin/env python3
"""
Gate.io 合约交易策略历史回测脚本

基于 Gate_OrderAPI_v.1.3.py 的交易逻辑：
- 对冲策略：同时持有多空仓位
- 根据信号调整仓位比例
- 止盈逻辑：未实现盈亏 > 手续费 * pnl_multiple
- 减仓逻辑：仓位超过阈值时自动减仓

用法：python backtest_gate_strategy.py --symbol BTC_USDT --start 2024-01-01 --end 2025-04-01
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

from models.enhanced_lstm import EnhancedLSTMPredictor
from services.model_meta import guard_in_sample_window
from config import get_all_klines, format_symbol, EXCHANGE


class Position:
    """仓位管理类"""
    def __init__(self):
        self.long_size: float = 0  # 多头仓位（合约数量）
        self.short_size: float = 0  # 空头仓位（合约数量）
        self.long_entry_price: float = 0  # 多头开仓均价
        self.short_entry_price: float = 0  # 空头开仓均价
        self.long_pnl_fee: float = 0  # 多头累计手续费
        self.short_pnl_fee: float = 0  # 空头累计手续费
        self.long_realised_pnl: float = 0  # 多头已实现盈亏
        self.short_realised_pnl: float = 0  # 空头已实现盈亏
    
    @property
    def total_size(self) -> float:
        """总仓位（合约数量）"""
        return self.long_size + self.short_size
    
    def get_unrealised_pnl(self, current_price: float, contract_value: float) -> float:
        """计算未实现盈亏"""
        long_pnl = 0
        short_pnl = 0
        
        if self.long_size > 0 and self.long_entry_price > 0:
            long_pnl = (current_price - self.long_entry_price) * self.long_size * contract_value
        
        if self.short_size > 0 and self.short_entry_price > 0:
            short_pnl = (self.short_entry_price - current_price) * self.short_size * contract_value
        
        return long_pnl + short_pnl
    
    def get_total_pnl_fee(self) -> float:
        """获取总手续费"""
        return self.long_pnl_fee + self.short_pnl_fee


class GateStrategyBacktester:
    """Gate.io 合约策略回测器"""
    
    def __init__(
        self,
        symbol: str,
        contract_value: float,
        take_threshold: float,
        pnl_multiple: float = 3.0,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.0005  # 手续费率 0.05%
    ):
        self.symbol = symbol
        self.contract_value = contract_value
        self.take_threshold = take_threshold
        self.pnl_multiple = pnl_multiple
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        
        # 账户状态
        self.capital = initial_capital
        self.position = Position()
        
        # 交易记录
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        # 中性(观望)信号计数：状态机现在会真的输出中性，单独统计便于评估信号质量
        self.neutral_signals: int = 0
    
    def calculate_contract_size(self, usdt_amount: float, price: float) -> int:
        """根据 USDT 金额计算合约数量"""
        return int(usdt_amount / (price * self.contract_value))
    
    def open_long(self, contracts: int, price: float, timestamp: datetime):
        """开多单"""
        if contracts <= 0:
            return
        
        # 计算手续费
        notional = contracts * price * self.contract_value
        fee = notional * self.fee_rate
        
        # 更新仓位（加权平均价）
        total_cost = self.position.long_entry_price * self.position.long_size + price * contracts
        self.position.long_size += contracts
        self.position.long_entry_price = total_cost / self.position.long_size if self.position.long_size > 0 else 0
        self.position.long_pnl_fee += fee
        
        # 扣除手续费
        self.capital -= fee
        
        self.trades.append({
            "timestamp": timestamp,
            "action": "OPEN_LONG",
            "contracts": contracts,
            "price": price,
            "fee": fee,
            "capital_after": self.capital
        })
    
    def open_short(self, contracts: int, price: float, timestamp: datetime):
        """开空单"""
        if contracts <= 0:
            return
        
        # 计算手续费
        notional = contracts * price * self.contract_value
        fee = notional * self.fee_rate
        
        # 更新仓位（加权平均价）
        total_cost = self.position.short_entry_price * self.position.short_size + price * contracts
        self.position.short_size += contracts
        self.position.short_entry_price = total_cost / self.position.short_size if self.position.short_size > 0 else 0
        self.position.short_pnl_fee += fee
        
        # 扣除手续费
        self.capital -= fee
        
        self.trades.append({
            "timestamp": timestamp,
            "action": "OPEN_SHORT",
            "contracts": contracts,
            "price": price,
            "fee": fee,
            "capital_after": self.capital
        })
    
    def close_long(self, contracts: int, price: float, timestamp: datetime):
        """平多单"""
        if contracts <= 0 or self.position.long_size <= 0:
            return
        
        contracts = min(contracts, int(self.position.long_size))
        
        # 计算盈亏
        pnl = (price - self.position.long_entry_price) * contracts * self.contract_value
        
        # 计算手续费
        notional = contracts * price * self.contract_value
        fee = notional * self.fee_rate
        
        # 更新仓位
        self.position.long_size -= contracts
        if self.position.long_size == 0:
            self.position.long_entry_price = 0
        
        # 更新资金
        self.capital += pnl - fee
        self.position.long_realised_pnl += pnl - fee
        
        self.trades.append({
            "timestamp": timestamp,
            "action": "CLOSE_LONG",
            "contracts": contracts,
            "price": price,
            "pnl": pnl,
            "fee": fee,
            "capital_after": self.capital
        })
    
    def close_short(self, contracts: int, price: float, timestamp: datetime):
        """平空单"""
        if contracts <= 0 or self.position.short_size <= 0:
            return
        
        contracts = min(contracts, int(self.position.short_size))
        
        # 计算盈亏
        pnl = (self.position.short_entry_price - price) * contracts * self.contract_value
        
        # 计算手续费
        notional = contracts * price * self.contract_value
        fee = notional * self.fee_rate
        
        # 更新仓位
        self.position.short_size -= contracts
        if self.position.short_size == 0:
            self.position.short_entry_price = 0
        
        # 更新资金
        self.capital += pnl - fee
        self.position.short_realised_pnl += pnl - fee
        
        self.trades.append({
            "timestamp": timestamp,
            "action": "CLOSE_SHORT",
            "contracts": contracts,
            "price": price,
            "pnl": pnl,
            "fee": fee,
            "capital_after": self.capital
        })
    
    def process_signal(self, signal: str, price: float, timestamp: datetime):
        """处理交易信号"""
        # 计算仓位倍数
        tot_size_usdt = self.position.total_size * price * self.contract_value
        multiple = 1.5 if tot_size_usdt > self.take_threshold else 1.3
        
        # 计算未实现盈亏
        unrealised_pnl = self.position.get_unrealised_pnl(price, self.contract_value)
        total_pnl_fee = self.position.get_total_pnl_fee()
        target_pnl = abs(total_pnl_fee) * self.pnl_multiple
        
        # 1. 止盈检查
        if unrealised_pnl > target_pnl and target_pnl > 0 and (tot_size_usdt * 0.8) > self.take_threshold:
            print(f"  [{timestamp}] 止盈触发: 未实现盈亏 {unrealised_pnl:.2f} > 目标 {target_pnl:.2f}")
            if self.position.long_size > 0:
                self.close_long(int(self.position.long_size), price, timestamp)
            if self.position.short_size > 0:
                self.close_short(int(self.position.short_size), price, timestamp)
            return
        
        # 2. 减仓检查
        if tot_size_usdt > self.take_threshold:
            short_reduction = int(self.position.short_size / (multiple * 2))
            long_reduction = int(self.position.long_size / (multiple * 2))
            
            if short_reduction > 0:
                self.close_short(short_reduction, price, timestamp)
            if long_reduction > 0:
                self.close_long(long_reduction, price, timestamp)
        
        # 3. 根据信号开仓
        # 中性(观望)不动仓位：既不新开仓也不强制平仓，只保留上面的止盈/减仓风控。
        # 注意状态机现在会真的输出"中性"（此前趋势被压成只有看涨/看跌两态）。
        if signal not in ("看涨", "看跌"):
            self.neutral_signals += 1
            return

        # 当没有仓位时，直接建立初始仓位
        if self.position.long_size == 0 and self.position.short_size == 0:
            # 初始仓位：用初始资金的 1% 开仓
            initial_contracts = max(1, int((self.initial_capital * 0.01) / (price * self.contract_value)))
            if signal == "看涨":
                self.open_long(initial_contracts, price, timestamp)
            elif signal == "看跌":
                self.open_short(initial_contracts, price, timestamp)
        else:
            # 有仓位时，根据信号调整仓位比例
            if signal == "看涨":
                if self.position.long_size < self.position.short_size:
                    contracts = int((self.position.short_size * multiple) - self.position.long_size)
                    if contracts > 0:
                        self.open_long(contracts, price, timestamp)
            elif signal == "看跌":
                if self.position.short_size < self.position.long_size:
                    contracts = int((self.position.long_size * multiple) - self.position.short_size)
                    if contracts > 0:
                        self.open_short(contracts, price, timestamp)
    
    def get_equity(self, price: float) -> float:
        """计算当前权益"""
        unrealised_pnl = self.position.get_unrealised_pnl(price, self.contract_value)
        return self.capital + unrealised_pnl


def fetch_kline_data(symbol: str, start_date: str, end_date: str, interval: str = "1h") -> pd.DataFrame:
    """获取历史 K 线数据（支持多交易所，统一输出 Binance 格式）"""
    print(f"正在从 {EXCHANGE} 获取 {symbol} 从 {start_date} 到 {end_date} 的历史数据...")
    
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    
    try:
        all_data = get_all_klines(symbol, interval, start_ts, end_ts)
    except Exception as e:
        print(f"获取数据失败: {e}")
        return pd.DataFrame()
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    
    print(f"共获取 {len(df)} 条 {interval} K线数据")
    return df


def run_backtest(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1h",
    contract_value: float = 0.0001,
    take_threshold: float = 100.0,
    pnl_multiple: float = 3.0,
    initial_capital: float = 10000.0,
    lookback: int = 120,
    output_file: str = None,
    allow_in_sample: bool = False
):
    """运行回测"""
    # 口径检查：回测区间与模型训练区间重叠时，结果含 in-sample 成分
    if not guard_in_sample_window(symbol, start_date, end_date, allow_in_sample):
        return False

    print(f"\n{'='*70}")
    print(f"Gate.io 合约策略历史回测")
    print(f"{'='*70}")
    print(f"交易对: {symbol}")
    print(f"时间范围: {start_date} 至 {end_date}")
    print(f"K线周期: {interval}")
    print(f"初始资金: ${initial_capital:,.2f}")
    print(f"合约价值: {contract_value}")
    print(f"仓位阈值: ${take_threshold}")
    print(f"止盈倍数: {pnl_multiple}")
    
    # 初始化模型
    predictor_symbol = symbol.replace("_", "")
    predictor = EnhancedLSTMPredictor(symbol=predictor_symbol)
    
    if not predictor.load_model():
        print(f"\n错误: 未找到 {predictor_symbol} 的已训练模型")
        print("请先训练模型")
        return
    
    # 获取历史数据
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    extended_start = (start_dt - timedelta(days=lookback * 2)).strftime("%Y-%m-%d")
    df = fetch_kline_data(symbol, extended_start, end_date, interval)
    
    if df.empty:
        print("未获取到有效数据")
        return
    
    # 初始化回测器
    backtester = GateStrategyBacktester(
        symbol=symbol,
        contract_value=contract_value,
        take_threshold=take_threshold,
        pnl_multiple=pnl_multiple,
        initial_capital=initial_capital
    )
    
    # 开始回测
    print(f"\n开始回测...")
    print(f"{'='*70}")
    
    start_idx = df.index.get_loc(start_dt) if start_dt in df.index else lookback
    
    for i in range(start_idx, len(df)):
        current_time = df.index[i]
        current_price = float(df.iloc[i]['close'])
        
        # 构建模型输入
        if i < lookback:
            continue
        
        klines = []
        for j in range(i - lookback, i):
            klines.append([
                float(df.index[j].timestamp() * 1000),
                float(df.iloc[j]['open']),
                float(df.iloc[j]['high']),
                float(df.iloc[j]['low']),
                float(df.iloc[j]['close']),
                float(df.iloc[j]['volume'])
            ])
        
        # 获取预测信号
        try:
            result = predictor.predict(klines)
            if "error" in result:
                continue
            
            signal = result["prediction"]["trend"]
            
            # 处理信号
            backtester.process_signal(signal, current_price, current_time)
            
            # 记录权益曲线
            equity = backtester.get_equity(current_price)
            backtester.equity_curve.append({
                "timestamp": current_time,
                "price": current_price,
                "equity": equity,
                "long_size": backtester.position.long_size,
                "short_size": backtester.position.short_size,
                "capital": backtester.capital
            })
            
        except Exception as e:
            continue
    
    # 输出结果
    if not backtester.equity_curve:
        print("未生成任何交易记录")
        return
    
    df_equity = pd.DataFrame(backtester.equity_curve)
    df_trades = pd.DataFrame(backtester.trades)
    
    # 打印交易统计
    print(f"\n{'='*70}")
    print(f"回测结果统计")
    print(f"{'='*70}")
    
    final_equity = df_equity['equity'].iloc[-1]
    total_return = (final_equity - initial_capital) / initial_capital * 100
    max_equity = df_equity['equity'].max()
    min_equity = df_equity['equity'].min()
    max_drawdown = (max_equity - min_equity) / max_equity * 100 if max_equity > 0 else 0
    
    print(f"初始资金: ${initial_capital:,.2f}")
    print(f"最终权益: ${final_equity:,.2f}")
    print(f"总收益率: {total_return:.2f}%")
    print(f"最大权益: ${max_equity:,.2f}")
    print(f"最小权益: ${min_equity:,.2f}")
    print(f"最大回撤: {max_drawdown:.2f}%")
    
    print(f"\n交易统计:")
    print(f"总交易次数: {len(df_trades)}")
    total_signals = len(df_equity)
    print(f"信号构成: 共 {total_signals} 根K线, 其中中性(观望) {backtester.neutral_signals} 次 "
          f"({backtester.neutral_signals / total_signals * 100:.1f}%)，其余为看涨/看跌")
    
    if len(df_trades) > 0:
        open_trades = df_trades[df_trades['action'].str.startswith('OPEN')]
        close_trades = df_trades[df_trades['action'].str.startswith('CLOSE')]
        
        print(f"开仓次数: {len(open_trades)}")
        print(f"平仓次数: {len(close_trades)}")
        
        if 'pnl' in df_trades.columns:
            pnl_trades = df_trades[df_trades['pnl'].notna()]
            if len(pnl_trades) > 0:
                winning_trades = pnl_trades[pnl_trades['pnl'] > 0]
                losing_trades = pnl_trades[pnl_trades['pnl'] < 0]
                print(f"盈利次数: {len(winning_trades)}")
                print(f"亏损次数: {len(losing_trades)}")
                print(f"总盈亏: ${pnl_trades['pnl'].sum():,.2f}")
                print(f"总手续费: ${df_trades['fee'].sum():,.2f}")
    
    print(f"\n最终仓位:")
    print(f"多头: {backtester.position.long_size} 合约")
    print(f"空头: {backtester.position.short_size} 合约")
    
    # 打印前10笔和后10笔交易
    if len(df_trades) > 0:
        print(f"\n{'='*70}")
        print(f"交易记录（前10笔）")
        print(f"{'='*70}")
        print(df_trades.head(10).to_string(index=False))
        
        if len(df_trades) > 10:
            print(f"\n{'='*70}")
            print(f"交易记录（后10笔）")
            print(f"{'='*70}")
            print(df_trades.tail(10).to_string(index=False))
    
    # 保存结果
    if output_file:
        df_equity.to_csv(output_file, index=False)
        trades_file = output_file.replace('.csv', '_trades.csv')
        df_trades.to_csv(trades_file, index=False)
        print(f"\n权益曲线已保存到: {output_file}")
        print(f"交易记录已保存到: {trades_file}")
    else:
        os.makedirs("data", exist_ok=True)
        equity_file = f"data/gate_backtest_{symbol}_{interval}_{start_date}_{end_date}.csv"
        trades_file = f"data/gate_backtest_{symbol}_{interval}_{start_date}_{end_date}_trades.csv"
        df_equity.to_csv(equity_file, index=False)
        df_trades.to_csv(trades_file, index=False)
        print(f"\n权益曲线已保存到: {equity_file}")
        print(f"交易记录已保存到: {trades_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gate.io 合约策略历史回测")
    parser.add_argument("--symbol", type=str, default="BTC_USDT", help="交易对 (默认: BTC_USDT)")
    parser.add_argument("--start", type=str, default="2024-01-01", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2025-04-01", help="结束日期 YYYY-MM-DD")
    parser.add_argument("--interval", type=str, default="1h", help="K线周期 (默认: 1h)")
    parser.add_argument("--contract-value", type=float, default=0.0001, help="合约价值 (默认: 0.0001)")
    parser.add_argument("--threshold", type=float, default=100.0, help="仓位阈值 USDT (默认: 100)")
    parser.add_argument("--pnl-multiple", type=float, default=3.0, help="止盈倍数 (默认: 3)")
    parser.add_argument("--capital", type=float, default=10000.0, help="初始资金 (默认: 10000)")
    parser.add_argument("--lookback", type=int, default=120, help="模型回看窗口 (默认: 120)")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--allow-in-sample", action="store_true",
                        help="允许回测区间与训练区间重叠（结果只能标注为 in-sample）")

    args = parser.parse_args()
    
    ok = run_backtest(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        interval=args.interval,
        contract_value=args.contract_value,
        take_threshold=args.threshold,
        pnl_multiple=args.pnl_multiple,
        initial_capital=args.capital,
        lookback=args.lookback,
        output_file=args.output,
        allow_in_sample=args.allow_in_sample
    )
    if ok is False:
        sys.exit(2)
