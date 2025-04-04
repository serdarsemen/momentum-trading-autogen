"""
Example of running an RSI trading strategy analysis.
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import download_stock_data
from src.strategies import rsi_trading_strategy, compute_returns

def run_rsi_analysis(
    symbol: str = "NVDA",
    start_date: str = "2024-01-01",
    period: int = 14,
    oversold: int = 30,
    overbought: int = 70
):
    df = download_stock_data(symbol, start_date)
    signals = rsi_trading_strategy(df, period, oversold, overbought)
    final_return, cum_returns = compute_returns(signals)
    return signals, final_return, cum_returns