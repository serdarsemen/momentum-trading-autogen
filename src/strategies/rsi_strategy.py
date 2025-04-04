"""
RSI (Relative Strength Index) trading strategy implementation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def rsi_trading_strategy(
    df: pd.DataFrame,
    period: int = 14,
    oversold: int = 30,
    overbought: int = 70
) -> pd.DataFrame:
    """Implement RSI-based trading strategy."""
    signals = pd.DataFrame(index=df.index)
    signals['price'] = df['Close']
    signals['rsi'] = calculate_rsi(signals['price'], period)
    
    # Generate signals
    signals['signal'] = 0.0
    signals.loc[signals['rsi'] < oversold, 'signal'] = 1.0  # Buy
    signals.loc[signals['rsi'] > overbought, 'signal'] = 0.0  # Sell
    
    signals['positions'] = signals['signal'].diff()
    
    return signals