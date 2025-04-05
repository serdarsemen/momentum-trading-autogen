"""
Fair Value Gap (FVG) trading strategy implementation.
Identifies and trades based on Fair Value Gaps across different timeframes.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime

@dataclass
class FVG:
    """Represents a Fair Value Gap."""
    timestamp: datetime
    high: float
    low: float
    type: str  # 'bullish' or 'bearish'
    filled: bool = False
    fill_time: Optional[datetime] = None

def identify_fvg(
    df: pd.DataFrame,
    min_gap_size: float = 0.001  # Default 0.1% minimum gap size
) -> Tuple[List[FVG], List[FVG]]:
    """
    Identify bullish and bearish Fair Value Gaps in price action.
    
    Args:
        df: DataFrame with OHLC data
        min_gap_size: Minimum size of gap relative to price
        
    Returns:
        Tuple of (bullish_fvgs, bearish_fvgs)
    """
    bullish_fvgs = []
    bearish_fvgs = []
    
    # Convert DataFrame to numpy array for faster access
    low_values = df['Low'].to_numpy()
    high_values = df['High'].to_numpy()
    
    for i in range(1, len(df) - 1):
        # Get values for three consecutive candles
        prev_low = low_values[i-1]
        prev_high = high_values[i-1]
        next_low = low_values[i+1]
        next_high = high_values[i+1]
        current_time = df.index[i]
        
        # Check for bullish FVG (gap up)
        if prev_low > next_high:
            gap_size = (prev_low - next_high) / next_high
            if gap_size >= min_gap_size:
                fvg = FVG(
                    timestamp=current_time,
                    high=prev_low,
                    low=next_high,
                    type='bullish'
                )
                bullish_fvgs.append(fvg)
        
        # Check for bearish FVG (gap down)
        if next_low > prev_high:
            gap_size = (next_low - prev_high) / prev_high
            if gap_size >= min_gap_size:
                fvg = FVG(
                    timestamp=current_time,
                    high=next_low,
                    low=prev_high,
                    type='bearish'
                )
                bearish_fvgs.append(fvg)
    
    return bullish_fvgs, bearish_fvgs

def check_fvg_fill(
    fvg: FVG,
    current_low: float,
    current_high: float
) -> bool:
    """
    Check if a candle fills a Fair Value Gap.
    
    Args:
        fvg: FVG object to check
        current_low: Current candle's low price
        current_high: Current candle's high price
        
    Returns:
        Boolean indicating if FVG is filled
    """
    if fvg.type == 'bullish':
        return current_high >= fvg.high or current_low <= fvg.low
    else:  # bearish
        return current_low <= fvg.low or current_high >= fvg.high

def fvg_trading_strategy(
    df: pd.DataFrame,
    timeframe: str = '4h',
    min_gap_size: float = 0.001,
    max_active_fvgs: int = 3,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.03
) -> pd.DataFrame:
    """
    Implement FVG-based trading strategy.
    """
    # Adjust parameters based on timeframe
    if timeframe == '15m':
        min_gap_size = min(min_gap_size, 0.0005)
        max_active_fvgs = max(max_active_fvgs, 5)
        stop_loss_pct = min(stop_loss_pct, 0.01)
        take_profit_pct = min(take_profit_pct, 0.015)
    
    signals = pd.DataFrame(index=df.index)
    signals['price'] = df['Close']
    
    # Identify FVGs
    bullish_fvgs, bearish_fvgs = identify_fvg(df, min_gap_size)
    
    # Initialize columns
    signals['signal'] = 0.0
    signals['stop_loss'] = 0.0
    signals['take_profit'] = 0.0
    signals['active_fvgs'] = 0
    
    active_bullish = []
    active_bearish = []
    
    for i in range(len(df)):
        current_candle = df.iloc[i]
        current_close = current_candle['Close'].iloc[0] if isinstance(current_candle['Close'], pd.Series) else current_candle['Close']
        current_open = current_candle['Open'].iloc[0] if isinstance(current_candle['Open'], pd.Series) else current_candle['Open']
        current_low = current_candle['Low'].iloc[0] if isinstance(current_candle['Low'], pd.Series) else current_candle['Low']
        current_high = current_candle['High'].iloc[0] if isinstance(current_candle['High'], pd.Series) else current_candle['High']
        
        # Update active FVGs
        for fvg in active_bullish[:]:
            if check_fvg_fill(fvg, current_low, current_high):
                fvg.filled = True
                fvg.fill_time = current_candle.name
                active_bullish.remove(fvg)
        
        for fvg in active_bearish[:]:
            if check_fvg_fill(fvg, current_low, current_high):
                fvg.filled = True
                fvg.fill_time = current_candle.name
                active_bearish.remove(fvg)
        
        # Add new FVGs if capacity allows
        while (len(bullish_fvgs) > 0 and 
               len(active_bullish) < max_active_fvgs and 
               bullish_fvgs[0].timestamp <= current_candle.name):
            active_bullish.append(bullish_fvgs.pop(0))
            
        while (len(bearish_fvgs) > 0 and 
               len(active_bearish) < max_active_fvgs and 
               bearish_fvgs[0].timestamp <= current_candle.name):
            active_bearish.append(bearish_fvgs.pop(0))
        
        # Generate trading signals with timeframe-specific conditions
        if len(active_bullish) > 0:
            nearest_fvg = min(active_bullish, key=lambda x: abs(current_close - x.low))
            threshold = 1.005 if timeframe == '15m' else 1.01
            
            if ((current_close <= nearest_fvg.low * threshold) and
                (timeframe == '15m' and current_close > current_open or
                 timeframe != '15m')):
                signals.loc[current_candle.name, 'signal'] = 1.0
                signals.loc[current_candle.name, 'stop_loss'] = current_close * (1 - stop_loss_pct)
                signals.loc[current_candle.name, 'take_profit'] = current_close * (1 + take_profit_pct)
        
        if len(active_bearish) > 0:
            nearest_fvg = min(active_bearish, key=lambda x: abs(current_close - x.high))
            threshold = 0.995 if timeframe == '15m' else 0.99
            
            if ((current_close >= nearest_fvg.high * threshold) and
                (timeframe == '15m' and current_close < current_open or
                 timeframe != '15m')):
                signals.loc[current_candle.name, 'signal'] = -1.0
                signals.loc[current_candle.name, 'stop_loss'] = current_close * (1 + stop_loss_pct)
                signals.loc[current_candle.name, 'take_profit'] = current_close * (1 - take_profit_pct)
        
        signals.loc[current_candle.name, 'active_fvgs'] = len(active_bullish) + len(active_bearish)
    
    # Calculate positions
    signals['positions'] = signals['signal'].diff()
    
    return signals

def plot_fvg_strategy(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    bullish_fvgs: List[FVG],
    bearish_fvgs: List[FVG],
    title: str = 'FVG Trading Strategy',
    filename: Optional[str] = None
) -> plt.Figure:
    """
    Plot the FVG trading strategy with identified gaps.
    
    Args:
        df: Original OHLC DataFrame
        signals: DataFrame with signals from fvg_trading_strategy()
        bullish_fvgs: List of bullish FVGs
        bearish_fvgs: List of bearish FVGs
        title: Plot title
        filename: If provided, save plot to this file
        
    Returns:
        Matplotlib figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
    
    # Plot price
    ax1.plot(df.index, df['Close'], label='Price', color='black', alpha=0.7)
    
    # Plot FVGs
    for fvg in bullish_fvgs:
        time_proportion = (fvg.timestamp - df.index[0]).total_seconds() / (df.index[-1] - df.index[0]).total_seconds()
        rect = plt.Rectangle(
            (df.index[0] + (df.index[-1] - df.index[0]) * time_proportion, fvg.low),
            df.index[-1] - df.index[0],
            fvg.high - fvg.low,
            alpha=0.2,
            color='green',
            label='Bullish FVG' if fvg == bullish_fvgs[0] else ""
        )
        ax1.add_patch(rect)
    
    for fvg in bearish_fvgs:
        time_proportion = (fvg.timestamp - df.index[0]).total_seconds() / (df.index[-1] - df.index[0]).total_seconds()
        rect = plt.Rectangle(
            (df.index[0] + (df.index[-1] - df.index[0]) * time_proportion, fvg.low),
            df.index[-1] - df.index[0],
            fvg.high - fvg.low,
            alpha=0.2,
            color='red',
            label='Bearish FVG' if fvg == bearish_fvgs[0] else ""
        )
        ax1.add_patch(rect)
    
    # Plot signals
    buy_signals = signals[signals['positions'] == 1.0]
    sell_signals = signals[signals['positions'] == -1.0]
    
    ax1.scatter(
        buy_signals.index,
        df.loc[buy_signals.index, 'Close'],
        marker='^',
        color='g',
        s=100,
        label='Buy Signal'
    )
    
    ax1.scatter(
        sell_signals.index,
        df.loc[sell_signals.index, 'Close'],
        marker='v',
        color='r',
        s=100,
        label='Sell Signal'
    )
    
    # Plot active FVGs count
    ax2.plot(signals.index, signals['active_fvgs'], label='Active FVGs', color='blue')
    ax2.set_ylabel('Active FVGs')
    ax2.grid(True)
    
    ax1.set_title(title)
    ax1.set_ylabel('Price')
    ax1.grid(True)
    ax1.legend()
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
    
    return fig

def run_fvg_analysis(
    df: pd.DataFrame,
    timeframe: str = '4h',
    min_gap_size: float = 0.001,
    max_active_fvgs: int = 3,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.03,
    symbol: str = 'STOCK',
    save_results: bool = True,
    output_dir: str = '.'
) -> Dict[str, Any]:
    """
    Run a complete analysis of the FVG trading strategy.
    
    Args:
        df: DataFrame with OHLC data
        timeframe: Trading timeframe (e.g., '15m', '4h')
        min_gap_size: Minimum size of gap relative to price
        max_active_fvgs: Maximum number of active FVGs to track
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        symbol: Stock symbol for naming output files
        save_results: Whether to save results to files
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with analysis results
    """
    # Generate signals
    signals = fvg_trading_strategy(
        df,
        timeframe=timeframe,
        min_gap_size=min_gap_size,
        max_active_fvgs=max_active_fvgs,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct
    )
    
    # Identify FVGs for plotting
    bullish_fvgs, bearish_fvgs = identify_fvg(df, min_gap_size)
    
    # Compute returns
    from .momentum_trading_strategy import compute_returns
    final_return, cumulative_returns = compute_returns(signals)
    
    # Base filename for outputs
    base_filename = f"{symbol.lower()}_fvg_{timeframe}_{min_gap_size}_{max_active_fvgs}"
    
    results = {
        'signals': signals,
        'final_return': final_return,
        'cumulative_returns': cumulative_returns,
        'bullish_fvgs': bullish_fvgs,
        'bearish_fvgs': bearish_fvgs,
        'parameters': {
            'timeframe': timeframe,
            'min_gap_size': min_gap_size,
            'max_active_fvgs': max_active_fvgs,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'symbol': symbol
        }
    }
    
    if save_results:
        # Save signals to CSV
        csv_path = f"{output_dir}/{base_filename}.csv"
        signals.to_csv(csv_path)
        results['csv_path'] = csv_path
        
        # Plot and save strategy
        strategy_path = f"{output_dir}/{base_filename}.png"
        plot_fvg_strategy(
            df,
            signals,
            bullish_fvgs,
            bearish_fvgs,
            title=f'FVG Strategy {timeframe} ({symbol})',
            filename=strategy_path
        )
        results['strategy_plot_path'] = strategy_path
    
    return results






