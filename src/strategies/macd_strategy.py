"""
MACD (Moving Average Convergence Divergence) trading strategy implementation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional

def calculate_macd(
    data: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD indicator components.
    
    Args:
        data: Price series
        fast_period: Period for fast EMA
        slow_period: Period for slow EMA
        signal_period: Period for signal line
        
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    # Calculate EMAs
    fast_ema = data.ewm(span=fast_period, adjust=False).mean()
    slow_ema = data.ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate Signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def macd_trading_strategy(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    histogram_threshold: float = 0.0
) -> pd.DataFrame:
    """
    Implement MACD-based trading strategy.
    
    Args:
        df: DataFrame with price data (must contain 'Close' column)
        fast_period: Period for fast EMA
        slow_period: Period for slow EMA
        signal_period: Period for signal line
        histogram_threshold: Minimum histogram value to generate signal
        
    Returns:
        DataFrame with signals and indicators
    """
    signals = pd.DataFrame(index=df.index)
    signals['price'] = df['Close']
    
    # Calculate MACD components
    macd_line, signal_line, histogram = calculate_macd(
        signals['price'],
        fast_period,
        slow_period,
        signal_period
    )
    
    # Store MACD components
    signals['macd_line'] = macd_line
    signals['signal_line'] = signal_line
    signals['histogram'] = histogram
    
    # Generate signals
    signals['signal'] = 0.0
    
    # Buy signal when histogram crosses above threshold
    signals.loc[signals['histogram'] > histogram_threshold, 'signal'] = 1.0
    
    # Sell signal when histogram crosses below negative threshold
    signals.loc[signals['histogram'] < -histogram_threshold, 'signal'] = 0.0
    
    # Calculate positions (1=buy, -1=sell, 0=hold)
    signals['positions'] = signals['signal'].diff()
    
    return signals

def plot_macd_strategy(
    signals: pd.DataFrame,
    title: str = 'MACD Trading Strategy',
    filename: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot MACD strategy components and signals.
    
    Args:
        signals: DataFrame with signals from macd_trading_strategy()
        title: Plot title
        filename: If provided, save plot to this file
        
    Returns:
        Tuple of (figure, axes)
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
    fig.suptitle(title)
    
    # Plot price
    ax1.plot(signals['price'], label='Price', color='black', alpha=0.7)
    
    # Plot buy/sell signals
    buy_signals = signals[signals['positions'] == 1.0]
    sell_signals = signals[signals['positions'] == -1.0]
    
    ax1.scatter(buy_signals.index, buy_signals['price'],
                marker='^', color='g', s=100, label='Buy Signal')
    ax1.scatter(sell_signals.index, sell_signals['price'],
                marker='v', color='r', s=100, label='Sell Signal')
    
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot MACD components
    ax2.plot(signals['macd_line'], label='MACD Line', color='blue')
    ax2.plot(signals['signal_line'], label='Signal Line', color='red')
    
    # Plot histogram
    ax2.bar(signals.index, signals['histogram'],
            label='Histogram', color='gray', alpha=0.3)
    
    # Add horizontal line at y=0
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    
    ax2.set_ylabel('MACD')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
    
    return fig, (ax1, ax2)

def run_macd_analysis(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    histogram_threshold: float = 0.0,
    symbol: str = 'STOCK',
    save_results: bool = True,
    output_dir: str = '.'
) -> Dict[str, Any]:
    """
    Run a complete analysis of the MACD trading strategy.
    
    Args:
        df: DataFrame with price data
        fast_period: Period for fast EMA
        slow_period: Period for slow EMA
        signal_period: Period for signal line
        histogram_threshold: Minimum histogram value for signals
        symbol: Stock symbol for naming output files
        save_results: Whether to save results to files
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with analysis results
    """
    # Generate signals
    signals = macd_trading_strategy(
        df,
        fast_period,
        slow_period,
        signal_period,
        histogram_threshold
    )
    
    # Compute returns
    from .momentum_trading_strategy import compute_returns
    final_return, cumulative_returns = compute_returns(signals)
    
    # Base filename for outputs
    base_filename = f"{symbol.lower()}_macd_strategy_{fast_period}_{slow_period}_{signal_period}"
    
    results = {
        'signals': signals,
        'final_return': final_return,
        'cumulative_returns': cumulative_returns,
        'parameters': {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'histogram_threshold': histogram_threshold,
            'symbol': symbol
        }
    }
    
    if save_results:
        # Save signals to CSV
        csv_path = f"{output_dir}/{base_filename}.csv"
        signals.to_csv(csv_path)
        results['csv_path'] = csv_path
        
        # Plot and save MACD strategy
        strategy_path = f"{output_dir}/{base_filename}.png"
        plot_macd_strategy(
            signals,
            title=f'MACD Strategy ({fast_period}, {slow_period}, {signal_period})',
            filename=strategy_path
        )
        results['strategy_plot_path'] = strategy_path
    
    return results