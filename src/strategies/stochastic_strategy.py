"""
Stochastic Oscillator trading strategy implementation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional

def calculate_stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator (%K and %D lines).
    
    Args:
        df: DataFrame with High, Low, Close prices
        k_period: Look-back period for %K
        d_period: Period for %D moving average
        smooth_k: Smoothing period for %K
        
    Returns:
        Tuple of (percent_k, percent_d)
    """
    # Calculate %K
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    
    # Raw %K
    k_raw = 100 * (df['Close'] - low_min) / (high_max - low_min)
    
    # Smooth %K
    percent_k = k_raw.rolling(window=smooth_k).mean()
    
    # Calculate %D
    percent_d = percent_k.rolling(window=d_period).mean()
    
    return percent_k, percent_d

def stochastic_trading_strategy(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
    oversold: int = 20,
    overbought: int = 80
) -> pd.DataFrame:
    """
    Implement Stochastic Oscillator-based trading strategy.
    
    Args:
        df: DataFrame with price data
        k_period: Look-back period for %K
        d_period: Period for %D moving average
        smooth_k: Smoothing period for %K
        oversold: Oversold threshold
        overbought: Overbought threshold
        
    Returns:
        DataFrame with signals and indicators
    """
    signals = pd.DataFrame(index=df.index)
    signals['price'] = df['Close']
    
    # Calculate Stochastic Oscillator
    percent_k, percent_d = calculate_stochastic(
        df,
        k_period,
        d_period,
        smooth_k
    )
    
    signals['percent_k'] = percent_k
    signals['percent_d'] = percent_d
    
    # Generate signals
    signals['signal'] = 0.0
    
    # Buy signal when both %K and %D are below oversold and %K crosses above %D
    buy_condition = (
        (percent_k < oversold) &
        (percent_d < oversold) &
        (percent_k > percent_d) &
        (percent_k.shift(1) <= percent_d.shift(1))
    )
    signals.loc[buy_condition, 'signal'] = 1.0
    
    # Sell signal when both %K and %D are above overbought and %K crosses below %D
    sell_condition = (
        (percent_k > overbought) &
        (percent_d > overbought) &
        (percent_k < percent_d) &
        (percent_k.shift(1) >= percent_d.shift(1))
    )
    signals.loc[sell_condition, 'signal'] = 0.0
    
    # Calculate positions
    signals['positions'] = signals['signal'].diff()
    
    return signals

def plot_stochastic_strategy(
    signals: pd.DataFrame,
    title: str = 'Stochastic Oscillator Strategy',
    filename: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot Stochastic strategy components and signals.
    
    Args:
        signals: DataFrame with signals from stochastic_trading_strategy()
        title: Plot title
        filename: If provided, save plot to this file
        
    Returns:
        Tuple of (figure, axes)
    """
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
    
    # Plot Stochastic Oscillator
    ax2.plot(signals['percent_k'], label='%K', color='blue')
    ax2.plot(signals['percent_d'], label='%D', color='red')
    
    # Add overbought/oversold lines
    ax2.axhline(y=80, color='r', linestyle='--', alpha=0.3)
    ax2.axhline(y=20, color='g', linestyle='--', alpha=0.3)
    
    ax2.set_ylabel('Stochastic')
    ax2.set_ylim(-5, 105)
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
    
    return fig, (ax1, ax2)

def run_stochastic_analysis(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
    oversold: int = 20,
    overbought: int = 80,
    symbol: str = 'STOCK',
    save_results: bool = True,
    output_dir: str = '.'
) -> Dict[str, Any]:
    """
    Run a complete analysis of the Stochastic Oscillator trading strategy.
    
    Args:
        df: DataFrame with price data
        k_period: Look-back period for %K
        d_period: Period for %D moving average
        smooth_k: Smoothing period for %K
        oversold: Oversold threshold
        overbought: Overbought threshold
        symbol: Stock symbol for naming output files
        save_results: Whether to save results to files
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with analysis results
    """
    # Generate signals
    signals = stochastic_trading_strategy(
        df,
        k_period,
        d_period,
        smooth_k,
        oversold,
        overbought
    )
    
    # Compute returns
    from .momentum_trading_strategy import compute_returns
    final_return, cumulative_returns = compute_returns(signals)
    
    # Base filename for outputs
    base_filename = f"{symbol.lower()}_stochastic_{k_period}_{d_period}_{smooth_k}"
    
    results = {
        'signals': signals,
        'final_return': final_return,
        'cumulative_returns': cumulative_returns,
        'parameters': {
            'k_period': k_period,
            'd_period': d_period,
            'smooth_k': smooth_k,
            'oversold': oversold,
            'overbought': overbought,
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
        plot_stochastic_strategy(
            signals,
            title=f'Stochastic Strategy ({k_period}, {d_period}, {smooth_k})',
            filename=strategy_path
        )
        results['strategy_plot_path'] = strategy_path
    
    return results