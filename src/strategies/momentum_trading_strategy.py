"""
Core implementation of momentum trading strategies using moving averages.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Union, List, Optional


def calculate_moving_average(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate the moving average of a data series.
    
    Args:
        data: Input price data series
        window: Window size for the moving average
        
    Returns:
        Series containing the moving average values
    """
    return data.rolling(window=window, min_periods=1).mean()


def momentum_trading_strategy(
    df: pd.DataFrame, 
    short_window: int = 40, 
    long_window: int = 100
) -> pd.DataFrame:
    """
    Implement a simple moving average crossover trading strategy.
    
    Args:
        df: DataFrame with price data (must contain 'Close' column)
        short_window: Window size for the short-term moving average
        long_window: Window size for the long-term moving average
        
    Returns:
        DataFrame with the original data and additional columns for the strategy
    """
    signals = pd.DataFrame(index=df.index)
    signals['price'] = df['Close']
    
    # Calculate moving averages
    signals['short_mavg'] = calculate_moving_average(signals['price'], short_window)
    signals['long_mavg'] = calculate_moving_average(signals['price'], long_window)
    
    # Initialize signal column
    signals['signal'] = 0.0
    
    # Generate signals - proper pandas method using .loc
    # Buy signal (1) when short MA > long MA
    signals.loc[signals.index[short_window:], 'signal'] = np.where(
        signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 
        1.0, 
        0.0
    )
    
    # Calculate positions (1=buy, -1=sell, 0=hold)
    signals['positions'] = signals['signal'].diff()
    
    return signals


def compute_returns(signals: pd.DataFrame) -> Tuple[float, pd.Series]:
    """
    Compute returns for the trading strategy.
    
    Args:
        signals: DataFrame with signals from momentum_trading_strategy()
        
    Returns:
        Tuple of (final_return, cumulative_returns_series)
    """
    # Calculate daily returns
    signals['daily_returns'] = signals['price'].pct_change().fillna(0)
    
    # Calculate strategy returns (position at previous day * today's return)
    signals['strategy_returns'] = signals['signal'].shift(1).fillna(0) * signals['daily_returns']
    
    # Calculate cumulative returns
    signals['cumulative_strategy_return'] = (1 + signals['strategy_returns']).cumprod()
    
    # Calculate final return
    final_return = signals['cumulative_strategy_return'].iloc[-1] - 1
    
    return final_return, signals['cumulative_strategy_return']


def plot_trading_strategy(
    signals: pd.DataFrame, 
    title: str = 'Trading Strategy', 
    filename: Optional[str] = None
) -> plt.Figure:
    """
    Plot the trading strategy with moving averages.
    
    Args:
        signals: DataFrame with signals from momentum_trading_strategy()
        title: Plot title
        filename: If provided, save the plot to this file
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot price and moving averages
    ax.plot(signals['price'], label='Price')
    ax.plot(signals['short_mavg'], label=f'Short MA ({len(signals) - len(signals["short_mavg"].dropna())+1})')
    ax.plot(signals['long_mavg'], label=f'Long MA ({len(signals) - len(signals["long_mavg"].dropna())+1})')
    
    # Add title and legend
    ax.set_title(title)
    ax.legend()
    
    # Save the figure if a filename is provided
    if filename:
        plt.savefig(filename)
    
    return fig


def plot_buy_sell_signals(
    signals: pd.DataFrame, 
    title: str = 'Buy and Sell Signals', 
    filename: Optional[str] = None
) -> plt.Figure:
    """
    Plot the buy and sell signals on a price chart.
    
    Args:
        signals: DataFrame with signals from momentum_trading_strategy()
        title: Plot title
        filename: If provided, save the plot to this file
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot price
    ax.plot(signals['price'], label='Price')
    
    # Find buy and sell signals
    buy_signals = signals[signals['positions'] == 1.0]
    sell_signals = signals[signals['positions'] == -1.0]
    
    # Plot buy signals (green triangles)
    ax.scatter(
        buy_signals.index, 
        buy_signals['price'], 
        marker='^', 
        color='g', 
        s=100, 
        label='Buy Signal'
    )
    
    # Plot sell signals (red triangles)
    ax.scatter(
        sell_signals.index, 
        sell_signals['price'], 
        marker='v', 
        color='r', 
        s=100, 
        label='Sell Signal'
    )
    
    # Add title and legend
    ax.set_title(title)
    ax.legend()
    
    # Save the figure if a filename is provided
    if filename:
        plt.savefig(filename)
    
    return fig


def plot_cumulative_returns(
    cumulative_returns: pd.Series, 
    title: str = 'Cumulative Returns', 
    filename: Optional[str] = None
) -> plt.Figure:
    """
    Plot the cumulative returns of the strategy.
    
    Args:
        cumulative_returns: Series of cumulative returns
        title: Plot title
        filename: If provided, save the plot to this file
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot cumulative returns
    ax.plot(cumulative_returns, label='Cumulative Returns')
    
    # Add a horizontal line at y=1 (break-even point)
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.3)
    
    # Add title and legend
    ax.set_title(title)
    ax.legend()
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=min(0.8, cumulative_returns.min()))
    
    # Save the figure if a filename is provided
    if filename:
        plt.savefig(filename)
    
    return fig


def run_strategy_analysis(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
    symbol: str = 'STOCK',
    save_results: bool = True,
    output_dir: str = '.'
) -> Dict[str, Any]:
    """
    Run a complete analysis of the momentum trading strategy with the given parameters.
    
    Args:
        df: DataFrame with price data (must contain 'Close' column)
        short_window: Window size for the short-term moving average
        long_window: Window size for the long-term moving average
        symbol: Stock symbol for naming output files
        save_results: Whether to save results to files
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with the analysis results
    """
    # Generate signals
    signals = momentum_trading_strategy(df, short_window, long_window)
    
    # Compute returns
    final_return, cumulative_returns = compute_returns(signals)
    
    # Base filename for outputs
    base_filename = f"{symbol.lower()}_trading_strategy_{short_window}_{long_window}"
    
    results = {
        'signals': signals,
        'final_return': final_return,
        'cumulative_returns': cumulative_returns,
        'parameters': {
            'short_window': short_window,
            'long_window': long_window,
            'symbol': symbol
        }
    }
    
    if save_results:
        # Save signals to CSV
        csv_path = f"{output_dir}/{base_filename}.csv"
        signals.to_csv(csv_path)
        results['csv_path'] = csv_path
        
        # Plot and save trading strategy
        strategy_path = f"{output_dir}/{base_filename}.png"
        plot_trading_strategy(
            signals, 
            title=f'Trading Strategy with MA {short_window} and {long_window}',
            filename=strategy_path
        )
        results['strategy_plot_path'] = strategy_path
        
        # Plot and save buy/sell signals
        signals_path = f"{output_dir}/buy_sell_signals_{short_window}_{long_window}.png"
        plot_buy_sell_signals(
            signals, 
            title=f'Buy and Sell Signals with MA {short_window} and {long_window}',
            filename=signals_path
        )
        results['signals_plot_path'] = signals_path
        
        # Plot and save cumulative returns
        returns_path = f"{output_dir}/cumulative_returns_{short_window}_{long_window}.png"
        plot_cumulative_returns(
            cumulative_returns,
            title=f'Cumulative Returns with MA {short_window} and {long_window}',
            filename=returns_path
        )
        results['returns_plot_path'] = returns_path
    
    return results